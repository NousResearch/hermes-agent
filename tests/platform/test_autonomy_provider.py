"""Triage providers and the rule that combines their answers (earned autonomy, phase 1).

No network: the real client is exercised against a stub server on 127.0.0.1 that answers
with the shapes the provider documents. Every way a provider can fail must come back as
"no answer" — which the hook turns into "ask a person" — never as a verdict.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from nova.autonomy import triage
from nova.autonomy.providers import ProviderError, provider_for
from nova.autonomy.providers.fake import FakeProvider
from nova.autonomy.providers.typesafe import (
    TriageProviderError, TypesafeProvider, build_request, parse_response, request_answers,
)
from nova.autonomy.questions import DEFAULT_SETS, QuestionSet, question_set_for
from nova.errors import SpecError

REPO = Path(__file__).resolve().parents[2]
EMAIL = question_set_for("send_external_email", "default")
#: What a provider is asked: the default set's provider questions, plus one Choice so every
#: answer shape the provider documents is exercised. (The default set's recipient is a fact
#: NOVA looks up itself; see the recipient tests at the end.)
TONE = {"id": "tone", "type": "choice", "prompt": "What is the tone of the message?",
        "options": ["friendly", "neutral", "hostile"], "allowed": ["friendly", "neutral"],
        "min_confidence": 0.90}
ASKED = QuestionSet.parse([q for q in DEFAULT_SETS["send_external_email"] if q["type"] != "recipient"] + [TONE])
QUESTIONS = ASKED.to_list()


def safe_wire_answers(questions=QUESTIONS):
    """A documented-shape response in which every answer is the safe one."""
    answers = {}
    for q in questions:
        if q["type"] == "noul":
            answers[q["id"]] = {"type": "noul", "noul": 0.01}
        elif q["type"] == "choice":
            pick = q["allowed"][0]
            answers[q["id"]] = {"type": "choice", "choice": pick, "confidence": 0.97,
                                "probabilities": {o: (0.99 if o == pick else 0.01 / (len(q["options"]) - 1))
                                                  for o in q["options"]}}
        else:
            n = len(q["levels"])
            answers[q["id"]] = {"type": "score", "score": 0.02, "confidence": 0.95,
                                "legend": {str(i): lvl for i, lvl in enumerate(q["levels"])},
                                "probabilities": {str(i): (0.98 if i == 0 else 0.02 / (n - 1)) for i in range(n)}}
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 300, "output_tokens": 40}}


# -- the request is the documented one ---------------------------------------------------


def test_every_question_goes_in_one_documented_request():
    body = build_request({"tool": "send_message"}, QUESTIONS)
    assert body["model"] == "jev-latest" and body["state"] == {"tool": "send_message"}
    wire = body["questions"]
    assert set(wire) == {q["id"] for q in QUESTIONS}
    assert wire["financial_commitment"] == {"type": "noul", "instructions": QUESTIONS[0]["instructions"]}
    assert wire["sensitivity"]["criteria"] == ["routine", "minor", "significant", "critical"]
    assert wire["tone"]["criteria"] == {"friendly": None, "neutral": None, "hostile": None}
    # Thresholds are ours, not the provider's: they never leave the environment.
    assert "block_above" not in json.dumps(wire) and "min_confidence" not in json.dumps(wire)


def test_a_documented_response_is_normalized():
    normalized, model, usage = parse_response(safe_wire_answers(), QUESTIONS)
    assert model == "jev-1.13.0" and usage == {"input_tokens": 300, "output_tokens": 40}
    assert normalized["financial_commitment"] == {"type": "noul", "p": 0.01}
    assert normalized["sensitivity"]["probabilities"][0] == 0.98
    assert normalized["tone"]["choice"] == "friendly"
    assert triage.combine(QUESTIONS, normalized)["verdict"] == triage.AUTO_OK


@pytest.mark.parametrize("break_it", [
    lambda b: b.pop("model"),
    lambda b: b.update(model=""),
    lambda b: b.pop("answers"),
    lambda b: b["answers"].pop("tone"),
    lambda b: b["answers"]["financial_commitment"].update(type="score"),
    lambda b: b["answers"]["financial_commitment"].update(noul=1.7),
    lambda b: b["answers"]["financial_commitment"].update(noul=float("nan")),
    lambda b: b["answers"]["financial_commitment"].update(noul=True),
    lambda b: b["answers"]["tone"].pop("confidence"),
    lambda b: b["answers"]["tone"].update(choice="sarcastic"),
    lambda b: b["answers"]["sensitivity"]["probabilities"].pop("3"),
], ids=["no-model", "blank-model", "no-answers", "missing-answer", "wrong-type", "p-above-1",
        "p-nan", "p-bool", "no-confidence", "unasked-option", "missing-level"])
def test_an_undocumented_response_is_refused(break_it):
    body = safe_wire_answers()
    break_it(body)
    with pytest.raises(TriageProviderError):
        parse_response(body, QUESTIONS)


# -- the real client, against a stub server ------------------------------------------------


class Stub:
    def __init__(self):
        self.status, self.body, self.delay, self.seen = 200, safe_wire_answers(), 0.0, []

    def __enter__(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                stub.seen.append({"auth": self.headers.get("Authorization"),
                                  "body": json.loads(self.rfile.read(length))})
                time.sleep(stub.delay)
                raw = json.dumps(stub.body).encode() if not isinstance(stub.body, bytes) else stub.body
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/systemone"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()


def test_the_client_sends_the_key_and_reads_the_answers():
    with Stub() as stub:
        answers = TypesafeProvider("k-123", endpoint=stub.url).ask({"tool": "x"}, ASKED.questions)
    assert stub.seen[0]["auth"] == "Bearer k-123"
    assert set(stub.seen[0]["body"]["questions"]) == {q["id"] for q in QUESTIONS}
    assert answers.provider == "typesafe" and answers.model_version == "jev-1.13.0"
    assert triage.combine(QUESTIONS, answers.normalized())["verdict"] == triage.AUTO_OK


@pytest.mark.parametrize("status", [401, 422, 429, 500, 529])
def test_any_error_status_is_no_answer(status):
    with Stub() as stub:
        stub.status = status
        with pytest.raises(ProviderError, match=str(status)):
            TypesafeProvider("k", endpoint=stub.url).ask({}, ASKED.questions)


def test_a_body_that_is_not_json_is_no_answer():
    with Stub() as stub:
        stub.body = b"<html>gateway error</html>"
        with pytest.raises(ProviderError, match="not JSON"):
            TypesafeProvider("k", endpoint=stub.url).ask({}, ASKED.questions)


def test_a_slow_provider_is_cut_off_at_the_deadline():
    with Stub() as stub:
        stub.delay = 3.0
        started = time.monotonic()
        with pytest.raises(ProviderError):
            TypesafeProvider("k", endpoint=stub.url, timeout=0.5).ask({}, ASKED.questions)
        assert time.monotonic() - started < 1.5, "the hook must not wait past its deadline"


def test_no_key_means_no_request(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with Stub() as stub:
        with pytest.raises(ProviderError, match="no API key"):
            TypesafeProvider(endpoint=stub.url).ask({}, ASKED.questions)
        assert stub.seen == []


def test_an_unreachable_provider_is_no_answer():
    with pytest.raises(TriageProviderError, match="could not be reached"):
        request_answers("k", {}, QUESTIONS, endpoint="http://127.0.0.1:9/v1/systemone", timeout=0.5)


# -- the rule ------------------------------------------------------------------------------


def answers(**overrides):
    base = FakeProvider().ask({}, ASKED.questions).normalized()
    base.update(overrides)
    return base


def failed_ids(result):
    return [f["id"] for f in result["failed"]]


def test_all_safe_answers_pass():
    assert triage.combine(QUESTIONS, answers()) == {"verdict": triage.AUTO_OK, "failed": []}


def test_a_noul_at_its_limit_escalates_and_just_below_passes():
    at = triage.combine(QUESTIONS, answers(financial_commitment={"type": "noul", "p": 0.10}))
    below = triage.combine(QUESTIONS, answers(financial_commitment={"type": "noul", "p": 0.0999}))
    assert at["verdict"] == triage.ESCALATE and failed_ids(at) == ["financial_commitment"]
    assert "at or above the limit" in at["failed"][0]["why"]
    assert below["verdict"] == triage.AUTO_OK


def test_a_choice_outside_allowed_escalates():
    result = triage.combine(QUESTIONS, answers(tone={
        "type": "choice", "choice": "hostile", "confidence": 0.99,
        "probabilities": {"friendly": 0.0, "neutral": 0.0, "hostile": 1.0}}))
    assert failed_ids(result) == ["tone"] and "not one of the allowed" in result["failed"][0]["why"]


def test_confidence_just_below_the_minimum_escalates():
    choice = answers()["tone"] | {"confidence": 0.8999}
    result = triage.combine(QUESTIONS, answers(tone=choice))
    assert failed_ids(result) == ["tone"] and "below the minimum" in result["failed"][0]["why"]


def test_a_score_above_the_allowed_level_escalates():
    result = triage.combine(QUESTIONS, answers(sensitivity={
        "type": "score", "confidence": 0.95, "probabilities": [0.05, 0.9, 0.05, 0.0]}))
    assert failed_ids(result) == ["sensitivity"] and "'minor'" in result["failed"][0]["why"]


def test_a_score_whose_mass_leaks_upward_escalates():
    """Likeliest level is allowed, but only 60% sure it is not higher: not safe enough."""
    result = triage.combine(QUESTIONS, answers(sensitivity={
        "type": "score", "confidence": 0.95, "probabilities": [0.6, 0.0, 0.0, 0.4]}))
    assert failed_ids(result) == ["sensitivity"]


@pytest.mark.parametrize("bad", [None, {}, {"type": "choice"}, {"type": "noul", "p": "0.0"}, {"type": "noul"}])
def test_a_missing_or_malformed_answer_escalates(bad):
    normalized = answers()
    if bad is None:
        normalized.pop("financial_commitment")
    else:
        normalized["financial_commitment"] = bad
    assert triage.combine(QUESTIONS, normalized)["verdict"] == triage.ESCALATE


def test_no_answers_and_no_questions_both_escalate():
    assert triage.combine(QUESTIONS, None)["failed"] == [{"id": "", "why": "provider_unavailable"}]
    assert triage.combine([], answers())["verdict"] == triage.ESCALATE


def test_several_failures_are_all_listed():
    result = triage.combine(QUESTIONS, answers(
        sensitive_data={"type": "noul", "p": 0.5}, injected_instructions={"type": "noul", "p": 0.06}))
    assert failed_ids(result) == ["sensitive_data", "injected_instructions"]


# -- the fake provider and the deadline ------------------------------------------------------


def test_the_fake_is_deterministic_and_records_what_it_was_sent():
    fake = FakeProvider({"tone": {"type": "choice", "choice": "hostile", "confidence": 0.5,
                                  "probabilities": {"friendly": 0.0, "neutral": 0.0, "hostile": 1.0}}})
    first = fake.ask({"tool": "send_message", "chars": 120}, ASKED.questions)
    second = fake.ask({"tool": "send_message", "chars": 120}, ASKED.questions)
    assert first == second and first.model_version == "fake-1"
    assert fake.requests[0]["state"] == {"tool": "send_message", "chars": 120}
    assert triage.combine(QUESTIONS, first.normalized())["verdict"] == triage.ESCALATE


def test_a_failing_fake_raises():
    with pytest.raises(ProviderError, match="down"):
        FakeProvider(fail="down").ask({}, ASKED.questions)


def test_the_deadline_returns_none_for_a_hang_or_a_crash():
    assert triage.within_deadline(lambda: time.sleep(2) or 1, 0.2) is None
    assert triage.within_deadline(lambda: 1 / 0, 1.0) is None
    assert triage.within_deadline(lambda: "ok", 1.0) == "ok"


def test_provider_names():
    assert provider_for("none") is None
    assert isinstance(provider_for("fake"), FakeProvider)
    assert isinstance(provider_for("typesafe", api_key="k"), TypesafeProvider)
    with pytest.raises(ValueError):
        provider_for("oracle")


# -- question sets ---------------------------------------------------------------------------


def test_the_default_email_set_is_the_specified_one():
    kinds = [q.type for q in EMAIL.questions]
    assert kinds.count("noul") == 5 and kinds.count("score") == 1 and kinds.count("recipient") == 1
    [recipient] = [q for q in EMAIL.questions if q.type == "recipient"]
    assert recipient.allowed == ("existing_customer", "internal"), "known contacts are not allowed by default"
    limits = {q.id: q.block_above for q in EMAIL.questions if q.type == "noul"}
    assert limits == {"financial_commitment": 0.10, "sensitive_data": 0.10, "promises_outcome": 0.20,
                      "professional_advice": 0.10, "injected_instructions": 0.05}
    assert all(q.min_confidence == 0.90 for q in EMAIL.questions if q.type in ("score", "choice"))


def test_a_tenant_can_raise_the_confidence_bar_on_the_default_set():
    strict = question_set_for("send_external_email", "default", min_confidence=0.97)
    assert {q.min_confidence for q in strict.questions if q.type in ("score", "choice")} == {0.97}


def test_an_action_without_a_default_set_must_write_its_own():
    with pytest.raises(SpecError, match="no default question set"):
        question_set_for("refund", "default")


@pytest.mark.parametrize("bad, words", [
    ({"id": "a", "type": "noul", "statement": "x", "block_above": 1.5}, "from 0 to 1"),
    ({"id": "a", "type": "noul", "statement": "", "block_above": 0.1}, "non-empty"),
    ({"id": "A b", "type": "noul", "statement": "x", "block_above": 0.1}, "lowercase"),
    ({"id": "a", "type": "noul", "statement": "x", "block_above": 0.1, "min_confidence": 0.9}, "unknown keys"),
    ({"id": "a", "type": "score", "rubric": "x", "levels": ["only"], "max_allowed_level": "only"}, "2 to 10"),
    ({"id": "a", "type": "score", "rubric": "x", "levels": ["a", "b"], "max_allowed_level": "c"}, "one of"),
    ({"id": "a", "type": "choice", "prompt": "x", "options": ["a", "b"], "allowed": ["c"]}, "subset"),
    ({"id": "a", "type": "choice", "prompt": "x", "options": ["a", "a"], "allowed": ["a"]}, "twice"),
    ({"id": "a", "type": "choice", "prompt": "x", "options": ["a", "b"], "allowed": ["a"],
      "min_confidence": -0.1}, "from 0 to 1"),
    ({"id": "a", "type": "vote", "prompt": "x"}, "noul, score, choice or recipient"),
])
def test_a_bad_question_is_refused_with_a_sentence(bad, words):
    with pytest.raises(SpecError, match=words):
        QuestionSet.parse([bad])


def test_a_set_cannot_reuse_an_id():
    q = DEFAULT_SETS["send_external_email"][0]
    with pytest.raises(SpecError, match="twice"):
        QuestionSet.parse([q, q])


# -- what ships into the runtime ------------------------------------------------------------


def test_the_rule_and_the_wire_client_run_with_only_the_standard_library(tmp_path):
    """Loaded by path, with site-packages and the repository off the import path."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import importlib.util, json, sys\n"
        "def load(name, path):\n"
        "    spec = importlib.util.spec_from_file_location(name, path)\n"
        "    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module\n"
        f"triage = load('_triage', {str(REPO / 'nova/autonomy/triage.py')!r})\n"
        f"wire = load('_triage_typesafe', {str(REPO / 'nova/autonomy/providers/typesafe.py')!r})\n"
        f"questions = {json.dumps(QUESTIONS)}\n"
        f"body = {json.dumps(safe_wire_answers())}\n"
        "normalized, model, _ = wire.parse_response(body, questions)\n"
        "print(triage.combine(questions, normalized)['verdict'], model)\n"
        "print(sorted(m for m in sys.modules if m.startswith('nova')))\n"
    )
    result = subprocess.run([sys.executable, "-I", "-S", str(script)], capture_output=True, text=True,
                            cwd=tmp_path, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["auto_ok jev-1.13.0", "[]"]
