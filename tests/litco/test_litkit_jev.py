"""``litkit_jev`` against a fake LitKit and a fake TypeSafe endpoint: the fan-out each document
gets, the cull rule at its boundaries, truncation, backoff on 429/529, the missing key, and
``firstPass`` on a proposed review run."""

from __future__ import annotations

import json
import logging
import threading

import httpx
import pytest

from litco.litkit import jev as J
from litco.litkit import tools as T
from litco.litkit.client import LitKitClient, LitKitConfig, set_default_client
from litco.litkit.context import TurnIdentity, bind_turn, reset_turn
from tests.litco._litkit_fake import HOST_SECRET, MATTER_ID, TOKEN, USER_ID, FakeLitKit, FakeReview, ndjson

M = MATTER_ID
KEY = "ts_test_key_123"
SECRET_TEXT = "The merger price is $41 per share; do not forward."


def _id(n: int) -> str:
    return f"00000000-0000-4000-8000-{n:012d}"


class FakeJev:
    """TypeSafe's /v1/systemone: answers every noul from ``probs[bates][question]`` (default 0.05)."""

    def __init__(self) -> None:
        self.requests: list = []
        self.probs: dict = {}
        self.script: list = []  # queued (status, body, headers) answered before the normal reply
        self.sleeps: list = []
        self._lock = threading.Lock()

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        with self._lock:
            self.requests.append({"headers": dict(request.headers), "body": body})
            if self.script:
                status, payload, headers = self.script.pop(0)
                return httpx.Response(status, json=payload, headers=headers)
        bates = body["state"]["document"].get("bates")
        table = self.probs.get(bates, {})
        answers = {qid: {"type": "noul", "noul": table.get(qid, 0.05)} for qid in body["questions"]}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                         "usage": {"input_tokens": 1000, "output_tokens": 20}})

    def client(self) -> J.JevClient:
        return J.JevClient(KEY, transport=httpx.MockTransport(self.handler), sleep=self.sleeps.append)


@pytest.fixture
def fake():
    server = FakeLitKit().start()
    yield server
    server.stop()


@pytest.fixture
def env(fake, tmp_path, monkeypatch):
    home = tmp_path / "matter"
    cwd = home / "shared"
    cwd.mkdir(parents=True)
    monkeypatch.setenv("LITCO_MATTER_HOME", str(home))
    client = LitKitClient(LitKitConfig(instance_url=fake.url, token=TOKEN, host_secret=HOST_SECRET, matter_id=M),
                          sleep=lambda s: None)
    set_default_client(client)
    token = bind_turn(TurnIdentity(turn_id="turn_1", matter_id=M, acting_user=USER_ID, cwd=cwd))
    yield {"cwd": cwd}
    reset_turn(token)
    set_default_client(None)
    J.set_default_jev(None, override=False)
    client.close()


@pytest.fixture
def jev(env):
    fake_jev = FakeJev()
    J.set_default_jev(fake_jev.client())
    return fake_jev


def call(name: str, /, **args):
    return json.loads(T.HANDLERS[name](args))


def export_route(fake, docs):
    """docs: {id: (bates, text)}; ids absent from docs come back not_found."""
    def handler(req):
        rows = []
        for doc_id in req.json()["docIds"]:
            if doc_id in docs:
                bates, text = docs[doc_id]
                rows.append({"docId": doc_id, "batesStart": bates, "custodian": "Doe, Jane", "date": "2024-03-01",
                             "text": text})
            else:
                rows.append({"docId": doc_id, "error": "not_found"})
        return ndjson(rows)
    fake.route("POST", rf"/api/matters/{M}/export/text", handler)


CRITERIA = [{"title": "Part 11, item 1: pricing", "description": "Communications about list or net prices."},
            {"title": "Part 11, item 2: rebates"}]


# ---------------------------------------------------------------------------
# fan-out and the cull rule end to end
# ---------------------------------------------------------------------------

def test_screen_sends_one_fanned_out_request_per_document_and_culls_in_code(fake, jev):
    export_route(fake, {_id(1): ("ABC001", "pricing email"), _id(2): ("ABC002", "lunch order"),
                        _id(3): ("ABC003", "note to counsel")})
    jev.probs = {"ABC001": {"responsive": 0.91, "c_0": 0.88, "c_1": 0.10},
                 "ABC002": {"responsive": 0.02, "c_0": 0.01, "c_1": 0.03, "privileged_signal": 0.02, "junk": 0.9},
                 "ABC003": {"responsive": 0.03, "c_0": 0.02, "c_1": 0.01, "privileged_signal": 0.75}}
    out = call("litkit_jev", action="screen", documentIds=[_id(1), _id(2), _id(3)], criteria=CRITERIA)

    assert (out["screened"], out["read_in_full"], out["set_aside"]) == (3, 2, 1)
    assert out["summary"].startswith("Jev set aside 1 of 3; 2 read in full")
    rows = {r["bates"]: r for r in out["rows"]}
    assert rows["ABC002"]["decision"] == "set_aside"
    assert rows["ABC003"]["decision"] == "read_in_full"  # a privilege signal alone keeps it in
    assert rows["ABC001"]["top"][0] == ["c_0", 0.88]
    assert out["criteria"] == {"c_0": CRITERIA[0]["title"], "c_1": CRITERIA[1]["title"]}

    assert len(jev.requests) == 3
    for sent in jev.requests:
        assert sent["headers"]["authorization"] == f"Bearer {KEY}"
        body = sent["body"]
        assert body["model"] == "jev-latest"
        assert set(body["questions"]) == {"responsive", "c_0", "c_1", "privileged_signal", "junk"}
        assert {q["type"] for q in body["questions"].values()} == {"noul"}
        # Criteria travel in structured instruction fields; the document sits under a named key.
        assert [r["title"] for r in body["questions"]["responsive"]["instructions"]["requests"]] == \
            [c["title"] for c in CRITERIA]
        assert body["questions"]["c_1"]["instructions"]["request"] == {"title": CRITERIA[1]["title"]}
        assert set(body["state"]) == {"document"} and body["state"]["document"]["custodian"] == "Doe, Jane"

    # Text came through LitKit's ACL-scoped export, as the acting lawyer.
    export = fake.calls("POST", rf"/api/matters/{M}/export/text")
    assert len(export) == 1 and export[0].headers["x-litkit-acting-user"] == USER_ID


def test_screen_saves_every_probability_and_reports_cost(fake, jev, env):
    export_route(fake, {_id(1): ("ABC001", "x")})
    out = call("litkit_jev", action="screen", documentIds=[_id(1)], criteria=["Pricing"])
    saved = json.loads((env["cwd"] / out["saved"]).read_text())
    assert saved["documents"][0]["probabilities"] == {"responsive": 0.05, "privileged_signal": 0.05, "junk": 0.05,
                                                      "c_0": 0.05}
    assert "_probs" not in saved["documents"][0] and "_probs" not in out["rows"][0]
    assert out["usage"] == {"inputTokens": 1000, "estCostUsd": round(1000 * 0.042 / 1e6, 4)}


@pytest.mark.parametrize("probs, decision, uncertain", [
    ({"responsive": 0.15, "c_0": 0.15, "privileged_signal": 0.30}, "set_aside", True),     # every value on its line
    ({"responsive": 0.04, "c_0": 0.04, "privileged_signal": 0.10}, "set_aside", False),    # clear
    ({"responsive": 0.1501, "c_0": 0.0, "privileged_signal": 0.0}, "read_in_full", True),  # just over low
    ({"responsive": 0.0, "c_0": 0.16, "privileged_signal": 0.0}, "read_in_full", True),    # one criterion over
    ({"responsive": 0.0, "c_0": 0.0, "privileged_signal": 0.31}, "read_in_full", True),    # privilege signal over
    ({"responsive": 0.0, "c_0": 0.0, "privileged_signal": 0.41}, "read_in_full", False),   # past the band
    ({"responsive": 0.26, "c_0": 0.0, "privileged_signal": 0.0}, "read_in_full", False),
    ({"responsive": 0.06, "c_0": 0.0, "privileged_signal": 0.0}, "set_aside", True),       # within 0.1 below low
])
def test_cull_rule_boundaries(probs, decision, uncertain):
    verdict = J.decide({"junk": 0.99, **probs}, ["c_0"], {"low": 0.15, "priv": 0.30})
    assert verdict == {"decision": decision, "uncertain": uncertain}


def test_custom_thresholds_move_the_line(fake, jev):
    export_route(fake, {_id(1): ("ABC001", "x")})
    jev.probs = {"ABC001": {"responsive": 0.2, "c_0": 0.2, "privileged_signal": 0.2}}
    default = call("litkit_jev", action="screen", documentIds=[_id(1)], criteria=["Pricing"])
    wider = call("litkit_jev", action="screen", documentIds=[_id(1)], criteria=["Pricing"],
                 thresholds={"low": 0.25, "priv": 0.25})
    assert default["rows"][0]["decision"] == "read_in_full"
    assert wider["rows"][0]["decision"] == "set_aside" and wider["thresholds"] == {"low": 0.25, "priv": 0.25}


def test_unseen_documents_are_never_set_aside(fake, jev):
    export_route(fake, {_id(1): ("ABC001", "   "), _id(2): ("ABC002", "text")})
    jev.script = [(422, {"detail": [{"loc": ["questions"], "msg": "bad", "input": SECRET_TEXT}]}, {})]
    out = call("litkit_jev", action="screen", documentIds=[_id(1), _id(2), _id(3)], criteria=["Pricing"])
    rows = {r["docId"]: r for r in out["rows"]}
    assert rows[_id(1)] == {"docId": _id(1), "bates": "ABC001", "decision": "read_in_full", "noText": True}
    assert rows[_id(2)]["decision"] == "read_in_full" and "422" in rows[_id(2)]["error"]
    assert rows[_id(3)]["decision"] == "not_found"
    assert (out["set_aside"], out["no_text"], out["errors"], out["not_found"]) == (0, 1, 1, 1)
    assert len(jev.requests) == 1  # nothing is sent for a document with no text


# ---------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------

def test_long_text_is_cut_head_and_tail_to_fit_and_flagged(fake, jev):
    text = "HEAD-START " + "a" * 300_000 + " TAIL-END"
    export_route(fake, {_id(1): ("ABC001", text), _id(2): ("ABC002", "short")})
    out = call("litkit_jev", action="screen", documentIds=[_id(1), _id(2)], criteria=CRITERIA)
    rows = {r["bates"]: r for r in out["rows"]}
    assert rows["ABC001"]["truncated"] is True and "truncated" not in rows["ABC002"]
    assert out["truncated"] == 1
    sent = next(r["body"] for r in jev.requests if r["body"]["state"]["document"]["bates"] == "ABC001")
    cut = sent["state"]["document"]["text"]
    assert cut.startswith("HEAD-START") and cut.endswith("TAIL-END") and "characters omitted" in cut
    longest = max(J.estimate_tokens(q) for q in sent["questions"].values())
    assert J.estimate_tokens(sent["state"]) + longest <= J.STATE_TOKEN_BUDGET


def test_litkit_truncation_is_flagged_too(fake, jev):
    def handler(req):
        return ndjson([{"docId": _id(1), "batesStart": "ABC001", "text": "short", "truncated": True,
                        "fullLength": 900_000}])
    fake.route("POST", rf"/api/matters/{M}/export/text", handler)
    out = call("litkit_jev", action="screen", documentIds=[_id(1)], criteria=["Pricing"])
    assert out["rows"][0]["truncated"] is True


def test_a_length_refusal_is_retried_once_with_half_the_room(fake, jev):
    export_route(fake, {_id(1): ("ABC001", "b" * 200_000)})
    jev.script = [(422, {"detail": "state plus longest question exceeds 32768 tokens"}, {})]
    out = call("litkit_jev", action="screen", documentIds=[_id(1)], criteria=["Pricing"])
    assert out["rows"][0]["decision"] == "set_aside" and out["rows"][0]["truncated"] is True
    first, second = (len(r["body"]["state"]["document"]["text"]) for r in jev.requests)
    assert second < first * 0.6


def test_questions_that_leave_no_room_for_the_document_are_refused():
    huge = {"q": {"type": "noul", "instructions": "x" * (J.STATE_TOKEN_BUDGET * J.CHARS_PER_TOKEN)}}
    with pytest.raises(ValueError, match="too little room"):
        J.document_state({}, "text", huge)


def test_head_tail_leaves_short_text_alone():
    assert J.head_tail("short", 100) == ("short", False)


# ---------------------------------------------------------------------------
# backoff
# ---------------------------------------------------------------------------

def test_429_and_529_back_off_honoring_retry_after(fake, jev):
    export_route(fake, {_id(1): ("ABC001", "x")})
    jev.script = [(429, {"error": "rate"}, {"retry-after": "2"}), (529, {"error": "overloaded"}, {})]
    out = call("litkit_jev", action="screen", documentIds=[_id(1)], criteria=["Pricing"])
    assert out["rows"][0]["decision"] == "set_aside"
    assert len(jev.requests) == 3
    assert jev.sleeps[0] == 2.0
    assert 1.0 <= jev.sleeps[1] <= 2.0  # 2**1 with jitter in [0.5, 1)


def test_backoff_gives_up_after_the_last_attempt_and_the_document_is_read(fake, jev):
    export_route(fake, {_id(1): ("ABC001", "x")})
    jev.script = [(429, {}, {"retry-after": "1"})] * J.MAX_ATTEMPTS
    out = call("litkit_jev", action="screen", documentIds=[_id(1)], criteria=["Pricing"])
    assert len(jev.requests) == J.MAX_ATTEMPTS and len(jev.sleeps) == J.MAX_ATTEMPTS - 1
    assert out["rows"][0]["decision"] == "read_in_full" and "429" in out["rows"][0]["error"]


def test_401_is_not_retried():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(401, json={"error": "bad key"})
    client = J.JevClient(KEY, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(J.JevError, match="refused this host's TypeSafe key"):
        client.evaluate("x", {"q": {"type": "noul", "instructions": "?"}})
    assert len(seen) == 1


def test_validation_errors_never_echo_the_input():
    def handler(request):
        return httpx.Response(422, json={"detail": [{"loc": ["state"], "msg": "bad", "input": SECRET_TEXT}]})
    client = J.JevClient(KEY, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(J.JevError) as info:
        client.evaluate(SECRET_TEXT, {"q": {"type": "noul", "instructions": "?"}})
    assert SECRET_TEXT not in json.dumps(info.value.to_dict())


def test_document_text_is_never_logged(fake, jev, caplog):
    export_route(fake, {_id(1): ("ABC001", SECRET_TEXT)})
    jev.script = [(429, {}, {"retry-after": "0"}), (500, {}, {})]
    with caplog.at_level(logging.DEBUG):
        call("litkit_jev", action="screen", documentIds=[_id(1)], criteria=["Pricing"])
    assert "merger price" not in caplog.text


# ---------------------------------------------------------------------------
# configuration, caps, scope
# ---------------------------------------------------------------------------

def test_missing_key_is_a_plain_result_not_a_traceback(fake, env, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    J.set_default_jev(None, override=False)
    for action in ("screen", "ask"):
        out = call("litkit_jev", action=action, documentIds=[_id(1)], criteria=["Pricing"])
        assert out["configured"] is False and out["error"].startswith("Jev not configured on this host")
    assert fake.requests == []


def test_key_comes_from_the_host_env(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.5}}})
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_from_env")
    client = J.JevClient.from_env(transport=httpx.MockTransport(handler))
    client.evaluate("x", {"q": {"type": "noul", "instructions": "?"}})
    assert seen == ["Bearer ts_from_env"] and "ts_from_env" not in repr(client)
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert J.JevClient.from_env() is None


def test_over_200_documents_points_to_a_review_run_with_first_pass(fake, jev):
    out = call("litkit_jev", action="screen", documentIds=[_id(n) for n in range(1, 202)], criteria=["Pricing"])
    assert "201 documents" in out["error"] and "firstPass" in out["error"]
    assert fake.requests == [] and jev.requests == []


def test_work_set_scope_and_criteria_set(fake, jev):
    review = FakeReview(fake)
    set_id = call("litkit_review", action="criteria", criteriaAction="create", name="Part 11",
                  criteria=CRITERIA)["set"]["id"]
    ws = _id(900)
    fake.route("GET", rf"/api/work-sets/{ws}", {"id": ws, "matterId": M, "docIds": [_id(1), _id(2)]})
    export_route(fake, {_id(1): ("ABC001", "x"), _id(2): ("ABC002", "y")})
    out = call("litkit_jev", action="screen", workSetId=ws, criteriaSetId=set_id)
    assert out["screened"] == 2 and list(out["criteria"]) == ["c_0", "c_1"]
    assert review.sets[set_id]["name"] == "Part 11"


def test_bates_scope_resolves_each_number(fake, jev):
    fake.route("GET", rf"/api/matters/{M}/bates-resolve",
               lambda r: {"documentId": {"ABC001": _id(1), "ABC002": _id(2)}[r.query["bates"][0]]})
    export_route(fake, {_id(1): ("ABC001", "x"), _id(2): ("ABC002", "y")})
    out = call("litkit_jev", action="screen", bates=["ABC001", "ABC002"], criteria=["Pricing"])
    assert [r["bates"] for r in out["rows"]] == ["ABC001", "ABC002"]


@pytest.mark.parametrize("args, message", [
    ({"criteria": ["Pricing"]}, "exactly one of documentIds"),
    ({"documentIds": [_id(1)], "workSetId": _id(2), "criteria": ["Pricing"]}, "exactly one of documentIds"),
    ({"documentIds": [_id(1)]}, "exactly one of criteria"),
    ({"documentIds": ["nope"], "criteria": ["Pricing"]}, "uuid"),
    ({"documentIds": [_id(1)], "criteria": [f"c{n}" for n in range(41)]}, "at most 40 criteria"),
    ({"documentIds": [_id(1)], "criteria": ["Pricing"], "thresholds": {"low": 1.5}}, "between 0 and 1"),
    ({"documentIds": [_id(1)], "criteria": ["Pricing"], "thresholds": {"high": 0.5}}, "only low and priv"),
])
def test_screen_rejects_bad_arguments_before_any_request(fake, jev, args, message):
    out = call("litkit_jev", action="screen", **args)
    assert message in out["error"]
    assert jev.requests == []


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

def test_ask_returns_typed_answers_for_one_document(fake, jev):
    fake.route("GET", rf"/api/documents/{_id(1)}", {"doc": {"id": _id(1), "batesStart": "ABC001",
                                                           "custodian": "Doe, Jane", "subject": "Re: price"}})
    fake.route("GET", rf"/api/documents/{_id(1)}/text", {"extractedText": "We should hold the price."})
    questions = {"privileged": {"type": "noul", "instructions": "Does `document` seek legal advice?"},
                 "tone": {"type": "score", "instructions": "How hostile is `document`?",
                          "criteria": ["Cordial", "Tense", "Hostile"]}}
    out = call("litkit_jev", action="ask", documentId=_id(1), questions=questions)
    assert out["bates"] == "ABC001" and set(out["answers"]) == {"privileged", "tone"}
    sent = jev.requests[0]["body"]
    assert sent["questions"] == questions
    assert sent["state"]["document"]["text"] == "We should hold the price."
    assert sent["state"]["document"]["subject"] == "Re: price"
    assert all(r.headers["x-litkit-acting-user"] == USER_ID for r in fake.requests)


def test_ask_on_pasted_text_makes_no_litkit_call(fake, jev):
    out = call("litkit_jev", action="ask", text="Call me about the deal.",
               questions={"urgent": {"type": "noul", "instructions": "Is `document` urgent?"}})
    assert out["answers"]["urgent"]["noul"] == 0.05 and fake.requests == []


@pytest.mark.parametrize("questions, message", [
    ({f"q{n}": {"type": "noul", "instructions": "?"} for n in range(21)}, "at most 20"),
    ({"q": {"type": "maybe", "instructions": "?"}}, "type must be"),
    ({"q": {"type": "noul"}}, "needs instructions"),
    ({"q": {"type": "choice", "instructions": "?", "criteria": {"a": None}}}, "two or more"),
    ({"q": {"type": "score", "instructions": "?", "criteria": ["only"]}}, "2 to 10"),
    ({"1bad": {"type": "noul", "instructions": "?"}}, "question id"),
    ({}, "'questions' must be"),
])
def test_ask_rejects_bad_questions(fake, jev, questions, message):
    out = call("litkit_jev", action="ask", text="x", questions=questions)
    assert message in out["error"] and jev.requests == []


# ---------------------------------------------------------------------------
# firstPass on a proposed review run, and registration
# ---------------------------------------------------------------------------

def _propose(fake, **extra):
    review = FakeReview(fake)
    call("litkit_review", action="create", criteriaSetId="builtin:privilege", tags=["Privileged"],
         scope={"workSetId": review.work_sets[0]["id"]}, **extra)
    return fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose")[-1].json()


def test_review_create_omits_first_pass_by_default(fake, env):
    assert "firstPass" not in _propose(fake)


def test_review_create_passes_first_pass_through(fake, env):
    body = _propose(fake, firstPass={"enabled": True, "thresholds": {"low": 0.1}})
    assert body["firstPass"] == {"enabled": True, "thresholds": {"low": 0.1}}
    assert _propose(fake, first_pass={"enabled": False})["firstPass"] == {"enabled": False}


@pytest.mark.parametrize("value, message", [
    ({"thresholds": {"low": 0.1}}, "enabled"),
    ({"enabled": "yes"}, "enabled"),
    ({"enabled": True, "provider": "x"}, "only enabled and thresholds"),
    ({"enabled": True, "thresholds": {"priv": 2}}, "between 0 and 1"),
])
def test_review_create_rejects_a_bad_first_pass(fake, env, value, message):
    out = call("litkit_review", action="create", criteriaSetId="builtin:privilege", tags=["Privileged"],
               scope={"workSetId": _id(1)}, firstPass=value)
    assert message in out["error"]
    assert fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose") == []


def test_litkit_jev_is_registered_with_the_toolset():
    names = [name for name, _schema, _handler in T.TOOLS]
    assert "litkit_jev" in names and T.SCHEMAS["litkit_jev"]["name"] == "litkit_jev"
    assert "firstPass" in T.SCHEMAS["litkit_review"]["parameters"]["properties"]
