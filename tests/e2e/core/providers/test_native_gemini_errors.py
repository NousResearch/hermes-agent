"""Gemini native wire conformance: documented errors, blocked candidates and a mid-stream drop.

Each scenario is one real ``hermes chat -q`` turn against the fake Google endpoint
(``tests/fakes/providers/gemini_native.py``); the scenarios are independent and run concurrently.

* retryable (429 ``RESOURCE_EXHAUSTED`` with ``RetryInfo``, 503 ``UNAVAILABLE``): retried, then the
  answer is printed and persisted once;
* terminal (400 ``INVALID_ARGUMENT``, candidate ``finishReason`` ``SAFETY`` / ``RECITATION``, a prompt
  blocked through ``promptFeedback.blockReason``): exactly one request, surfaced once, no fake
  success persisted;
* a TLS connection dropped mid-SSE: recovered with no partial/duplicated assistant row.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.e2e.core._pending_fixes import known_gate
from tests.e2e.core.providers import _native_helpers as nh
from tests.fakes.providers.gemini_native import (
    HERMES_ENV,
    Blocked,
    Drop,
    GeminiFake,
    GoogleError,
    Recorded,
    Reply,
    Responder,
    Text,
    hermes_model,
)

KNOWN: dict[str, tuple[str, str]] = {
    "prompt_blocked": (r"terminal Google response was re-sent \d+x",
                       "#121317 promptFeedback.blockReason is retried 9x as an empty stream and reported as "
                       "'temporarily unavailable'"),
}

NEVER = "GEMINI-MUST-NOT-BE-SHOWN"
QUOTA = "Resource has been exhausted (e.g. check quota)."
PARTIAL = "GEMINI-PARTIAL-DROPPED "


@dataclass
class Case:
    script: list[Reply]
    route: Responder | None = None
    max_retries: int = 2


CASES: dict[str, Case] = {
    "rate_limited_429": Case([GoogleError(429, "RESOURCE_EXHAUSTED", QUOTA, 1),
                              GoogleError(429, "RESOURCE_EXHAUSTED", QUOTA, 1),
                              Text("GEMINI-RECOVERED-429")], max_retries=3),
    "unavailable_503": Case([GoogleError(503, "UNAVAILABLE", "The model is overloaded. Please try again later."),
                             Text("GEMINI-RECOVERED-503")]),
    "invalid_argument_400": Case([GoogleError(400, "INVALID_ARGUMENT", "Request contains an invalid argument. "
                                              "GEMINI-MARK-400"), Text(NEVER)]),
    "safety": Case([Blocked("SAFETY"), Text(NEVER)]),
    "recitation": Case([Blocked("RECITATION"), Text(NEVER)]),
    # Google blocks the same prompt every time: answer every attempt with the block.
    "prompt_blocked": Case([], route=lambda rec: Blocked("SAFETY", prompt=True), max_retries=3),
    "stream_drop": Case([Drop(PARTIAL), Text("GEMINI-FULL-AFTER-DROP")]),
}


@dataclass
class Outcome:
    result: nh.ChatResult
    calls: list[Recorded]
    rows: list[dict]

    @property
    def output(self) -> str:
        return self.result.stdout + self.result.stderr

    def describe(self) -> str:
        return f"{self.result.describe()}\ncalls={[(c.reply, c.status) for c in self.calls]}"


def _run(root: Path, name: str, case: Case) -> Outcome:
    home = nh.make_home(root / name, hermes_model(), env_file=HERMES_ENV,
                        extra_config={"agent": {"api_max_retries": case.max_retries}})
    with GeminiFake(root / name / "fake", case.script, route=case.route) as fake:
        result = nh.run_chat(home, f"Say hello ({name}).", env=fake.child_env())
        calls = fake.generate_calls()
    return Outcome(result, calls, nh.messages(home))


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Outcome]:
    base = tmp_path_factory.mktemp("gemini_errors")
    with ThreadPoolExecutor(len(CASES)) as pool:
        futures = {name: pool.submit(_run, base, name, case) for name, case in CASES.items()}
        return {name: f.result() for name, f in futures.items()}


def _answer_once(o: Outcome, answer: str) -> None:
    assert o.result.returncode == 0, o.describe()
    assert o.result.stdout.count(answer) == 1, o.describe()
    assistant = [r for r in o.rows if r["role"] == "assistant" and r.get("content")]
    assert [r["content"] for r in assistant] == [answer], assistant


RETRYABLE = {  # case -> (statuses the fake must have served, in order; the answer)
    "rate_limited_429": ([429, 429, 200], "GEMINI-RECOVERED-429"),
    "unavailable_503": ([503, 200], "GEMINI-RECOVERED-503"),
}


@pytest.mark.parametrize("name", list(RETRYABLE))
def test_retryable_error_is_retried_then_succeeds(outcomes: dict[str, Outcome], name: str) -> None:
    o = outcomes[name]
    statuses, answer = RETRYABLE[name]
    assert [c.status for c in o.calls] == statuses, o.describe()
    _answer_once(o, answer)


TERMINAL = {  # case -> a word the surfaced error must carry (None: any visible error)
    "invalid_argument_400": "GEMINI-MARK-400",
    "safety": "safety",
    "recitation": None,
    "prompt_blocked": "block",
}


@pytest.mark.parametrize("name", list(TERMINAL))
def test_non_retryable_surfaced_once(outcomes: dict[str, Outcome], name: str) -> None:
    o = outcomes[name]
    assert o.calls, f"no request reached the fake\n{o.describe()}"
    # The tracked bug's symptom for KNOWN cases; a plain failure for every other terminal case.
    with known_gate(KNOWN, name, raises=nh.KnownSymptom):
        if len(o.calls) != 1:
            raise nh.KnownSymptom(f"terminal Google response was re-sent {len(o.calls)}x\n{o.describe()}")
    assert o.result.returncode != 0, o.describe()
    word = TERMINAL[name]
    lines = [ln.strip() for ln in o.result.stdout.splitlines() if ln.strip()]
    assert lines, f"nothing surfaced to the user\n{o.describe()}"
    assert len(lines) == len(set(lines)), f"error surfaced more than once\n{o.describe()}"
    if word:
        assert word.lower() in o.output.lower(), o.describe()
    assert NEVER not in o.output
    assert not [r for r in o.rows if r["role"] == "assistant" and NEVER in (r.get("content") or "")]


def test_stream_drop_recovers_without_duplicate_rows(outcomes: dict[str, Outcome]) -> None:
    o = outcomes["stream_drop"]
    assert [c.reply for c in o.calls] == ["script:Drop", "script:Text"], o.describe()
    assert all(c.stream for c in o.calls)
    _answer_once(o, "GEMINI-FULL-AFTER-DROP")
    assert not [r for r in o.rows if PARTIAL.strip() in (r.get("content") or "")], o.rows
    nh.assert_no_duplicate_assistant_text(o.rows, "GEMINI-FULL-AFTER-DROP")
