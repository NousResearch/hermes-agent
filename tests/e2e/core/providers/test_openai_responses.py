"""OpenAI Responses (Codex) wire conformance through REAL Hermes processes.

The fake (``tests/fakes/providers/openai_responses.py``) speaks the streamed Responses
dialect built from the installed ``openai`` SDK's own models and validates every
request Hermes sends against the SDK's request schema. The provider is a config-defined
Responses relay (``providers.<name>.transport: codex_responses``) — the same transport
the ChatGPT Codex and api.openai.com routes use.

Proven here:

* encrypted reasoning items are replayed verbatim, in output order, on the next API call
  of the same turn, on the next user turn after a process restart (``--resume`` from
  state.db), and across turns of one long-lived ``tui_gateway`` session;
* parallel ``function_call`` items round-trip: every call gets exactly one
  ``function_call_output`` with its own ``call_id`` and its own result, persisted once;
* a stale encrypted blob rejected with HTTP 400 ``invalid_encrypted_content`` is
  stripped and the SAME primary is retried (no fallback, no give-up) — and the HTTP-200
  soft-failure twin of that rejection must behave the same (#120399);
* a stream that dies mid-``function_call`` (arguments half streamed, no terminal event) is
  retried without running the half call, and the completed call runs EXACTLY once;
* a 429 carrying ``Retry-After`` is retried after the advertised wait, and answers once.
"""

from __future__ import annotations

import json
import sys

import pytest

from tests.e2e.core.providers._openai_helpers import (
    READ_TOOL,
    Home,
    bug_assertions,
    db_messages,
    db_tool_calls,
    known_marks,
    oneshot,
    responses_input_items,
    responses_provider_config,
    tool_call_args,
)
from tests.e2e.core.providers._openai_tui import TuiGateway
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.providers.openai_responses import (
    FakeResponsesServer,
    FunctionCall,
    HttpError,
    Message,
    Reasoning,
    SoftFail,
    Turn,
)

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated")

# Scenario -> "#issue one-line symptom" for scenarios red on origin/main (strict xfail that
# only a KnownBugError from bug_assertions() satisfies; see known_marks).
KNOWN: dict[str, str] = {
    "soft_failure_recovers_on_primary": "#120399 HTTP-200 invalid_encrypted_content skips replay-strip recovery",
}


def known(name: str) -> list:
    return known_marks(KNOWN, name)


def _encs(body: dict) -> list:
    return [i.get("encrypted_content") for i in responses_input_items(body, "reasoning")]


def _answer(run) -> str:
    return run.stdout.strip()


def test_encrypted_reasoning_replayed_in_turn_and_after_resume(tmp_path) -> None:
    h = Home(tmp_path)
    script = [
        Turn([Reasoning("ENC-A", "plan: read the file"), FunctionCall(READ_TOOL, {"path": "notes.txt"})]),
        Turn([Reasoning("ENC-B"), Message("ANSWER-ONE")]),
        Turn([Reasoning("ENC-C"), Message("ANSWER-TWO")]),
    ]
    with FakeResponsesServer(script) as srv:
        h.write(responses_provider_config(srv.base_url))
        (h.project / "notes.txt").write_text("CANARY-RESP-1\n", encoding="utf-8")
        first = oneshot(h, "read notes.txt and report")
        assert first.proc.returncode == 0 and _answer(first) == "ANSWER-ONE", first.describe()
        second = oneshot(h, "and again", resume=first.session_id)
        assert second.proc.returncode == 0 and _answer(second) == "ANSWER-TWO", second.describe()
        mains = srv.main_requests()
        invalid = srv.invalid_requests()

    assert len(mains) == 3, [b.get("input") for b in mains]
    assert invalid == [], invalid
    for body in mains:
        assert body.get("store") is False and "reasoning.encrypted_content" in (body.get("include") or []), body
    assert _encs(mains[0]) == []
    # Same turn, next call: the blob precedes the function_call it produced, and the tool
    # output answering that call follows it.
    kinds = [i.get("type") or i.get("role") for i in mains[1]["input"]]
    assert _encs(mains[1]) == ["ENC-A"], mains[1]["input"]
    assert kinds.index("reasoning") < kinds.index("function_call") < kinds.index("function_call_output"), kinds
    # After a process restart the replay comes from state.db, in the original order.
    assert _encs(mains[2]) == ["ENC-A", "ENC-B"], mains[2]["input"]
    persisted = [r for r in db_messages(h, first.session_id) if r["role"] == "assistant"]
    assert all(r["codex_reasoning_items"] for r in persisted), persisted


def test_parallel_function_calls_round_trip_once(tmp_path) -> None:
    h = Home(tmp_path)
    script = [
        Turn([Reasoning("ENC-P"), FunctionCall(READ_TOOL, {"path": "a.txt"}), FunctionCall(READ_TOOL, {"path": "b.txt"})]),
        Turn([Message("BOTH-READ")]),
    ]
    with FakeResponsesServer(script) as srv:
        h.write(responses_provider_config(srv.base_url))
        (h.project / "a.txt").write_text("CANARY-ALPHA\n", encoding="utf-8")
        (h.project / "b.txt").write_text("CANARY-BRAVO\n", encoding="utf-8")
        run = oneshot(h, "read a.txt and b.txt")
        mains = srv.main_requests()
        invalid = srv.invalid_requests()

    assert run.proc.returncode == 0 and _answer(run) == "BOTH-READ", run.describe()
    assert invalid == [], invalid
    calls = responses_input_items(mains[1], "function_call")
    outputs = {o["call_id"]: o["output"] for o in responses_input_items(mains[1], "function_call_output")}
    assert len(calls) == 2 and len(outputs) == 2, mains[1]["input"]
    by_path = {json.loads(c["arguments"])["path"]: c["call_id"] for c in calls}
    assert "CANARY-ALPHA" in outputs[by_path["a.txt"]] and "CANARY-BRAVO" in outputs[by_path["b.txt"]], outputs
    rows = db_messages(h, run.session_id)
    tool_rows = [r for r in rows if r["role"] == "tool"]
    assert sorted(r["tool_call_id"] for r in tool_rows) == sorted(outputs), tool_rows


def _stale_blob_session(h: Home, srv: FakeResponsesServer, fallback: FakeLLMServer) -> str:
    """Turn 1 mints a reasoning blob; returns the session id whose resume replays it."""
    cfg = responses_provider_config(srv.base_url)
    cfg["fallback_providers"] = [{"provider": "custom", "model": "fallback-model", "base_url": fallback.base_url}]
    h.write(cfg)
    first = oneshot(h, "first question")
    assert first.proc.returncode == 0 and _answer(first) == "FIRST", first.describe()
    return first.session_id


@pytest.mark.parametrize("rejection", [
    pytest.param(HttpError(400, "Encrypted content could not be decrypted or parsed.", code="invalid_encrypted_content",
                           type="invalid_request_error"), id="http_400"),
    pytest.param(SoftFail(), id="http_200_soft_failure",
                 marks=known("soft_failure_recovers_on_primary")),
])
def test_rejected_encrypted_replay_is_stripped_and_primary_retried(tmp_path, rejection) -> None:
    h = Home(tmp_path)
    script = [Turn([Reasoning("ENC-STALE"), Message("FIRST")]), rejection, Turn([Message("RECOVERED-ON-PRIMARY")])]
    with FakeResponsesServer(script) as srv, FakeLLMServer(default_text="FROM-FALLBACK") as fallback:
        sid = _stale_blob_session(h, srv, fallback)
        run = oneshot(h, "second question", resume=sid)
        mains = srv.main_requests()
        fallback_mains = fallback.main_requests()

    assert _encs(mains[1]) == ["ENC-STALE"], "precondition: the resumed turn replays the stale blob"
    with bug_assertions():
        assert fallback_mains == [], f"fallback engaged instead of replay recovery: {run.describe()}"
        assert len(mains) == 3, [m.get("input") for m in mains]
        assert _encs(mains[2]) == [], "the retry must drop the rejected blob"
        assert run.proc.returncode == 0 and _answer(run) == "RECOVERED-ON-PRIMARY", run.describe()


def test_tui_gateway_session_replays_reasoning_across_turns(tmp_path) -> None:
    """One long-lived backend (the Desktop/TUI shape): turn 2's request carries turn 1's
    blob without any restart, and the fallback chat endpoint is never touched."""
    h = Home(tmp_path)
    script = [Turn([Reasoning("ENC-T1"), Message("TUI-ONE")]), Turn([Reasoning("ENC-T2"), Message("TUI-TWO")])]
    with FakeResponsesServer(script) as srv:
        h.write(responses_provider_config(srv.base_url))
        gw = TuiGateway(h)
        try:
            sid = gw.call("session.create", {"cols": 120})["session_id"]
            assert "TUI-ONE" in gw.turn(sid, "hello")
            assert "TUI-TWO" in gw.turn(sid, "again")
        finally:
            gw.close()
        mains = srv.main_requests()
        invalid = srv.invalid_requests()
    assert invalid == [], invalid
    assert len(mains) == 2 and _encs(mains[1]) == ["ENC-T1"], [m.get("input") for m in mains]


def _main_records(srv: FakeResponsesServer) -> list[dict]:
    return [r for r in srv.requests if r["kind"] == "main"]


def test_stream_drop_mid_function_call_runs_the_call_once(tmp_path) -> None:
    """The first stream emits ``response.created``, the ``function_call`` item and HALF its
    arguments, then the socket closes (no ``.done``, no ``response.completed``)."""
    command = "echo ran >> count.txt"
    script = [
        Turn([FunctionCall("terminal", '{"command": "echo ran >> cou')], drop_after_events=3),
        Turn([FunctionCall("terminal", {"command": command})]),
        Turn([Message("TOOL-DONE")]),
    ]
    h = Home(tmp_path)
    with FakeResponsesServer(script) as srv:
        cfg = responses_provider_config(srv.base_url)
        cfg["approvals"] = {"mode": "off"}
        h.write(cfg)
        run = oneshot(h, "append a line to count.txt", env={"TERMINAL_ENV": "local"})
        mains = srv.main_requests()
        invalid = srv.invalid_requests()

    assert run.proc.returncode == 0 and _answer(run) == "TOOL-DONE", run.describe()
    assert len(mains) == 3, [m.get("input") for m in mains]
    assert invalid == [], invalid
    count = h.project / "count.txt"
    assert count.exists(), f"tool never ran: {run.describe()}"
    assert count.read_text(encoding="utf-8").splitlines() == ["ran"], "side effect ran more than once"
    # The half call never reaches the wire: the retry resends exactly the first request's input.
    assert mains[1]["input"] == mains[0]["input"], (mains[0]["input"], mains[1]["input"])
    # The final request replays exactly one call and its one result.
    calls = responses_input_items(mains[2], "function_call")
    outputs = responses_input_items(mains[2], "function_call_output")
    assert [json.loads(c["arguments"]) for c in calls] == [{"command": command}], calls
    assert [o["call_id"] for o in outputs] == [c["call_id"] for c in calls], outputs
    persisted = db_tool_calls(db_messages(h, run.session_id))
    assert [tool_call_args(c) for c in persisted] == [{"command": command}], persisted


def test_rate_limit_retry_after_is_honoured(tmp_path) -> None:
    """HTTP 429 + ``Retry-After: 4`` (clearly above the generic 2-3 s first backoff): the
    retry waits the advertised time, resends the same input, and the turn answers once."""
    retry_after = 4.0
    script = [HttpError(429, "Rate limit reached for requests", code="rate_limit_exceeded", type="requests",
                        retry_after=retry_after),
              Turn([Message("AFTER-RATE-LIMIT")])]
    h = Home(tmp_path)
    with FakeResponsesServer(script) as srv:
        h.write(responses_provider_config(srv.base_url))
        run = oneshot(h, "say hello")
        records = _main_records(srv)

    assert run.proc.returncode == 0 and _answer(run) == "AFTER-RATE-LIMIT", run.describe()
    assert [r["response"] for r in records] == ["HttpError", "Turn"], run.describe()
    gap = records[1]["t"] - records[0]["t"]
    assert gap >= retry_after - 0.1, f"retried {gap:.2f}s after a 429 advertising Retry-After {retry_after}s"
    assert records[1]["body"]["input"] == records[0]["body"]["input"]
    answers = [r for r in db_messages(h, run.session_id)
               if r["role"] == "assistant" and "AFTER-RATE-LIMIT" in (r.get("content") or "")]
    assert len(answers) == 1, db_messages(h, run.session_id)
