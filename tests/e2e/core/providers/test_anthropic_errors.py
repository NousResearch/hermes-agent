"""Anthropic Messages wire conformance: documented error envelopes and stream faults.

A real ``hermes -z`` process on the native route meets the vendor's documented
failures (built from the SDK's error models): ``429 rate_limit_error`` with
``retry-after``, ``529 overloaded_error``, ``400 invalid_request_error`` and a
socket dropped in the middle of a thinking block. Each test asserts the retry
cadence/count the fake observed and what the user and state.db end up with.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.e2e.core.providers._anthropic_helpers import Rig, dump, normalised, start_rig, thinking_of
from tests.fakes.providers.anthropic_messages import ApiError, DropStream, Reply, Text, Thinking

pytestmark = [pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process-tree cleanup uses /proc"),
              pytest.mark.live_system_guard_bypass]


class StreamDropAcceptedAsAnswer(AssertionError):
    """#121320's signature, raised ONLY where the user is handed the dropped stream's fragment.

    Strict xfails below accept nothing else, so a harness failure (process death, timeout, a
    later assertion once the bug is fixed) still fails the run."""


# Red on current main for a tracked, open bug. Strict: the test FAILS the moment the bug is
# fixed, so the entry is removed with the fix instead of masking a later regression.
KNOWN: dict[str, str] = {
    "stream_drop_retry": "#121320 stream closed before message_stop is accepted as a complete answer",
}


def known(key: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, raises=StreamDropAcceptedAsAnswer, reason=KNOWN[key])


def _answer_must_be(stdout: str, expected: str, fragment: str) -> None:
    if stdout.count(expected) != 1 or fragment in stdout:
        raise StreamDropAcceptedAsAnswer(f"user got {stdout[-800:]!r} instead of exactly one {expected!r}")


@pytest.fixture
def rig_factory(tmp_path: Path):
    rigs: list[Rig] = []

    def make(script, **kw) -> Rig:
        rig = start_rig(tmp_path / f"r{len(rigs)}", script, **kw)
        rigs.append(rig)
        return rig

    yield make
    for rig in rigs:
        rig.stop()


def _gaps(rig: Rig) -> list[float]:
    t = [r["t"] for r in rig.srv.main_requests()]
    return [round(b - a, 2) for a, b in zip(t, t[1:])]


def test_429_waits_the_retry_after_then_succeeds(rig_factory) -> None:
    retry_after = 4.0  # above the 2-3 s jittered default backoff, so ignoring it is visible
    rig = rig_factory([ApiError(429, "rate_limit_error", "Number of requests exceeded", retry_after=retry_after),
                       Reply([Text("AFTER-RATE-LIMIT")])], config={"agent": {"api_max_retries": 2}})
    proc = rig.run("-z", "hello")
    assert proc.returncode == 0 and "AFTER-RATE-LIMIT" in proc.stdout, proc.stderr[-2000:]
    assert len(rig.srv.main_requests()) == 2, [r.get("response") for r in rig.srv.requests]
    (gap,) = _gaps(rig)
    assert gap >= retry_after - 0.05, f"retried after {gap}s, before the server's retry-after {retry_after}s"
    assert gap < retry_after + 15, f"retry-after {retry_after}s stretched to {gap}s"


def test_529_overloaded_is_retried_then_surfaced(rig_factory) -> None:
    retries = 2
    vendor_message = "Overloaded (e2e-529-marker)"
    # auto_recovery_cycles: 0 disables the documented post-exhaustion wait ladder (15/30/60 s
    # cycles) so the attempt budget alone decides when the error reaches the user.
    rig = rig_factory(lambda _r: ApiError(529, "overloaded_error", vendor_message),
                      config={"agent": {"api_max_retries": retries, "auto_recovery_cycles": 0}})
    proc = rig.run("-z", "hello")
    mains = rig.srv.main_requests()
    assert len(mains) == retries, (
        f"529 must use exactly the api_max_retries={retries} attempt budget; saw {len(mains)} requests, "
        f"gaps {_gaps(rig)}")
    assert all(g >= 0.5 for g in _gaps(rig)), f"529 retries must back off, gaps {_gaps(rig)}"
    surfaced = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"an exhausted 529 must fail the one-shot run: {surfaced[-800:]!r}"
    assert vendor_message in surfaced, f"the vendor's overloaded_error message never reached the user: {surfaced[-800:]}"


def test_400_invalid_request_is_surfaced_without_a_retry_storm(rig_factory) -> None:
    vendor_message = "tools.0.custom.name: e2e-400-marker rejection"
    rig = rig_factory(lambda _r: ApiError(400, "invalid_request_error", vendor_message),
                      config={"agent": {"api_max_retries": 3}})
    proc = rig.run("-z", "hello")
    mains = rig.srv.main_requests()
    assert len(mains) == 1, f"a non-retryable 400 was re-sent {len(mains) - 1} times: gaps {_gaps(rig)}"
    surfaced = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"a rejected request must fail the one-shot run: {surfaced[-800:]!r}"
    assert vendor_message in surfaced, f"the vendor's invalid_request_error message never reached the user: {surfaced[-800:]}"


@known("stream_drop_retry")
def test_stream_drop_mid_thinking_retries_without_duplicate_persisted_content(rig_factory) -> None:
    """The socket dies after two thinking deltas (no signature, no message_stop). The retry must
    resend the same history (no half-streamed assistant turn leaks into it), the user sees the
    answer once, and state.db holds exactly one assistant row carrying only the completed turn."""
    partial = "PARTIAL-THOUGHT " * 6
    rig = rig_factory([
        DropStream(Reply([Thinking(partial, "sig-never-sent"), Text("LOST-ANSWER")], chunk_chars=16), after_deltas=2),
        Reply([Thinking("Clean retry reasoning.", "EqRetrySig+/=="), Text("RECOVERED-ANSWER")]),
    ], config={"agent": {"api_max_retries": 2}})
    proc = rig.run("chat", "-q", "hello", "-Q")
    assert proc.returncode == 0, proc.stderr[-2000:]
    _answer_must_be(proc.stdout, "RECOVERED-ANSWER", "PARTIAL-THOUGHT")
    assert "LOST-ANSWER" not in proc.stdout, proc.stdout[-800:]
    mains = [r["body"] for r in rig.srv.main_requests()]
    assert len(mains) == 2, [r.get("response") for r in rig.srv.requests]
    assert normalised(mains[1]["messages"]) == normalised(mains[0]["messages"]), (
        f"retry history differs from the dropped request's: {dump(mains[1])}")

    (session_id,) = rig.session_ids()
    rows = rig.messages(session_id)
    assistants = [r for r in rows if r["role"] == "assistant"]
    assert len(assistants) == 1, [(r["role"], (r["content"] or "")[:60]) for r in rows]
    persisted = json.dumps(assistants[0], default=str)
    assert "RECOVERED-ANSWER" in persisted and "PARTIAL-THOUGHT" not in persisted and "LOST-ANSWER" not in persisted
    assert persisted.count("Clean retry reasoning.") >= 1


@known("stream_drop_retry")
def test_stream_drop_then_next_turn_replays_only_the_completed_signature(rig_factory) -> None:
    """After a mid-thinking drop + successful retry, the NEXT user turn replays the retry's signed
    thinking byte-exact — never the dropped stream's unsigned fragment."""
    rig = rig_factory([
        DropStream(Reply([Thinking("FRAGMENT " * 8, "sig-never-sent"), Text("x")], chunk_chars=12), after_deltas=2),
        Reply([Thinking("Completed reasoning.", "EqCompleted+/Sig=="), Text("TURN-ONE")]),
        Reply([Text("TURN-TWO")]),
    ], config={"agent": {"api_max_retries": 2}})
    first = rig.run("chat", "-q", "one", "-Q")
    assert first.returncode == 0, first.stderr[-2000:]
    _answer_must_be(first.stdout, "TURN-ONE", "FRAGMENT")
    (session_id,) = rig.session_ids()
    second = rig.run("chat", "--resume", session_id, "-q", "two", "-Q")
    assert second.returncode == 0 and "TURN-TWO" in second.stdout, second.stderr[-2000:]
    last = rig.srv.main_requests()[-1]["body"]
    prior = [m for m in last["messages"] if m["role"] == "assistant"]
    assert len(prior) == 1 and thinking_of(prior[0]) == [("Completed reasoning.", "EqCompleted+/Sig==")], dump(last)
    assert "FRAGMENT" not in json.dumps(last["messages"])
    assert not rig.srv.schema_errors(), rig.srv.schema_errors()
