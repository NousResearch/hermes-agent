"""A stream whose reader survives the stale kill must not be re-killed forever.

Regression for #141098. The stale timer re-fires every window while the worker has
not unwound; ``_count_stale_attempt`` counts each attempt only once by design, and the
cross-call breaker (``_check_stale_giveup``) is consulted only when a NEW call starts
— so a reader that the kill's socket shutdown cannot unblock kept the monitor
re-killing the same attempt (1,231 re-kills over 3.5 days in the report), with every
re-kill refreshing the activity heartbeat, which also kept the kanban wedge reclaim
from ever seeing the worker.

The contract under test: an attempt may be re-killed only inside its own read patience
(the timeout that legitimately ends a parked read by itself). Past that window it is
abandoned — the monitor stops, the call fails so the retry/fallback path runs, and the
inline path stops claiming liveness so the external wedge reclaim owns the worker.

The common path stays guarded by ``test_stream_stale_kill_unwedge.py`` (a kill that
DOES unblock the reader still reconnects), by ``test_stream_stale_circuit_breaker.py``
(one attempt still counts once), and by the recoverable-kill control at the bottom of
this file.
"""
import threading
import time
from types import SimpleNamespace

import pytest

import run_agent
from agent import chat_completion_helpers as helpers
from agent import chat_completion_stream_monitor as stream_monitor
from agent.error_classifier import FailoverReason, classify_api_error
from agent.status_output import StatusOutputMixin

# The contract is "bounded", not any particular number of kills: the window lives in
# agent/chat_completion_stream_monitor.py. On the unfixed tree the loop never ends.
_MAX_KILLS = 40


class _Clock:
    """Deterministic clock for the patience window (patched in as the module's ``time``)."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def time(self) -> float:
        return self.now

    def sleep(self, _seconds: float) -> None:
        pass


class _RecordingAgent(StatusOutputMixin):
    """The real status emitter, with activity touches and wait notices captured."""

    def __init__(self, touches: list, waits: list):
        self.base_url = ""
        self._interrupt_requested = False
        self._consecutive_stale_streams = 0
        self.provider = "custom"
        self.thinking_callback = waits.append
        self._touches = touches

    def _touch_activity(self, text: str) -> None:
        self._touches.append(text)


def _wedged_call(patience: float = 0.25):
    """A ``_StreamingCall`` whose worker never unwinds: ``_call_done`` is never set."""
    touches: list = []
    waits: list = []
    agent = _RecordingAgent(touches, waits)
    call = helpers._StreamingCall.__new__(helpers._StreamingCall)
    call.agent = agent
    call.api_kwargs = {"model": "m", "messages": []}
    call._call_done = threading.Event()  # never set: the worker stays parked
    call._stream_stale_timeout = 0.02
    call.last_chunk_time = {"t": time.time() - 60.0}  # already past the stale bound
    call.stream_attempt_lock = threading.Lock()
    call.stream_attempt_state = {"current": 7, "cancelled": set()}
    call._stale_counted_attempts = set()
    call._attempt_stream_response = None
    call.prompt_progress = {"progress": None, "t": 0.0}
    call.clients = SimpleNamespace(diag={}, close_once=lambda *_a: None)
    call._cancel_current_stream_attempt = lambda *_a: None
    call._shutdown_stale_attempt_socket = lambda *_a: None
    call._stream_timeouts = lambda: (1.0, patience, 1.0)
    return call, touches, waits


def _run_call_in_thread(agent):
    """Call the streaming API on a worker thread; return (thread, outcome list)."""
    outcome: list = []

    def _call():
        try:
            agent._interruptible_streaming_api_call(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            outcome.append(None)
        except BaseException as exc:  # a test thread: capture whatever the call raises
            outcome.append(exc)

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    return worker, outcome


def _agent(base_url: str):
    agent = run_agent.AIAgent(
        api_key="test-key", base_url=base_url, model="m", provider="custom",
        platform="cli",  # worker thread + monitor thread: the gateway shape from the report
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        max_iterations=1,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


# ── deterministic: the monitor stops re-killing one wedged attempt ──────────


def test_monitor_abandons_an_attempt_its_kills_cannot_unblock():
    call, touches, _waits = _wedged_call()
    kills: list = []
    touch_counts: list[int] = []
    real_kill = call._kill_stale_stream
    real_note = call._note_stale_kill
    call._kill_stale_stream = lambda elapsed: (kills.append(elapsed), real_kill(elapsed))[1]

    def note_stale_kill(counted, elapsed):
        real_note(counted, elapsed)
        touch_counts.append(len(touches))

    call._note_stale_kill = note_stale_kill

    monitor = threading.Thread(target=call._monitor_loop, daemon=True)
    monitor.start()
    monitor.join(timeout=10.0)

    assert not monitor.is_alive(), (
        f"the monitor kept re-killing a stream whose reader never unwound "
        f"({len(kills)} kills in 10s and still running)")
    assert 1 < len(kills) <= _MAX_KILLS, f"kills are not bounded: {len(kills)}"
    assert call._attempt_is_wedged(), "the attempt was never abandoned"
    # The first kill may claim liveness (status notice + activity); every re-kill and the
    # terminal abandonment must not. Real emitter: a `waits.append` stub hid the touch.
    assert touch_counts[0] >= 1, "the first kill stopped claiming liveness"
    assert set(touch_counts[1:]) == {touch_counts[0]}, (
        f"a re-kill or abandonment refreshed the activity heartbeat: {touch_counts}; "
        "the kanban wedge reclaim reads that heartbeat as liveness")


def test_the_patience_window_is_per_attempt(monkeypatch):
    """The window runs from the attempt's own first kill, and a retry that starts a
    new attempt gets a fresh one instead of inheriting an abandonment."""
    clock = _Clock()
    monkeypatch.setattr(stream_monitor, "time", clock)
    call, _touches, _waits = _wedged_call(patience=10.0)

    call._note_stale_kill(True, 5.0)  # attempt 7's first kill starts its window
    clock.now += 5.0
    call._note_stale_kill(False, 5.0)  # a re-kill inside the window stays recoverable
    assert not call._attempt_is_wedged()

    call.stream_attempt_state["current"] = 8  # the worker unwound; a retry starts
    clock.now += 20.0  # ... long after the previous attempt's window
    call._note_stale_kill(True, 5.0)
    assert not call._attempt_is_wedged(), (
        "a fresh attempt inherited the previous attempt's wedge clock")

    clock.now += 11.0  # past THIS attempt's patience, reader still not unwinding
    call._note_stale_kill(False, 5.0)
    assert call._attempt_is_wedged(), "the patience window never expired for a wedged attempt"


def test_the_bounds_floor_is_the_loops_own_windows(monkeypatch):
    """The bound must hold on the loop's own clock: with a read patience far SHORTER than
    the stale budget, three silent windows govern, and the abandon lands on wall clock even
    though the read timeout never fired (it never did in #141098)."""
    clock = _Clock()
    monkeypatch.setattr(stream_monitor, "time", clock)
    call, _touches, _waits = _wedged_call(patience=0.01)  # read timeout: tiny
    call._stream_stale_timeout = 10.0  # three windows = 30s

    call._note_stale_kill(True, 5.0)
    clock.now += 20.0  # two windows in: still inside
    call._note_stale_kill(False, 5.0)
    assert not call._attempt_is_wedged(), "the read timeout governed instead of the windows"

    clock.now += 11.0  # past the third window
    call._note_stale_kill(False, 5.0)
    assert call._attempt_is_wedged(), "the bound did not hold on the loop's own clock"


def test_a_wedged_attempt_claims_no_liveness_from_the_heartbeat():
    """The 30s heartbeat touches activity independently of the kill path, so pin the whole
    property: before abandonment it claims liveness, after it claims none, and the status
    line stops promising a reconnect."""
    call, touches, waits = _wedged_call(patience=0.0)
    call._stream_stale_timeout = 0.0  # no window floor: the re-kill abandons immediately

    call._heartbeat(5)
    assert len(touches) == 1, "the heartbeat must claim liveness while the attempt is alive"

    call._note_stale_kill(True, 5.0)  # first kill: counted, reconnect notice
    before_abandon = len(touches)
    call._note_stale_kill(False, 5.0)  # abandoned on the re-kill
    assert call._attempt_is_wedged()
    assert len(touches) == before_abandon, (
        "the terminal abandoning diagnostic refreshed the activity heartbeat")

    before_heartbeat = len(touches)
    call._heartbeat(5)
    assert len(touches) == before_heartbeat, "a wedged attempt still refreshed the activity heartbeat"
    assert "reconnecting" in waits[0], "the first kill must still show the reconnect notice"
    assert "abandoning" in waits[-1], "the abandonment must replace the reconnect notice"


def test_the_wedge_error_classifies_as_a_transport_failure_for_fallback():
    """Raised as a plain unknown error the turn would replay the same wedged route until its
    retry budget is gone; the message carries the give-up phrases so the classifier lands on
    ``timeout`` + fallback, exactly like ``_check_stale_giveup``'s error."""
    call, _touches, _waits = _wedged_call(patience=0.0)
    call._stream_stale_timeout = 0.0  # no window floor: the re-kill abandons immediately
    call._note_stale_kill(True, 5.0)
    call._note_stale_kill(False, 5.0)
    assert call._attempt_is_wedged()

    with pytest.raises(RuntimeError) as caught:
        call._raise_wedged_stream()

    verdict = classify_api_error(caught.value, provider="custom", model="m")
    assert verdict.reason == FailoverReason.timeout, f"classified as {verdict.reason}"
    assert verdict.should_fallback is True, "the fallback chain would never be walked"
    assert verdict.retryable is False, "a wedged route must not be retried on the spot"


# ── end to end: the kill cannot reach the reader, the call must end anyway ──


class _ParkedStream:
    """Headers arrived and the body read never returns: the report's shape, where the
    kill's socket shutdown has no reader to wake. Nothing in the test can end that read
    — only the attempt being abandoned can end the call — so the test releases the
    parked thread at teardown."""

    def __init__(self):
        self.release = threading.Event()  # set by the test's teardown

    def __iter__(self):
        self.release.wait()  # the parked read: released only after the assertions
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_wedged_stream_fails_the_call_instead_of_re_killing_forever(monkeypatch):
    """Pre-fix this call never returned: the monitor re-killed the same attempt every
    stale window (the 3.5-day loop from the report) while the activity heartbeat kept
    the worker looking alive. Now the attempt is abandoned past its read patience and
    the call fails with an actionable error, so the caller's retry/fallback path runs."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.2")
    monkeypatch.setenv("HERMES_STREAM_READ_TIMEOUT", "0.5")  # the abandoned-attempt patience
    parked = _ParkedStream()
    monkeypatch.setattr(helpers._StreamingCall, "_open_chat_stream",
                        lambda self, stream_kwargs: parked)

    try:
        worker, outcome = _run_call_in_thread(_agent("http://127.0.0.1:1/v1"))
        worker.join(timeout=15.0)

        assert not worker.is_alive(), (
            "the call never ended: the monitor re-killed a reader the kill could not unblock")
        assert len(outcome) == 1, "the call neither returned nor raised"
        assert isinstance(outcome[0], RuntimeError), f"unexpected outcome: {outcome[0]!r}"
        assert "wedged" in str(outcome[0]), f"the error does not name the wedge: {outcome[0]}"
    finally:
        parked.release.set()


def test_a_kill_that_does_unblock_the_reader_is_not_called_wedged(monkeypatch):
    """The same-file guard for the common path: a reader the kill CAN unblock ends the
    call through the normal error path, never through the abandon verdict."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.2")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    released = threading.Event()

    class _UnblockedByKill:
        def __iter__(self):
            released.wait(timeout=5.0)  # the kill releases this (patched below)
            raise ConnectionError("connection dropped")

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(helpers._StreamingCall, "_open_chat_stream",
                        lambda self, stream_kwargs: _UnblockedByKill())
    monkeypatch.setattr(helpers._StreamingCall, "_shutdown_stale_attempt_socket",
                        lambda self, response: released.set())

    worker, outcome = _run_call_in_thread(_agent("http://127.0.0.1:1/v1"))
    worker.join(timeout=15.0)

    assert not worker.is_alive(), "the recoverable call never ended"
    assert len(outcome) == 1, "the call neither returned nor raised"
    assert "wedged" not in str(outcome[0]), (
        f"a recoverable kill was reported as a wedged attempt: {outcome[0]!r}")
