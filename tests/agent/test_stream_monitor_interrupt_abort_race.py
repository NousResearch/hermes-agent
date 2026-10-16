"""Race: the monitor must not abort for /stop when the worker already finished.

`_monitor_loop` wakes on `self._call_done.wait(timeout=0.3)`. The worker can
complete *during* that 0.3s sleep and unwind its own request client (its SSE
loop also checks `_interrupt_requested`, see chat_completion_helpers). If the
loop body then runs unconditionally it fires `_abort_for_interrupt` against a
client the worker already closed and possibly cached/rebuilt for the next
request -- a double abort. Re-checking `_call_done` after the wake closes that
window.

These tests fail if that post-wake re-check is removed, so they pin the fix
rather than merely exercising the loop (the pre-existing wait-notice tests call
`_monitor_loop` too, but never with `_interrupt_requested` set, which is why
they do not catch a regression here).
"""
from types import SimpleNamespace

from agent import chat_completion_helpers as h


class _DoneAfterWait:
    """`_call_done` that is clear at loop entry and set during the 0.3s wait.

    This is the exact interleaving the fix guards: `while not is_set()` is
    entered, the worker finishes while we sleep, and the loop body is about to
    run against a worker that has already unwound.
    """

    def __init__(self, unset_runs: int = 1):
        self.calls = 0
        self._unset_runs = unset_runs
        self._set = False

    def is_set(self):
        return self._set

    def wait(self, timeout):
        self.calls += 1
        # First call is the loop-entry wait; the worker completes here.
        if self.calls >= self._unset_runs:
            self._set = True
        return self._set


def _make_call(interrupt_requested: bool):
    call = h._StreamingCall.__new__(h._StreamingCall)
    aborts, notices = [], []
    call.agent = SimpleNamespace(
        base_url="https://example.com",  # not local: skip the load-notice branch
        _interrupt_requested=interrupt_requested,
        _emit_wait_notice=lambda text: notices.append(text),
        _touch_activity=lambda text: None,
    )
    call.api_kwargs = {"model": "test-model"}
    call.last_chunk_time = {"t": 1000.0}
    # Large stale timeout so the stale-reconnect branch cannot fire and be
    # mistaken for the interrupt abort under test.
    call._stream_stale_timeout = 10_000.0
    call._call_done = _DoneAfterWait()
    call._abort_for_interrupt = lambda stale_elapsed: aborts.append(stale_elapsed)
    call._heartbeat = lambda waiting_secs: None
    return call, aborts, notices


def test_no_abort_when_worker_finished_during_wake(monkeypatch):
    """Worker completed while the monitor slept -> no interrupt abort."""
    call, aborts, notices = _make_call(interrupt_requested=True)
    monkeypatch.setattr(h.time, "time", lambda: 1000.0)

    call._monitor_loop()

    assert aborts == [], (
        "monitor fired _abort_for_interrupt after the worker had already "
        "finished -- this is the double-abort the post-wake _call_done "
        "re-check exists to prevent"
    )
    # It must return promptly, not keep looping: one wake is enough.
    assert call._call_done.calls == 1


def test_abort_still_fires_while_worker_still_running(monkeypatch):
    """Guard the opposite arm: a genuine /stop must still abort.

    Without this, deleting the interrupt branch entirely would pass the test
    above -- the abort path has to stay reachable.
    """
    call, aborts, _ = _make_call(interrupt_requested=True)
    # Worker is still running: it never completes during the wake.
    call._call_done = _DoneAfterWait(unset_runs=10_000)
    # Make the loop body terminate after the first abort via the interrupt branch.
    monkeypatch.setattr(h.time, "time", lambda: 1000.0)

    call._monitor_loop()

    assert len(aborts) == 1, "a real /stop while the worker runs must still abort"


def test_no_abort_and_no_spurious_work_when_not_interrupted(monkeypatch):
    """No /stop requested -> the loop must exit on completion without aborting."""
    call, aborts, _ = _make_call(interrupt_requested=False)
    monkeypatch.setattr(h.time, "time", lambda: 1000.0)

    call._monitor_loop()

    assert aborts == []
