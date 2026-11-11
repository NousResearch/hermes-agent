"""The turn executor reuses parked workers, and a reused thread carries nothing from its last turn.

``submit`` runs on the gateway loop. Starting a thread there waits until the new thread gets the
GIL, which froze the loop on every submit while many turns ran (#142061); a parked worker takes
the item with a queue put instead. Reuse is only safe if turn 1's thread-keyed state (interrupt
bit, per-turn thread-locals, the thread-scoped sudo cache, context variables) is gone in turn 2.
"""

from __future__ import annotations

import contextvars
import threading
import time

from gateway.turn_executor import _UnboundedThreadExecutor


def _wait_parked(ex, n=1):
    deadline = time.monotonic() + 5
    while len(getattr(ex, "_idle", ())) < n:  # a thread-per-item executor never parks one
        assert time.monotonic() < deadline, "the worker never parked"
        time.sleep(0.01)


def test_a_reused_worker_sees_none_of_the_previous_turns_thread_state():
    from hermes_constants import get_hermes_home, set_hermes_home_override
    from tools import interrupt, terminal_tool
    from tools import terminal_tool_sudo as sudo

    fresh_home = contextvars.Context().run(get_hermes_home)
    ex = _UnboundedThreadExecutor("reuse-probe")
    try:
        def turn_one():
            interrupt.set_interrupt(True, reason="user pressed stop")
            terminal_tool._callback_tls.approval = "session-a approval callback"
            set_hermes_home_override("/profiles/session-a")  # left set: the turn never resets it
            sudo._sudo_password_cache[f"thread:{threading.get_ident()}"] = "session-a sudo password"
            return threading.get_ident()

        first = ex.submit(turn_one).result(5)
        _wait_parked(ex)

        def turn_two():
            return dict(
                tid=threading.get_ident(),
                interrupted=interrupt.is_interrupted(),
                reason=interrupt.get_interrupt_reason(),
                approval=getattr(terminal_tool._callback_tls, "approval", None),
                home=get_hermes_home(),
                sudo=sudo._sudo_password_cache.get(f"thread:{threading.get_ident()}"),
            )

        seen = ex.submit(turn_two).result(5)
        assert seen.pop("tid") == first, "the parked worker was not reused"
        assert seen == dict(interrupted=False, reason=None, approval=None, home=fresh_home, sudo=None)
    finally:
        ex.shutdown(wait=True)


def test_submit_to_a_parked_worker_starts_no_thread_and_shutdown_releases_it(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._executor_lock = threading.Lock()
    runner._executor = None
    runner._housekeeping_executor = None
    runner._executor_closing = False
    ex = runner._get_executor()
    assert isinstance(ex, _UnboundedThreadExecutor)
    ex.submit(lambda: None).result(5)
    _wait_parked(ex)
    parked = ex._idle[0].thread
    assert parked is not None

    def no_thread_start(self):
        raise AssertionError("submit started a thread on the event loop while a worker was parked")

    with monkeypatch.context() as m:
        m.setattr(threading.Thread, "start", no_thread_start)
        assert ex.submit(lambda: "ran").result(5) == "ran"
    _wait_parked(ex)
    # A parked worker holds no work: shutdown counts nothing live and the worker exits.
    assert runner._shutdown_executor(drain_timeout=5) == 0
    parked.join(5)
    assert not parked.is_alive()


def test_a_reused_worker_does_not_log_as_the_previous_turns_session():
    import logging

    import hermes_logging

    ex = _UnboundedThreadExecutor("reuse-probe")
    try:
        def turn_one():
            hermes_logging.set_session_context("session-a")  # build_turn_context does this, never clears it
            return threading.get_ident()

        first = ex.submit(turn_one).result(5)
        _wait_parked(ex)

        def next_item():  # another session's pre-turn work, an idle gate, another profile's turn
            record = logging.getLogRecordFactory()("gateway.run", logging.INFO, __file__, 1, "msg", (), None)
            return threading.get_ident(), record.session_tag

        tid, tag = ex.submit(next_item).result(5)
        assert tid == first, "the parked worker was not reused"
        assert tag == "", f"the reused thread logs as {tag!r}"
    finally:
        ex.shutdown(wait=True)
