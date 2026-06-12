"""Regression test: notification poller exits when session removed from _sessions.

Issue #44789 — leaked poller threads from earlier tests race on the shared
completion_queue because the loop only checked stop_event and _finalized.
Adding `sid in _sessions` makes _sessions.pop(sid) sufficient to reap the
poller in both test cleanup and production (dead-session reaping).
"""
import threading
import time

import pytest


def test_poller_exits_when_session_removed_from_sessions(monkeypatch):
    """Removing the session from _sessions causes the poller loop to exit."""
    from tools.process_registry import process_registry
    import tui_gateway.server as server

    # Drain the queue first
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    sess = {
        "running": False,
        "history_lock": threading.Lock(),
        "_finalized": False,
    }
    sid = "sid_poller_exit_test"
    server._sessions[sid] = sess

    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: None)

    stop = threading.Event()
    t = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, sid, sess),
        daemon=True,
    )
    t.start()

    # Let the poller start and settle into the get(timeout=0.5) wait
    time.sleep(0.3)

    # Remove the session — this should cause the loop to exit
    server._sessions.pop(sid, None)

    # Wait for the thread to finish (it should exit within one iteration)
    t.join(timeout=2.0)
    assert not t.is_alive(), "poller thread should exit when session removed from _sessions"

    # Cleanup
    stop.set()


def test_poller_stays_alive_when_session_present(monkeypatch):
    """Poller keeps running while session is still in _sessions and not stopped."""
    from tools.process_registry import process_registry
    import tui_gateway.server as server

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    sess = {
        "running": False,
        "history_lock": threading.Lock(),
        "_finalized": False,
    }
    sid = "sid_poller_alive_test"
    server._sessions[sid] = sess

    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: emitted.append(a))
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: None)

    stop = threading.Event()
    t = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, sid, sess),
        daemon=True,
    )
    t.start()

    # Let it settle
    time.sleep(0.3)

    # Session is still present — thread should be alive
    assert t.is_alive(), "poller thread should keep running while session is in _sessions"

    # Cleanup
    stop.set()
    server._sessions.pop(sid, None)
    t.join(timeout=2.0)
