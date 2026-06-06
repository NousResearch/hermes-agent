"""Regression test: ``_handoff_watcher`` must offload its blocking SessionDB
calls off the platform event loop (issue #40695).

The watcher (``gateway/run.py``) polls ``state.db`` for pending CLI->gateway
session handoffs. It made four *synchronous* ``SessionDB`` calls
(``list_pending_handoffs`` / ``claim_handoff`` / ``complete_handoff`` /
``fail_handoff``) directly on the event loop, so under SQLite WAL-lock
contention the Discord gateway heartbeat stalled (10-40s) -> WebSocket
disconnects and ``404 Unknown interaction`` errors.

The fix wraps each call in ``asyncio.to_thread`` (the file's own dominant
convention), so the loop never blocks on SQLite. This test proves the offload
by asserting each blocking call runs on a worker thread, not the loop thread.
Without the fix the calls run on the loop thread and these assertions fail.
"""

import asyncio
import threading
from unittest.mock import patch

from gateway.run import GatewayRunner


class _FakeSessionDB:
    """Stand-in for ``SessionDB`` recording which thread each call runs on.

    Uses the real ``SessionDB`` method names/signatures that
    ``_handoff_watcher`` invokes in ``gateway/run.py``.
    """

    def __init__(self):
        self.threads = {}

    def list_pending_handoffs(self):
        self.threads["list"] = threading.current_thread()
        return [{"id": "session-1"}]

    def claim_handoff(self, session_id):
        self.threads["claim"] = threading.current_thread()
        return True

    def complete_handoff(self, session_id):
        self.threads["complete"] = threading.current_thread()

    def fail_handoff(self, session_id, error):
        self.threads["fail"] = threading.current_thread()


def _make_runner(session_db):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._session_db = session_db
    return runner


def test_handoff_watcher_offloads_sqlite_calls_off_the_event_loop():
    """The blocking SessionDB calls run on a worker thread (``asyncio.to_thread``),
    not the event loop thread (issue #40695)."""
    session_db = _FakeSessionDB()
    runner = _make_runner(session_db)
    loop_thread = {}
    sleep_calls = []
    processed = []

    async def fake_process_handoff(row):
        processed.append(row)

    runner._process_handoff = fake_process_handoff

    async def fake_sleep(delay):
        # First sleep is the watcher's initial connect delay, running on the
        # event loop's own thread -- capture it. Stop the loop after the body
        # runs once (initial sleep + first per-interval sleep).
        loop_thread.setdefault("t", threading.current_thread())
        sleep_calls.append(delay)
        if len(sleep_calls) >= 2:
            runner._running = False

    with patch("asyncio.sleep", side_effect=fake_sleep):
        asyncio.run(runner._handoff_watcher(interval=0.0))

    # The watcher body ran exactly once and processed the pending row.
    assert processed == [{"id": "session-1"}]
    # Each blocking SessionDB call ran OFF the event loop thread.
    assert session_db.threads["list"] is not loop_thread["t"]
    assert session_db.threads["claim"] is not loop_thread["t"]
    assert session_db.threads["complete"] is not loop_thread["t"]


def test_handoff_watcher_offloads_the_failure_path_off_the_event_loop():
    """When ``_process_handoff`` raises, ``fail_handoff`` (the 4th blocking call)
    is also offloaded to a worker thread rather than run on the event loop
    (issue #40695)."""
    session_db = _FakeSessionDB()
    runner = _make_runner(session_db)
    loop_thread = {}
    sleep_calls = []

    async def failing_process_handoff(row):
        raise RuntimeError("boom")

    runner._process_handoff = failing_process_handoff

    async def fake_sleep(delay):
        loop_thread.setdefault("t", threading.current_thread())
        sleep_calls.append(delay)
        if len(sleep_calls) >= 2:
            runner._running = False

    with patch("asyncio.sleep", side_effect=fake_sleep):
        asyncio.run(runner._handoff_watcher(interval=0.0))

    # Processing raised, so the failure branch ran fail_handoff (not complete),
    # and it ran OFF the event loop thread.
    assert "complete" not in session_db.threads
    assert session_db.threads["fail"] is not loop_thread["t"]
