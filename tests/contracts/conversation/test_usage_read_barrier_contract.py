"""Contract cases for the usage read barrier: the exact session readers drain queued usage first, except on a thread running an event loop, where they never wait for the usage writer and read the committed totals.

The writer is held only on SQLite, whose holder applies queued deltas on a writer thread; the reference applies a delta inside ``queue_usage``, so its barrier is always met and the case checks the same outcome without a held writer.
"""

from __future__ import annotations

import asyncio
import threading
import time

from persistence.conversation.sessions import NewSession, SessionQuery, SessionWindow
from persistence.conversation.usage import TokenCounts, UsageDelta
from persistence.core.identity import SessionId

SID = SessionId("usage-barrier")


def _hold_usage_writer(backend_name: str, monkeypatch) -> tuple[threading.Event, threading.Event]:
    """``(entered, release)``: on SQLite the holder's writer blocks inside its first apply, as behind another connection's write lock, until ``release`` is set; on the reference nothing is held and ``entered`` is already set."""
    entered, release = threading.Event(), threading.Event()
    if backend_name != "sqlite":
        entered.set()
        return entered, release
    from persistence.sqlite import _holder

    apply = _holder._token_delta_applier

    def held(holder, session_id, kwargs):
        entered.set()
        release.wait(timeout=30)
        apply(holder, session_id, kwargs)

    monkeypatch.setattr(_holder, "_token_delta_applier", held)
    return entered, release


def test_reads_on_an_event_loop_never_wait_for_queued_usage(backend, backend_name, scope_a, monkeypatch):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(SID, "telegram", session_key="agent:main:telegram:dm:barrier"))
    entered, release = _hold_usage_writer(backend_name, monkeypatch)
    try:
        store.usage.queue_usage(SID, UsageDelta(tokens=TokenCounts(input_tokens=7), api_call_count=1))
        assert entered.wait(timeout=10)

        async def read_on_loop():
            started = time.monotonic()
            row = store.sessions.get(SID)
            listed = store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=5))
            gateway = store.sessions.list_gateway_sessions()
            store.usage.recent_route(SID)
            return row, listed, gateway, time.monotonic() - started

        row, listed, gateway, elapsed = asyncio.run(read_on_loop())
        assert elapsed < 2.0, f"a read on the event loop waited {elapsed:.2f}s for the usage writer"
        assert row is not None and row.session_id == SID
        assert SID in [entry.session.session_id for entry in listed]
        assert SID in [entry.session.session_id for entry in gateway]
    finally:
        release.set()
    # Off the loop the reader drains first, so the queued delta is visible.
    assert store.sessions.get(SID).input_tokens == 7
