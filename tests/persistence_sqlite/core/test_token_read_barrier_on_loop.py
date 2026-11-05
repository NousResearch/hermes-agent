"""A SQLite session read on an event-loop thread never waits on the token writer; off the loop it still waits, so the
queued usage is visible."""

import asyncio
import threading
import time

from persistence.conversation.sessions import NewSession, SessionQuery, SessionWindow
from persistence.sqlite import _holder

SID = "s-loop"


def test_loop_thread_session_read_does_not_wait_for_a_stuck_writer(backend, scope_a, monkeypatch):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(SID, "test"))
    holder = store.holder()
    entered, release = threading.Event(), threading.Event()
    apply = _holder._token_delta_applier

    def stuck(holder, session_id, kwargs):
        entered.set()
        release.wait(timeout=30)  # the writer is held up, like a write lock held elsewhere
        apply(holder, session_id, kwargs)

    monkeypatch.setattr(_holder, "_token_delta_applier", stuck)
    try:
        holder.queue_token_counts(SID, input_tokens=7, api_call_count=1)
        assert entered.wait(timeout=10)

        async def read_on_loop():
            started = time.monotonic()
            row = store.sessions.get(SID)
            listed = store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=5))
            return row, listed, time.monotonic() - started

        row, listed, elapsed = asyncio.run(read_on_loop())
        assert elapsed < 2.0, f"loop-thread read waited {elapsed:.2f}s on the token writer"
        assert row is not None and row.session_id == SID
        assert any(entry.session.session_id == SID for entry in listed)
    finally:
        release.set()
    assert store.sessions.get(SID).input_tokens == 7
