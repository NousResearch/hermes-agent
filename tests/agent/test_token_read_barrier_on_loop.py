"""A session read on an event-loop thread never waits on the token writer (#141966).

Gateway handlers read sessions on the loop; when the writer sits behind another connection's write
lock, waiting for it froze every socket for the whole flush budget. Off the loop the read barrier
still gives read-your-writes (``test_async_token_accounting``).
"""

import asyncio
import threading
import time

from hermes_state import SessionDB


def test_loop_thread_session_read_does_not_wait_for_a_stuck_writer(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s-loop", "test")
        db.flush_token_counts()
        entered, release = threading.Event(), threading.Event()
        original = db.update_token_counts

        def stuck(session_id, **kwargs):
            entered.set()
            release.wait(timeout=30)  # the writer is held up, like a write lock held elsewhere
            return original(session_id, **kwargs)

        db.update_token_counts = stuck
        db.queue_token_counts("s-loop", input_tokens=7, api_call_count=1)
        assert entered.wait(timeout=10)

        async def read_on_loop():
            started = time.monotonic()
            row = db.get_session("s-loop")
            listed = db.list_sessions_rich(limit=5)
            return row, listed, time.monotonic() - started

        row, listed, elapsed = asyncio.run(read_on_loop())
        assert elapsed < 2.0, f"loop-thread read waited {elapsed:.2f}s on the token writer"
        assert row is not None and row["id"] == "s-loop"
        assert any(r["id"] == "s-loop" for r in listed)

        release.set()
        db.update_token_counts = original
        # Off the loop the read barrier still waits, so the queued delta is visible.
        assert db.get_session("s-loop")["input_tokens"] == 7
    finally:
        release.set()
        db.close()
