"""Regression tests for #94736 — append_message after close() raced a live writer.

Subagent/cron sessions were dying mid-run with ``Session DB append_message
failed: 'NoneType' object has no attribute 'execute'``: a teardown owner
(cron ``run_job``'s ``finally`` block, a delegate timeout owner abandoning
its worker, ``AIAgent.close()``) called ``SessionDB.close()`` — which nulls
``_conn`` — while a still-unwinding worker thread had one more transcript
flush to land. ``_execute_write`` then hit ``None.execute`` and the
conversation loop force-ended the turn as ``session_persistence_failed``,
silently dropping the tail of the session.

The fix self-heals at the shared persistence boundary: when ``_conn`` is
``None`` (only possible after an explicit ``close()``), the writer reopens
a connection to the same database file with a loud WARNING, and the write
lands. These tests exercise the REAL SessionDB against a real sqlite file —
no mocks of the code under test.
"""

import logging
import threading

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


class TestAppendAfterClose:
    def test_append_message_after_close_reopens_and_lands(self, db, caplog):
        """The exact #94736 shape: close() then one more transcript flush."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", content="hello")

        db.close()
        assert db._conn is None

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            msg_id = db.append_message(
                "s1", "assistant", content="flushed after teardown"
            )

        assert isinstance(msg_id, int) and msg_id > 0
        rows = db.get_messages("s1")
        assert [r["role"] for r in rows] == ["user", "assistant"]
        assert rows[-1]["content"] == "flushed after teardown"
        # The recovery is loud, not silent.
        assert any("reopening" in r.message for r in caplog.records)

    def test_reopened_connection_survives_subsequent_writes_and_close(self, db):
        """The reopened handle is a full writer: more appends work, and a
        second close() releases it cleanly (idempotent contract)."""
        db.create_session("s1", "cli")
        db.close()

        db.append_message("s1", "user", content="a")
        db.append_message("s1", "assistant", content="b")
        assert len(db.get_messages("s1")) == 2

        db.close()
        assert db._conn is None
        db.close()  # idempotent

    def test_read_after_close_recovers_too(self, db):
        """The locked-read fallback path shares the same guard: a read that
        lands after close() must not die on None.execute either."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", content="hello")
        db.close()

        rows = db.get_messages("s1")
        assert len(rows) == 1

    def test_concurrent_close_during_flush_loses_no_writes(self, tmp_path):
        """Race a teardown close() against a worker mid-flush (the cron
        inactivity-timeout shape): every append must land or raise loudly —
        never vanish into a swallowed NoneType error."""
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", "cli")

        n_writes = 40
        start = threading.Event()
        errors: list = []

        def _worker():
            start.wait()
            for i in range(n_writes):
                try:
                    db.append_message("s1", "tool", content=f"result {i}")
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

        t = threading.Thread(target=_worker)
        t.start()
        start.set()
        # Teardown owner closes mid-flight, twice for good measure.
        db.close()
        db.close()
        t.join(timeout=30)
        assert not t.is_alive()

        assert errors == [], f"appends died during teardown race: {errors!r}"
        rows = db.get_messages("s1")
        assert len(rows) == n_writes
        db.close()

    def test_read_only_handle_still_refuses_after_close(self, tmp_path):
        """A read-only cross-profile handle must NOT silently reopen — it
        raises an explicit error naming the closed handle."""
        # Initialise a real DB first with a writable handle.
        writer = SessionDB(db_path=tmp_path / "state.db")
        writer.create_session("s1", "cli")
        writer.append_message("s1", "user", content="hello")
        writer.close()

        ro = SessionDB(db_path=tmp_path / "state.db", read_only=True)
        ro.close()
        with pytest.raises(Exception) as excinfo:
            ro.get_messages("s1")
        assert "closed" in str(excinfo.value).lower()
