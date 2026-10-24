"""A retained boot batch stays ahead of later transcript debt across retries."""

import pytest

from gateway.shutdown_flush import (
    flush_pending_to_file,
    recover_pending_spool,
    spool_dropped_transcript_message,
)
from hermes_state import SessionDB


def test_partial_order_write_aborts_before_replay_and_keeps_batch_order(monkeypatch, tmp_path):
    import utils

    for text in ("A pending", "B pending"):
        assert flush_pending_to_file({"key": {"session_id": "recovered", "text": text}}) == 1
    assert spool_dropped_transcript_message("recovered", {"role": "assistant", "content": "history"})
    write = utils.atomic_json_write
    calls = 0

    def fail_second_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("controlled order write failure")
        return write(*args, **kwargs)

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.ensure_session("recovered", "gateway")
        monkeypatch.setattr(utils, "atomic_json_write", fail_second_write)
        with pytest.raises(OSError, match="controlled order write failure"):
            recover_pending_spool(db)
        assert db.get_messages("recovered") == []
        monkeypatch.setattr(utils, "atomic_json_write", write)
        assert recover_pending_spool(db) == (3, set())
        assert [row["content"] for row in db.get_messages("recovered")] == [
            "history", "A pending", "B pending",
        ]
    finally:
        db.close()


def test_unresolved_retained_queue_precedes_new_transcript_debt(tmp_path):
    assert flush_pending_to_file({"key": "old pending"}) == 1
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.ensure_session("recovered", "gateway")
        assert recover_pending_spool(db, session_resolver=lambda *a, **k: None) == (0, set())
        assert spool_dropped_transcript_message(
            "recovered", {"role": "assistant", "content": "later history"},
        )
        assert recover_pending_spool(
            db, session_resolver=lambda key, not_after=None: ("recovered", db),
        ) == (2, set())
        assert [row["content"] for row in db.get_messages("recovered")] == [
            "old pending", "later history",
        ]
    finally:
        db.close()
