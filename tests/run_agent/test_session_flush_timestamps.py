"""Regression coverage for timestamps assigned during append-only persistence."""

from __future__ import annotations

from types import SimpleNamespace

from run_agent import AIAgent


class _RecordingSessionDB:
    def __init__(self):
        self.batches: list[list[dict]] = []

    def append_messages_batch(self, *, session_id, messages, compression_lock_holder=None):
        self.batches.append(list(messages))


def test_flush_stamps_live_message_with_the_durable_write_timestamp():
    """A later transcript rewrite must retain the original event timestamp."""
    db = _RecordingSessionDB()
    agent = SimpleNamespace(
        _persist_disabled=False,
        _session_db=db,
        _session_db_created=True,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _flushed_db_message_ids=set(),
        _db_flush_scan_prefix=None,
        _active_compression_lock_holder=None,
        session_id="session",
    )
    messages = [{"role": "user", "content": "keep this timestamp"}]

    assert AIAgent._flush_messages_to_session_db_unlocked(agent, messages) is True

    assert isinstance(messages[0]["timestamp"], float)
    assert len(db.batches) == 1
    assert db.batches[0][0]["timestamp"] == messages[0]["timestamp"]
