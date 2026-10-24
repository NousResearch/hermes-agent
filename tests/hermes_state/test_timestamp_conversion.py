"""Unconvertible timestamps use the existing fallback without losing session content."""

import json
import time
from datetime import datetime, timezone

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("invalid", [10**400, -(10**400), datetime.min],
                         ids=["positive-overflow", "negative-overflow", "pre-epoch-datetime"])
def test_message_writers_fall_back_when_timestamp_conversion_fails(tmp_path, invalid):
    db = SessionDB(db_path=tmp_path / "state.db")
    valid = datetime(2024, 1, 1, tzinfo=timezone.utc)
    try:
        db.create_session("s", "cli")
        before = time.time()
        db.append_message("s", "user", "single", timestamp=invalid)
        db.append_messages_batch("s", [
            {"role": "assistant", "content": "batch", "timestamp": invalid},
            {"role": "user", "content": "valid", "timestamp": valid},
        ])
        after = time.time()
        messages = db.get_messages("s")
    finally:
        db.close()

    assert [msg["content"] for msg in messages] == ["single", "batch", "valid"]
    assert all(before <= msg["timestamp"] <= after for msg in messages[:2])
    assert messages[2]["timestamp"] == valid.timestamp()


@pytest.mark.parametrize("invalid", [10**400, -(10**400)],
                         ids=["positive-overflow", "negative-overflow"])
def test_json_import_falls_back_for_unconvertible_session_and_message_times(tmp_path, invalid):
    valid = 1_700_000_000.0
    payload = json.loads(json.dumps([
        {"id": "bad", "started_at": invalid, "messages": [
            {"role": "user", "content": "preserved", "timestamp": invalid},
        ]},
        {"id": "good", "started_at": valid, "messages": [
            {"role": "user", "content": "unchanged", "timestamp": valid},
        ]},
    ]))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        before = time.time()
        result = db.import_sessions(payload)
        after = time.time()
        assert result["ok"], result
        exported = {session["id"]: session for session in db.export_all()}
    finally:
        db.close()

    assert exported.keys() == {"bad", "good"}
    assert exported["bad"]["messages"][0]["content"] == "preserved"
    assert before <= exported["bad"]["started_at"] <= after
    assert before <= exported["bad"]["messages"][0]["timestamp"] <= after
    assert exported["good"]["started_at"] == valid
    assert exported["good"]["messages"][0]["timestamp"] == valid
