"""Unconvertible timestamps respect write fallback and strict import validation."""

import json
import time
from datetime import UTC, datetime

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("invalid", [10**400, -(10**400), datetime.min],  # noqa: DTZ901 -- intentional invalid naive datetime
                         ids=["positive-overflow", "negative-overflow", "pre-epoch-datetime"])
def test_message_writers_fall_back_when_timestamp_conversion_fails(tmp_path, invalid):
    db = SessionDB(db_path=tmp_path / "state.db")
    valid = datetime(2024, 1, 1, tzinfo=UTC)
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
@pytest.mark.parametrize("field", ["started_at", "timestamp"])
def test_json_import_rejects_unconvertible_times_without_partial_writes(tmp_path, invalid, field):
    valid = 1_700_000_000.0
    good = {"id": "good", "started_at": valid, "messages": [
        {"role": "user", "content": "unchanged", "timestamp": valid},
    ]}
    bad = {"id": "bad", "started_at": valid, "messages": [
        {"role": "user", "content": "rejected", "timestamp": valid},
    ]}
    if field == "started_at":
        bad[field] = invalid
        error_path = "session.started_at"
    else:
        bad["messages"][0][field] = invalid
        error_path = "session.messages[0].timestamp"
    payload = json.loads(json.dumps([good, bad]))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("existing", "cli")
        db.append_message("existing", "user", "preserved", timestamp=valid)
        before = db.export_all()
        result = db.import_sessions(payload)
        assert result == {
            "ok": False, "imported": 0, "skipped": 0, "detached": 0,
            "errors": [{"index": 1, "session_id": "bad",
                        "error": f"{error_path} must be a supported Unix timestamp"}],
        }
        assert db.export_all() == before
        assert db.import_sessions([payload[0]])["ok"]
        exported = db.export_session("good")
        assert exported["started_at"] == valid
        assert exported["messages"][0]["timestamp"] == valid
        assert exported["messages"][0]["content"] == "unchanged"
    finally:
        db.close()
