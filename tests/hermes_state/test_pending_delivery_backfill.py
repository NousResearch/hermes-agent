"""Structured delivery references survive both sidecar backfill entrypoints."""

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("writer", ["latest", "row"])
def test_structured_delivery_backfill_replays_content_parts(tmp_path, writer):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("delivery", source="discord")
        original = [{"type": "text", "text": "What about this brief?"}]
        row_id = db.append_message("delivery", "user", original)
        wire = [{"type": "text", "text": "Delivered reference data"}, *original]
        if writer == "latest":
            count = db.set_latest_user_api_content("delivery", original, wire)
        else:
            count = db.set_message_api_content("delivery", row_id, original, wire)
        assert count == 1
        restored = db.get_messages_as_conversation("delivery")
        assert restored[-1]["content"] == original
        assert restored[-1]["api_content"] == wire
    finally:
        db.close()
