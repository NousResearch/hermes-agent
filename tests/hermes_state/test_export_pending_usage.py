import json

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("include_compacted", [False, True])
@pytest.mark.parametrize("source", [None, "cli"])
def test_export_all_preserves_pending_usage_on_restore(tmp_path, monkeypatch, include_compacted, source):
    expected_usage = {
        "input_tokens": 123,
        "output_tokens": 7,
        "cache_read_tokens": 11,
        "cache_write_tokens": 13,
        "reasoning_tokens": 17,
        "api_call_count": 2,
        "estimated_cost_usd": 0.5,
        "actual_cost_usd": 0.25,
    }

    def stopped_writer():
        return

    with SessionDB(tmp_path / "original.db") as db:
        db.create_session("pending-usage", "cli")
        db.append_message("pending-usage", "user", "Preserve this conversation.")
        db.create_session("other-source", "telegram")
        monkeypatch.setattr(db, "_token_writer_loop", stopped_writer)
        db.queue_token_counts("pending-usage", **expected_usage)
        writer = db._token_writer_thread
        assert writer is not None
        writer.join(timeout=5)
        assert not writer.is_alive()

        exported = db.export_all(source=source, include_compacted=include_compacted, include_inactive=True)
        exported_ids = set()
        pending = None
        for session in exported:
            exported_ids.add(session["id"])
            if session["id"] == "pending-usage":
                pending = session
        expected_ids = {"pending-usage", "other-source"} if source is None else {"pending-usage"}
        assert exported_ids == expected_ids
        assert pending is not None
        for field, expected in expected_usage.items():
            assert pending[field] == expected

    backup = json.loads(json.dumps(exported))
    with SessionDB(tmp_path / "restored.db") as restored:
        receipt = restored.import_sessions(backup)
        assert receipt["ok"]
        assert receipt["imported"] == len(expected_ids)
        session = restored.get_session("pending-usage")
        assert session is not None
        for field, expected in expected_usage.items():
            assert session[field] == expected
        messages = restored.get_messages("pending-usage")
        assert [message["content"] for message in messages] == ["Preserve this conversation."]
