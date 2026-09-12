"""Changing the storage sentinel must preserve an indexed message's display identity."""

import json

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("content", [
    [{"type": "text", "text": "original multimodal message"}],
    {"text": "original structured message", "metadata": {"source": "user"}},
    "original\x00text",
])
def test_legacy_indexed_content_deduplicates_after_compaction(tmp_path, content):
    db = SessionDB(tmp_path / "legacy.db")
    try:
        db.create_session("session", "cli")
        row_id = db.append_message("session", "assistant", content, timestamp=10)
        # Reproduce an indexed row from a pre-PostgreSQL SQLite installation.
        db._write_sql("UPDATE messages SET content = ? WHERE id = ?", (
            content if isinstance(content, str) else db._CONTENT_JSON_PREFIX_LEGACY + json.dumps(content), row_id,
        ))
        db.get_messages("session", include_compacted=True, limit=50)
        identity_before = db._read_one("SELECT display_identity FROM messages WHERE id = ?", (row_id,))[0]
        db.archive_and_compact("session", [
            {"role": "user", "content": "summary", "timestamp": 20},
            {"role": "assistant", "content": content, "timestamp": 10},
        ])
        displayed = db.get_messages("session", include_compacted=True, limit=50)
        assert sum(message["content"] == content for message in displayed) == 1
        copied = db._read_one("SELECT display_identity FROM messages WHERE active = 1 AND role = 'assistant'")
        assert copied[0] == identity_before
    finally:
        db.close()
