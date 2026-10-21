"""Rewriting an accepted user row retains durable projection guarantees."""
import copy
import json

import pytest

from agent.redact import redact_sensitive_text
from hermes_state import SessionDB


@pytest.mark.parametrize("structured", [False, True])
def test_public_user_content_rewrite_redacts_without_mutating_input(tmp_path, structured):
    secret = "ghp_" + "B" * 36
    assert secret not in redact_sensitive_text(secret, force=True)
    raw = [{"type": "text", "text": "credential=" + secret}] if structured else "credential=" + secret
    before = copy.deepcopy(raw)
    path = tmp_path / "rewrite.db"
    db = SessionDB(path)
    try:
        db.create_session("rewrite", source="cli")
        row_id = db.append_message("rewrite", "user", content="accepted")
        assert db.set_user_message_content("rewrite", row_id, raw) == 1
        assert raw == before
    finally:
        db.close()
    reopened = SessionDB(path)
    try:
        row = reopened._conn.execute("SELECT content FROM messages WHERE id = ?", (row_id,)).fetchone()
        assert secret not in row["content"]
        assert secret not in json.dumps(reopened.get_messages("rewrite"))
        assert reopened._conn.execute(
            "SELECT 1 FROM messages_fts WHERE messages_fts MATCH ?", (f'"{secret}"',)
        ).fetchone() is None
    finally:
        reopened.close()
