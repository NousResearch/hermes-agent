"""Session preview must skip the adapter ``channel_context`` backfill and show the user's typed turn.

Regression for #148988: ``gateway/run_inbound._prepare_inbound_message_text`` stores a user turn as
``{channel_context}\\n\\n[New message]\\n{text}``, so the preview's head-char SELECT painted the
context block (e.g. ``[Room id: X]``) instead of what the user typed. The fix reuses
``_sql_after_marker`` (already used for the summary end marker) to take the text after the
``[New message]`` marker the gateway writes in one place, so the preview does not depend on
adapter text.
"""
import sqlite3

from hermes_state import SessionDB
from hermes_state_common import _shape_preview, _sql_preview_raw


def _preview_for(db_path, session_id):
    """Read the ``_preview_raw`` column the listing queries use, then shape it the same way."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            f"SELECT {_sql_preview_raw()} FROM sessions s WHERE s.id = ?", (session_id,)
        ).fetchone()
    finally:
        conn.close()
    return _shape_preview(row[0] if row else "")


def test_preview_skips_channel_context_block(tmp_path):
    """A user turn stored with the gateway's ``[New message]`` marker previews the typed text,
    not the context block head."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="ctx_session", source="cli", model="m")
    stored = "[Room id: X]\n\n[New message]\nfix the login bug"
    db.append_message("ctx_session", role="user", content=stored)
    db.close()

    preview = _preview_for(tmp_path / "state.db", "ctx_session")
    assert "fix the login bug" in preview
    assert "[Room id" not in preview
    assert "[New message]" not in preview


def test_preview_unchanged_for_rows_without_marker(tmp_path):
    """A normal user message with no marker is previewed exactly as before."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="plain_session", source="cli", model="m")
    db.append_message("plain_session", role="user", content="just a normal message")
    db.close()

    assert _preview_for(tmp_path / "state.db", "plain_session") == "just a normal message"


def test_preview_short_typed_turn_after_context_block(tmp_path):
    """A short typed turn after a long context block is shown whole, not truncated to the block."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="short_session", source="cli", model="m")
    db.append_message(
        "short_session",
        role="user",
        content="[Context: backfill line one two three]\n\n[New message]\nhi",
    )
    db.close()

    assert _preview_for(tmp_path / "state.db", "short_session") == "hi"
