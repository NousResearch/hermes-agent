"""``transcript_revision`` keeps moving on a reaction, a display rewrite and checkpoint pruning after an older
Hermes opened the store (#150102 review). Every writable open runs its build's ``DEFERRED_INDEX_SQL`` through
``_execute_ddl_skipping_settled_triggers``, which recreates a trigger whose stored body differs from the
build's own. Every build since 1fc1999769 defines ``messages_transcript_revision_update`` with the content
columns only, so the display columns live in a trigger of their own name that an older build never touches.
"""
from hermes_state import SessionDB

# Verbatim from an older build's DEFERRED_INDEX_SQL (6b21d0564d): the pair its open runs for the shared name.
OLDER_BUILD_DDL = """DROP TRIGGER IF EXISTS messages_transcript_revision_update;
CREATE TRIGGER IF NOT EXISTS messages_transcript_revision_update
AFTER UPDATE OF role, content, api_content, tool_calls, tool_call_id, tool_name, reasoning, reasoning_content,
    reasoning_details ON messages
BEGIN
    -- An in-place rewrite (prompt expansion, multimodal context, a redaction) moves neither the
    -- active row count nor the highest id; this write-side revision is what a transcript-derived
    -- cache keys on (conversation_revision). Whoever wrote the row, old builds included.
    UPDATE sessions SET transcript_revision = transcript_revision + 1 WHERE id = new.session_id;
END;
"""


def open_as_older_build(db):
    """What an older build's writable open does to this store's revision trigger, by its own DDL path.
    Returns the statements it executed."""
    executed = []
    with db._lock:
        db._conn.set_trace_callback(executed.append)
        try:
            SessionDB._execute_ddl_skipping_settled_triggers(db._conn.cursor(), OLDER_BUILD_DDL)
        finally:
            db._conn.set_trace_callback(None)
    return executed


def _revision(db, session_id):
    return db._read_one("SELECT transcript_revision FROM sessions WHERE id = ?", (session_id,))[0]


def test_a_reaction_moves_the_revision_after_an_older_build_opened_the_store(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        row = db.append_message("s", "assistant", "answer")
        executed = open_as_older_build(db)
        # Settled for the older build too: its open drops nothing, so it takes no write lock for it.
        assert not [sql for sql in executed if sql.lstrip().upper().startswith("DROP TRIGGER")], executed
        before = _revision(db, "s")
        db.set_message_reaction("s", row, "\U0001f44d", author="user")
        assert _revision(db, "s") > before
    finally:
        db.close()
