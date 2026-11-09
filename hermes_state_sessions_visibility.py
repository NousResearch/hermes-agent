"""Direct sidebar visibility writes under complete compression-lineage ownership."""

from hermes_state_common import emit_session_deleted
from hermes_state_lineage import guarded_discovered_session_write
from hermes_state_mutation_ownership import lineage_column_ids


def set_lineage_column(db, column, session_id, value, *, extra_set_sql=""):
    """Apply one column to the full lineage; a tip-only flag can resurrect the root."""
    return bool(guarded_discovered_session_write(
        db, (session_id,), lambda conn: lineage_column_ids(conn, session_id),
        lambda conn: db._set_lineage_column_in_transaction(
            conn, column, session_id, value, extra_set_sql=extra_set_sql)))


def set_session_archived(db, session_id, archived, *, notify=True):
    """Archive the full lineage, clearing automatic provenance on a deliberate action."""
    def write(conn):
        row = conn.execute("SELECT archived FROM sessions WHERE id = ?", (session_id,)).fetchone()
        ids = db._set_lineage_column_in_transaction(
            conn, "archived", session_id, int(archived), extra_set_sql=", auto_archived = 0")
        if archived and ids:
            db._retire_canonical_bot_chat_in_transaction(conn, session_id)
        return ids, bool(archived and ids and row is not None and not row[0])
    ids, newly_archived = guarded_discovered_session_write(
        db, (session_id,), lambda conn: lineage_column_ids(conn, session_id), write)
    if newly_archived and notify:
        emit_session_deleted(db, ids, reason="archive", primary=session_id)
    return bool(ids)


def set_session_pinned(db, session_id, pinned):
    """Pin and unhide in one guarded transaction, except the canonical hidden Bot Chat."""
    def write(conn):
        row = conn.execute('SELECT hidden,title FROM sessions WHERE id=?', (session_id,)).fetchone()
        ids = db._set_lineage_column_in_transaction(conn, 'pinned', session_id, int(pinned))
        is_canonical_bot_chat = bool(row) and bool(row['hidden']) and (
            (row['title'] or '') == db.CANONICAL_BOT_CHAT_TITLE)
        if pinned and not is_canonical_bot_chat:
            db._set_lineage_column_in_transaction(conn, 'hidden', session_id, 0)
        return bool(ids)
    return guarded_discovered_session_write(
        db, (session_id,), lambda conn: lineage_column_ids(conn, session_id), write)
