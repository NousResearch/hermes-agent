"""Recoverable-unarchive transaction with the exact archive write-set contract."""


def unarchive_recoverable_on(db, conn, session_id, guarded_ids, needed):
    from hermes_state_sessions import _LINEAGE_CTE_SQL

    needed.update(row[0] for row in conn.execute(
        _LINEAGE_CTE_SQL + " SELECT id FROM sessions WHERE id IN (SELECT id FROM lineage)",
        (session_id, session_id)))
    actual_tip = db._compression_tip_on(conn, session_id)
    needed.add(actual_tip)
    if needed - guarded_ids:
        return None
    current = conn.execute("SELECT archived FROM sessions WHERE id=?", (session_id,)).fetchone()
    end = conn.execute("SELECT end_reason FROM sessions WHERE id=?", (actual_tip,)).fetchone()
    if not current or not current[0] or not end or end[0] not in db.RECOVERABLE_END_REASONS:
        return False
    if not db._set_lineage_column_in_transaction(conn, "archived", session_id, 0):
        return False
    # A later deliberate archive must not inherit the accidental end stamp.
    conn.execute("UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?", (actual_tip,))
    return True
