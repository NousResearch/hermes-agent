"""Transaction-scoped rewind shared by storage and canonical mutations."""
from hermes_state_messages import _ACTIVE_IDS_SQL, _SET_COUNTERS_SQL, _placeholders


def rewind_in_transaction(self, conn, session_id, target_message_id, *,
                          preserve_compaction_handoff=False, expected_active_ids=None,
                          expected_target_content=None):
    self._check_transcript_write_guards(
        conn, session_id, None, reject_active_turn_lease=True, reject_active_compression_lock=True)
    if expected_active_ids is not None:
        active_rows = conn.execute(_ACTIVE_IDS_SQL, (session_id,)).fetchall()
        if [int(r[0]) for r in active_rows] != expected_active_ids:
            raise RuntimeError("active transcript changed before the rewind could be persisted")
    row = conn.execute(
        "SELECT * FROM messages WHERE id = ? AND session_id = ?", (target_message_id, session_id)).fetchone()
    if row is None:
        raise ValueError(f"message {target_message_id} not found in session {session_id}")
    target_row = dict(row)
    if target_row.get("role") != "user":
        raise ValueError(
            f"rewind target must be a 'user' message (got role={target_row.get('role')!r}, id={target_message_id})")
    replacement_message_id = replacement = None
    if preserve_compaction_handoff or expected_target_content is not None:
        replacement = self._split_rewind_target(target_row, expected_target_content, preserve_compaction_handoff)
    ids = [r[0] for r in conn.execute("SELECT id FROM messages WHERE session_id = ? AND id >= ? AND active = 1",
                                     (session_id, target_message_id)).fetchall()]
    if ids:
        conn.execute(f"UPDATE messages SET active = 0 WHERE id IN ({_placeholders(ids)})", ids)
    if replacement is not None:
        self._insert_message_rows(conn, session_id, [replacement])
        replacement_message_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute(
        "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 WHERE id = ?", (session_id,))
    message_count, tool_call_count = self._active_transcript_counts(conn, session_id)
    conn.execute(f"{_SET_COUNTERS_SQL} WHERE id = ?", (message_count, tool_call_count, session_id))
    head_id = conn.execute(
        "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1", (session_id,)).fetchone()[0]
    return target_row, ids, head_id, replacement_message_id
