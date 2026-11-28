"""The user-row rewrite shared by direct writes and worker receipt transactions."""

from persistence.sqlite._schema_ddl import _MERGE_DISPLAY_META


def rewrite_user_row(conn, session_id, row_id, content, display_metadata=None):
    """Return the changed count and ownership digest from the same transaction."""
    from .messages import _encode_content, _encode_display_metadata
    from .transcript_repair import transcript_row_snapshot

    params = (_encode_content(content), row_id, session_id)
    assignments = "content = ?"
    if display_metadata is not None:
        assignments += f", api_content = NULL, display_metadata = {_MERGE_DISPLAY_META}"
        params = (params[0], _encode_display_metadata(display_metadata), *params[1:])
    count = conn.execute(
        f"UPDATE messages SET {assignments} WHERE id = ? AND session_id = ? AND role = 'user' AND active = 1",
        params).rowcount
    if not count:
        return 0, None
    row = conn.execute("SELECT * FROM messages WHERE id = ?", (row_id,)).fetchone()
    return count, transcript_row_snapshot(row)
