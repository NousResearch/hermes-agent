"""What a session picker shows for each ``session.list`` row, and its folder filter (#150091).

Both feeds (live and stored) list conversations by their logical id; what a picker shows comes from
the conversation's CURRENT segment, as ``list_sessions_rich`` projects a compressed chain onto its
tip: the first prompt and folder of that segment, its newest activity and its newest reply. Every
field is one batched read in a worker thread, so a 200-row page costs a few queries, never a
transcript load.
"""
from __future__ import annotations

import json

# The newest reply a reader saw in each segment: an assistant row with text, live or archived by an
# in-place compaction, never a failed turn's notice or hidden scaffolding.
_LAST_REPLY_SQL = (
    "SELECT m.session_id, m.content FROM messages m WHERE m.id IN ("
    "SELECT MAX(id) FROM messages WHERE session_id IN (SELECT value FROM json_each(?)) AND role = 'assistant'"
    " AND (active = 1 OR compacted = 1) AND content IS NOT NULL AND content <> ''"
    " AND COALESCE(display_kind, '') NOT IN ('hidden', 'failed_turn'){visible} GROUP BY session_id)")


def workspace_members(db, session_ids, workspace):
    """The ids among *session_ids* whose row is in *workspace*: the clause ``list_sessions_rich``
    filters the stored feed by (``hermes -c``'s rule: the same repo root, or at/under the folder)."""
    from hermes_state_sessions import _workspace_key_clause
    clause, params = _workspace_key_clause(workspace)
    with db._read_ctx() as conn:
        rows = conn.execute(f"SELECT s.id FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?)) AND {clause}",
                            (json.dumps(list(session_ids)), *params)).fetchall()
    return {row[0] for row in rows}


def with_picker_fields(authority, rows):
    """*rows* (``session.list`` rows, each naming its conversation by ``session_id``) with the picker
    fields of the conversation's current segment (a local receipt's entry, else the compression tip)."""
    from gateway.session_viewer_close import _physical_target
    from hermes_state_common import _shape_preview, _sql_preview_raw, _sql_session_last_active
    from hermes_state_messages import DISPLAY_VISIBLE_SQL
    from hermes_state_sessions import workspace_key
    from agent.compaction_display import compaction_summary_text
    from agent.message_content import flatten_message_text
    db = authority.db
    targets = {row['session_id']: _physical_target(authority, row['session_id']) for row in rows}
    ids = json.dumps(sorted(set(targets.values())))
    with db._read_ctx() as conn:
        segments = {row['id']: dict(row) for row in conn.execute(
            f"SELECT s.id, s.cwd, s.git_branch, s.git_repo_root, s.pinned, {_sql_preview_raw('s.id')}, "
            f"{_sql_session_last_active('s')} AS last_active FROM sessions s "
            "WHERE s.id IN (SELECT value FROM json_each(?))", (ids,)).fetchall()}
        replies = dict(conn.execute(_LAST_REPLY_SQL.format(visible=DISPLAY_VISIBLE_SQL), (ids,)).fetchall())
    for row in rows:
        target = targets[row['session_id']]
        segment = segments.get(target)
        if segment is None:
            continue  # deleted since it was listed: the row keeps what the feed read
        reply = {'role': 'assistant', 'content': db._decode_content(replies.get(target))}
        row.update(preview=_shape_preview(segment['_preview_raw']), cwd=segment['cwd'] or None,
                   git_branch=segment['git_branch'] or None, workspace=workspace_key(segment),
                   last_active_at=segment['last_active'], pinned=bool(segment['pinned']),
                   # A handoff the compressor wrote as an assistant row is no reply anyone gave.
                   last_reply=None if compaction_summary_text(reply) is not None
                   else _shape_preview(flatten_message_text(reply['content'])) or None)
    return rows


def live_picker_page(authority, sessions, limit, workspace=None):
    """The live feed's page: *sessions* (newest first) in *workspace*, cut to *limit*, with the picker fields."""
    if workspace is not None:
        members = workspace_members(authority.db, [row['session_id'] for row in sessions], workspace)
        sessions = [row for row in sessions if row['session_id'] in members]
    return with_picker_fields(authority, sessions[:limit])
