"""Atomic REST session metadata patch under its complete ownership write set."""

import time

from hermes_state_common import emit_session_deleted
from hermes_state_lineage import guarded_discovered_session_write
from hermes_state_mutation_ownership import lineage_column_ids, mutation_write_ids
from hermes_state_runtime import RuntimeStoreError


def patch_session_metadata(db, session_id, body):
    """Apply the HTTP PATCH fields together; a later refused flag cannot retain a title."""
    title = ('' if body['title'] is None else str(body['title'])) if 'title' in body else None

    def discover(conn):
        ids = {session_id}
        if any(name in body for name in ('archived', 'hidden', 'pinned', 'unread')):
            ids.update(lineage_column_ids(conn, session_id))
        if 'title' in body:
            ids.update(mutation_write_ids(db, conn, session_id, 'rename', {'title': title}))
        return ids

    def write(conn):
        row = conn.execute('SELECT archived FROM sessions WHERE id=?', (session_id,)).fetchone()
        if row is None:
            raise RuntimeStoreError('not_found')
        if 'title' in body:
            db._set_session_title_in_transaction(conn, session_id, title, source=db.TITLE_SOURCE_USER)
        archived_ids = []
        if 'archived' in body:
            archived_ids = db._set_lineage_column_in_transaction(
                conn, 'archived', session_id, int(body['archived']))
        if 'hidden' in body:
            db._set_lineage_column_in_transaction(conn, 'hidden', session_id, int(body['hidden']))
        if 'pinned' in body:
            db._set_lineage_column_in_transaction(conn, 'pinned', session_id, int(body['pinned']))
            current = conn.execute('SELECT hidden,title FROM sessions WHERE id=?', (session_id,)).fetchone()
            canonical = bool(current['hidden']) and current['title'] == db.CANONICAL_BOT_CHAT_TITLE
            if body['pinned'] and not canonical:
                db._set_lineage_column_in_transaction(conn, 'hidden', session_id, 0)
        if 'unread' in body:
            db._set_lineage_column_in_transaction(
                conn, 'last_read_at', session_id, 0.0 if body['unread'] else time.time())
        if body.get('end_reason'):
            reason = str(body['end_reason'])
            db._end_and_bump(conn,
                'UPDATE sessions SET ended_at=?,end_reason=? WHERE id=? AND ended_at IS NULL',
                (time.time(), reason, session_id), session_id, reason)
        return archived_ids, bool(body.get('archived') and archived_ids and not row['archived'])

    ids, newly_archived = guarded_discovered_session_write(db, (session_id,), discover, write)
    if newly_archived:
        emit_session_deleted(db, ids, reason='archive', primary=session_id)
