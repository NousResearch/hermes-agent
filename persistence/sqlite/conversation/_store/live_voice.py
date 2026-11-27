"""Write-ahead GPT-Live text, receipts, and atomic publication into the owning chat."""

from hashlib import sha256
from itertools import groupby
import json
import math
import re
import sqlite3

from persistence.conversation.live_voice import VoiceFragment, VoiceTurn
from persistence.core.errors import Conflict, NotFound, SessionRowMissing
from persistence.core.identity import SessionId
from persistence.sqlite.conversation._store.compression import _compression_tip_on
from persistence.sqlite.conversation._store.messages import (
    _append_messages_in_transaction, _check_transcript_write_guards, _decode_content,
)


def _session(conn, call_id):
    row = conn.execute('SELECT session_id, closed, incomplete FROM live_voice_calls WHERE id=?', (call_id,)).fetchone()
    if row is None:
        raise NotFound('live voice call', call_id)
    return row


def _writable_target(holder, conn, session_id):
    target = _compression_tip_on(holder, conn, session_id)
    row = conn.execute('SELECT ended_at FROM sessions WHERE id=?', (target,)).fetchone()
    if row is None:
        raise SessionRowMissing(SessionId(target))
    if row['ended_at'] is not None:
        raise Conflict('retired live voice target', target)
    return target


def start(holder, session_id, call_id):
    if not call_id:
        raise ValueError('A voice call needs a stable identity')

    def write(conn):
        _writable_target(holder, conn, session_id)
        row = conn.execute('SELECT session_id FROM live_voice_calls WHERE id=?', (call_id,)).fetchone()
        if row is not None:
            if row['session_id'] != session_id:
                raise Conflict('live voice call', call_id)
        else:
            conn.execute('INSERT INTO live_voice_calls(id, session_id) VALUES (?,?)', (call_id, session_id))
        return call_id

    return holder._execute_write(write)


def ingest(holder, call_id, fragment: VoiceFragment):
    if not fragment.event_id or fragment.role not in ('user', 'assistant') or not fragment.text:
        raise ValueError('Invalid voice transcript fragment')
    start_ms, end_ms = fragment.start_ms, fragment.end_ms
    if (start_ms is None) != (end_ms is None) or (
        start_ms is not None and end_ms is not None
        and (not math.isfinite(start_ms) or not math.isfinite(end_ms) or not 0 <= start_ms <= end_ms)
    ):
        raise ValueError('Invalid voice transcript timing')
    timing = tuple(None if value is None else float(value) for value in (start_ms, end_ms))
    digest = sha256(json.dumps((fragment.role, fragment.text, *timing), ensure_ascii=True).encode()).hexdigest()

    def write(conn):
        call = _session(conn, call_id)
        row = conn.execute('SELECT digest FROM live_voice_fragments WHERE call_id=? AND event_id=?',
                           (call_id, fragment.event_id)).fetchone()
        if row is not None:
            if row['digest'] != digest:
                raise Conflict('live voice fragment', fragment.event_id)
            return False
        if call['closed']:
            raise Conflict('closed live voice call', call_id)
        conn.execute('''INSERT INTO live_voice_fragments(call_id,event_id,digest,role,text,start_ms,end_ms)
                     VALUES (?,?,?,?,?,?,?)''', (call_id, fragment.event_id, digest, fragment.role, fragment.text, *timing))
        return True

    return holder._execute_write(write)


def record_marker(holder, call_id, delegation_id, event_id, offset_ms):
    if not delegation_id or not event_id or (offset_ms is not None and (not math.isfinite(offset_ms) or offset_ms < 0)):
        raise ValueError('Invalid voice delegation marker')

    def write(conn):
        call = _session(conn, call_id)
        old = conn.execute('SELECT delegation_id,event_id,offset_ms FROM live_voice_markers '
                           'WHERE call_id=? AND (delegation_id=? OR event_id=?)',
                           (call_id, delegation_id, event_id)).fetchall()
        if old:
            if len(old) != 1 or tuple(old[0]) != (delegation_id, event_id, offset_ms):
                raise Conflict('live voice marker', call_id)
            return False
        if call['closed']:
            raise Conflict('closed live voice call', call_id)
        conn.execute('''INSERT INTO live_voice_markers(call_id,delegation_id,event_id,offset_ms,fragment_seq)
                        SELECT ?,?,?,?,COALESCE(MAX(seq),0) FROM live_voice_fragments WHERE call_id=?''',
                     (call_id, delegation_id, event_id, offset_ms, call_id))
        return True

    return holder._execute_write(write)


def pending(holder, call_id):
    with holder._read_ctx() as conn:
        _session(conn, call_id)
        rows = conn.execute('''SELECT event_id,role,text,start_ms,end_ms FROM live_voice_fragments
                            WHERE call_id=? AND text IS NOT NULL ORDER BY seq''', (call_id,)).fetchall()
    return tuple(VoiceFragment(*row) for row in rows)


def pending_calls(holder, *, session_id=None):
    with holder._read_ctx() as conn:
        rows = conn.execute('''SELECT id, session_id, closed FROM live_voice_calls AS c WHERE EXISTS (
            SELECT 1 FROM live_voice_fragments AS f WHERE f.call_id=c.id AND f.text IS NOT NULL
        ) ORDER BY id''').fetchall()
        return tuple((row['id'], bool(row['closed'])) for row in rows
                     if session_id is None or _compression_tip_on(holder, conn, row['session_id']) == session_id)


def invalidate_for_rewind(holder, conn, session_id):
    """Revoke calls on the rewound tip, in the rewind's own transaction."""
    calls = conn.execute('''WITH RECURSIVE ancestors(id) AS (
            SELECT ? UNION SELECT s.parent_session_id FROM sessions AS s
            JOIN ancestors AS a ON a.id=s.id WHERE s.parent_session_id IS NOT NULL
        ) SELECT c.id, c.session_id FROM ancestors AS a JOIN live_voice_calls AS c ON c.session_id=a.id
        WHERE c.closed=0 OR EXISTS (SELECT 1 FROM live_voice_fragments AS f
                                    WHERE f.call_id=c.id AND f.text IS NOT NULL)''', (session_id,)).fetchall()
    for call in calls:
        if _compression_tip_on(holder, conn, call['session_id']) != session_id:
            continue  # A fork/reset can have the same parent but is not a continuation.
        conn.execute('UPDATE live_voice_calls SET closed=1, incomplete=1 WHERE id=?', (call['id'],))
        conn.execute('UPDATE live_voice_fragments SET text=NULL WHERE call_id=? AND text IS NOT NULL', (call['id'],))


def finish(holder, call_id, *, incomplete):
    def write(conn):
        call = _session(conn, call_id)
        final_incomplete = incomplete or bool(call['incomplete'])
        conn.execute('UPDATE live_voice_calls SET closed=1, incomplete=? WHERE id=?',
                     (int(final_incomplete), call_id))
        return final_incomplete

    return holder._execute_write(write)


def _fragment_rows(conn, call_id, ids):
    rows = [conn.execute('SELECT * FROM live_voice_fragments WHERE call_id=? AND event_id=?', (call_id, event_id)).fetchone()
            for event_id in ids]
    if any(row is None for row in rows):
        raise Conflict('live voice projection coverage', call_id)
    return rows


def _projection_key(ids, partial):
    return sha256(json.dumps((ids, partial), ensure_ascii=True).encode()).hexdigest()


def _prepare(conn, call_id, turn, incomplete):
    ids = turn.event_ids
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('A voice turn needs distinct fragment IDs')
    rows = _fragment_rows(conn, call_id, ids)
    projection = _projection_key(ids, turn.partial)
    if all(row['projection'] == projection for row in rows):
        return None
    if any(row['text'] is None or row['projection'] is not None or row['role'] != rows[0]['role'] for row in rows):
        raise Conflict('live voice projection coverage', call_id)
    return projection, _consume_fragments(conn, call_id, rows, projection, partial=bool(turn.partial or incomplete))


def _consume_fragments(conn, call_id, rows, projection, *, partial):
    starts = [row['start_ms'] for row in rows]
    ends = [row['end_ms'] for row in rows]
    # The run's first provider event identifies it to the client that showed its
    # words live; the opaque call key keeps display joins inside one call (timing
    # is call-relative and restarts). Display only.
    message = {'role': rows[0]['role'], 'content': ''.join(row['text'] for row in rows),
               'display_metadata': {'voice_live': {'partial': partial,
                                                    'start_ms': None if None in starts else min(starts),
                                                    'end_ms': None if None in ends else max(ends),
                                                    'first_event_id': rows[0]['event_id'],
                                                    'call_key': sha256(call_id.encode()).hexdigest()[:16]}}}
    conn.executemany('UPDATE live_voice_fragments SET text=NULL, projection=? WHERE call_id=? AND event_id=?',
                     [(projection, call_id, row['event_id']) for row in rows])
    return message


def _remember_projection(conn, call_id, projection, message, admission_id=None):
    conn.execute('''INSERT INTO live_voice_projections(call_id, projection, row_id, content_digest, partial, admission_id)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                 (call_id, projection, message['_row_id'], _content_digest(message['content']),
                  message['display_metadata']['voice_live']['partial'], admission_id))


def publish(holder, call_id, turns: tuple[VoiceTurn, ...]):
    def write(conn):
        call = _session(conn, call_id)
        prepared = [row for turn in turns if (row := _prepare(conn, call_id, turn, call['incomplete'])) is not None]
        if not prepared:
            return 0
        target = _writable_target(holder, conn, call['session_id'])
        _check_transcript_write_guards(holder, conn, target, None, reject_active_turn_lease=True)
        inserted = _append_messages_in_transaction(holder, conn, target, [message for _, message in prepared])
        for projection, message in prepared:
            _remember_projection(conn, call_id, projection, message)
        return inserted

    return holder._execute_write(write, patience_s=holder._TRANSCRIPT_WRITE_PATIENCE_S)


def _content_digest(text: str) -> str:
    return sha256(text.encode()).hexdigest()
