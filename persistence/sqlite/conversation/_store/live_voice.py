"""Write-ahead GPT-Live text, receipts, and atomic publication into the owning chat."""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ResolvedVoiceUtterance:
    row_id: int
    message_uid: str
    text: str


# Desktop's JS /\s+/ whitespace set differs from Python's (e.g. BOM and NEL).
_PROMPT_WHITESPACE = re.compile(r'[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]+')


def _matches_prompt(text, expected_prompt):
    return isinstance(text, str) and _PROMPT_WHITESPACE.sub(' ', text).strip(' ') == expected_prompt


def _content_digest(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def _observed_group_is_exact(conn, call_id, rows, *, through=None):
    first, last = rows[0]['seq'], rows[-1]['seq']
    # The last user fragment in the marker prefix belongs to the claim. A
    # reply after the candidate does not make an earlier request current.
    wire_count = conn.execute('''SELECT count(*) FROM live_voice_fragments
        WHERE call_id=? AND seq BETWEEN ? AND ?''', (call_id, first, last)).fetchone()[0]
    if wire_count != len(rows) or conn.execute('''SELECT 1 FROM live_voice_fragments
        WHERE call_id=? AND seq>? AND (? IS NULL OR seq<=?) AND role='user' LIMIT 1''',
        (call_id, last, through, through)).fetchone():
        return False

    # The times only reject overlap with the other speaker; they never order
    # text, split one wire-order run at a pause, or cut it at a delegation
    # offset. Provider deltas leave ordinary gaps between words. The open right
    # edge is the observed wire prefix: a later event remains pending, never
    # extends a settled claim.
    if any(row['start_ms'] is None or row['start_ms'] >= row['end_ms'] for row in rows):
        return False
    start = min(row['start_ms'] for row in rows)
    end = max(row['end_ms'] for row in rows)

    previous = conn.execute('''SELECT role FROM live_voice_fragments
        WHERE call_id=? AND seq<? ORDER BY seq DESC LIMIT 1''', (call_id, first)).fetchone()
    if previous is not None and previous['role'] == 'user':
        return False
    # Check the entire adjacent reply run, not just its last (or first) delta:
    # arrival order does not imply chronological order of the timestamps.
    if conn.execute('''SELECT 1 FROM live_voice_fragments
        WHERE call_id=? AND seq<? AND (end_ms IS NULL OR end_ms>?)
          AND seq > COALESCE((SELECT max(seq) FROM live_voice_fragments
                              WHERE call_id=? AND seq<? AND role='user'), 0) LIMIT 1''',
        (call_id, first, start, call_id, first)).fetchone():
        return False
    return not conn.execute('''SELECT 1 FROM live_voice_fragments
        WHERE call_id=? AND seq>? AND (? IS NULL OR seq<=?) AND (start_ms IS NULL OR start_ms<?) LIMIT 1''',
        (call_id, last, through, through, end)).fetchone()


def _pending_prefix(conn, call_id, before):
    return conn.execute('''SELECT * FROM live_voice_fragments
        WHERE call_id=? AND seq<? AND text IS NOT NULL ORDER BY seq''', (call_id, before)).fetchall()


def _adopt_projection(conn, session_id, call_id, receipt, admission_id, expected_prompt, require_tail):
    if receipt['partial'] or receipt['admission_id'] not in (None, admission_id):
        raise Conflict('live voice claim projection', call_id)
    row = conn.execute('''SELECT id, message_uid, role, content FROM messages
        WHERE session_id=? AND id=? AND active=1''', (session_id, receipt['row_id'])).fetchone()
    if row is None or row['role'] != 'user' or not row['message_uid']:
        raise Conflict('live voice claim row', call_id)
    content = _decode_content(row['content'])
    if not _matches_prompt(content, expected_prompt) or _content_digest(content) != receipt['content_digest']:
        raise Conflict('live voice claim row', call_id)
    if require_tail or receipt['admission_id'] is None:
        if conn.execute('SELECT 1 FROM messages WHERE session_id=? AND active=1 AND id>? LIMIT 1',
                        (session_id, row['id'])).fetchone():
            raise Conflict('live voice claim tail', call_id)
    if receipt['admission_id'] is None:
        conn.execute('UPDATE live_voice_projections SET admission_id=? WHERE call_id=? AND projection=?',
                     (admission_id, call_id, receipt['projection']))
    return ResolvedVoiceUtterance(row['id'], row['message_uid'], content)


def resolve_claim(holder, conn: sqlite3.Connection, *, session_id: str, call_id: str,
                  event_ids: tuple[str, ...], admission_id: str, expected_prompt: str,
                  delegation_id: str | None = None, require_pinned_tail: bool = False) -> ResolvedVoiceUtterance:
    """Resolve exact observed user coverage inside the caller's BEGIN IMMEDIATE.

    The caller owns authority/generation and transcript-write admission, and
    must roll back on refusal. ``session_id`` is its verified physical target.
    A first adoption must be the live tail; exact replays return the same still
    active row, even after later appends. Never follow a copied/compacted row.
    The FIFO requires the original capture target and a live tail even on re-claim.
    """
    if not conn.in_transaction:
        raise RuntimeError('Voice claims require an existing write transaction')
    if not admission_id or not expected_prompt or not event_ids or len(set(event_ids)) != len(event_ids):
        raise Conflict('live voice claim identity', call_id)
    call = _session(conn, call_id)
    if (call['incomplete'] or (require_pinned_tail and call['session_id'] != session_id)
            or _writable_target(holder, conn, call['session_id']) != session_id):
        raise Conflict('live voice claim target', call_id)

    projection = _projection_key(event_ids, False)
    reserved = conn.execute('SELECT projection FROM live_voice_projections WHERE call_id=? AND admission_id=?',
                            (call_id, admission_id)).fetchone()
    if reserved is not None and reserved['projection'] != projection:
        raise Conflict('live voice claim replay', call_id)
    rows = _fragment_rows(conn, call_id, event_ids)
    seqs = [row['seq'] for row in rows]
    through = None
    if delegation_id is not None:
        marker = conn.execute('SELECT fragment_seq FROM live_voice_markers WHERE call_id=? AND delegation_id=?',
                              (call_id, delegation_id)).fetchone()
        if marker is None or max(seqs) > marker['fragment_seq']:
            raise Conflict('live voice claim marker', call_id)
        through = marker['fragment_seq']
    if seqs != sorted(seqs) or any(row['role'] != 'user' for row in rows):
        raise Conflict('live voice claim order or role', call_id)

    if all(row['text'] is None and row['projection'] == projection for row in rows):
        receipt = conn.execute('SELECT * FROM live_voice_projections WHERE call_id=? AND projection=?',
                               (call_id, projection)).fetchone()
        if receipt is not None:
            if reserved is None and (not _observed_group_is_exact(conn, call_id, rows, through=through)
                                     or _pending_prefix(conn, call_id, seqs[0])):
                raise Conflict('live voice claim utterance', call_id)
            return _adopt_projection(conn, session_id, call_id, receipt, admission_id, expected_prompt,
                                     require_pinned_tail)
    if reserved is not None or any(row['text'] is None or row['projection'] is not None for row in rows):
        raise Conflict('live voice claim coverage', call_id)
    if (not _matches_prompt(''.join(row['text'] for row in rows), expected_prompt)
            or not _observed_group_is_exact(conn, call_id, rows, through=through)):
        raise Conflict('live voice claim utterance', call_id)

    # A blocked ordinary projection may have left a completed exchange pending.
    # Append it before the claimed row in the same transaction, never past the
    # marker or ahead of the claim's validation.
    preceding = []
    for _, run in groupby(_pending_prefix(conn, call_id, seqs[0]), key=lambda row: row['role']):
        group = list(run)
        key = _projection_key(tuple(row['event_id'] for row in group), False)
        preceding.append((key, _consume_fragments(conn, call_id, group, key, partial=False)))
    message = _consume_fragments(conn, call_id, rows, projection, partial=False)
    _append_messages_in_transaction(holder, conn, session_id, [prior for _, prior in preceding] + [message])
    for key, prior in preceding:
        _remember_projection(conn, call_id, key, prior)
    _remember_projection(conn, call_id, projection, message, admission_id)
    return ResolvedVoiceUtterance(message['_row_id'], message['message_uid'], message['content'])
