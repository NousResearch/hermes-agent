"""Reads of a stored conversation that never attach the caller (#150091): ``session.transcript``
pages its display rows, ``session.compactions`` lists the handoff summaries its compactions wrote.

A picker previews a conversation before it opens it. ``session.resume`` would subscribe the caller,
and ``session.history`` / ``session.events.since`` serve an attached session. Neither read builds an
agent, subscribes a viewer or writes a row.

A page is bounded three ways: ``limit`` display rows (plus at most ``_TOOL_RUN`` tool results kept
with the call that made them), ``PAGE_BYTES`` of projected JSON, and the store work behind it. A
single-segment conversation (every in-place compaction) is paged in SQL on its durable display
index, so only the page's bodies are read. A rotated lineage needs the one display dedupe over every
segment; it is read whole only while it is under ``sessions.max_resume_messages`` (the resume's own
guard), and past that only its current segment is paged, as a resume past it shows only live rows.
"""
import asyncio
import base64
import binascii
import json
from functools import partial

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError

MAX_PAGE_ROWS = 500
# Projected JSON per page: well under the gateway clients' 8 MiB frame (hermes_cli/gateway_client.py),
# with room for the reply envelope; the resume reply's archive budget is 3 MiB.
PAGE_BYTES = 2 * 1024 * 1024
# Tool results kept on their call's page beyond ``limit`` (a tool row is labelled from its call).
_TOOL_RUN = 64
_CURSOR_TAG = 't2'
_CURSOR_MAX_CHARS = 1024


def handlers(connection):
    return {'session.transcript': partial(transcript, connection),
            'session.compactions': partial(compactions, connection)}


def _readable_target(connection, ref, params):
    """The current physical segment of the conversation *ref* names, once the caller may read it.

    The check is ``session.resume``'s (``SessionAuthority.authorize`` for ``session:read``: profile,
    owner or operator, read-only history mounts). Like ``session.info`` and ``session.history``, it
    publishes the conversation's route entry in ``authority.sessions`` when it is not there (a
    restored local receipt, a messaging lane, a read-only history mount); that builds no agent,
    subscribes nobody, and an ended local conversation stays out of the live list (closed and
    unviewed). An old local row (no receipt and no route, so ``authorize`` finds nothing to mount) is
    judged at its compression tip by the rule the stored feed lists it by and a resume adopts it by,
    and is never adopted."""
    authority = connection.authority
    try:
        authorize(connection, ref, params, 'session:read')
    except RuntimeStoreError as exc:
        from gateway.session_listing import legacy_readable
        if exc.reason != 'not_found':
            raise
        tip = authority.db.get_compression_tip(ref.session_id) or ref.session_id
        if not legacy_readable(authority, connection.actor, tip):
            raise
        return tip
    from gateway.session_local_recovery import transcript_target
    return transcript_target(authority, ref)


def _check_params(params, fields):
    if set(params) - fields or not isinstance(params.get('session_id'), str) or not params['session_id']:
        raise RuntimeStoreError('invalid_params')


def _encode_cursor(session_id, order, key):
    if key is None:
        return None
    return base64.urlsafe_b64encode(json.dumps([_CURSOR_TAG, session_id, order, list(key)]).encode()).decode().rstrip('=')


def _decode_cursor(cursor, session_id, order):
    """The key a cursor this read issued for *session_id* in *order*; anything else is ``invalid_params``."""
    try:
        if len(cursor) > _CURSOR_MAX_CHARS:
            raise ValueError('cursor too long')
        tag, owner, issued_order, key = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
    except (TypeError, ValueError, binascii.Error):
        raise RuntimeStoreError('invalid_params', field='cursor') from None
    if ((tag, owner, issued_order) != (_CURSOR_TAG, session_id, order) or not isinstance(key, list) or len(key) != 2
            or any(type(part) is not int or part < 0 for part in key)):
        raise RuntimeStoreError('invalid_params', field='cursor')
    return tuple(key)


# A row's key is ``(position, durable)``: where the read orders it, and the id of its message's first durable
# row, a superseded original's included (a compaction that carries the protected tail writes new rows and
# marks the originals removed). A later compaction never changes ``durable``, so a page boundary holds.


def _lineage_window(db, target, latest, after, count):
    """A rotated lineage: the one display dedupe over every segment (``get_messages_as_conversation
    (include_ancestors=True, include_compacted=True)``), keyed by first durable id. The page starts
    next to the row the cursor names, wherever a compaction since moved it."""
    first_ids = {}
    rows = db._dedupe_display_generations(db._fetch_conversation_rows(
        db._resume_lineage_ids(target), '', with_session_id=False, with_compacted=True),
        drop_removed=True, first_ids=first_ids)
    rows = db._rows_to_conversation(rows, session_id=target, include_ancestors=True,
                                    repair_alternation=False, include_row_ids=True)
    keys = [(first_ids.get(row['_row_id'], row['_row_id']),) * 2 for row in rows]
    durable = [key[1] for key in keys]
    exact = None if after is None else next((index for index, key in enumerate(durable) if key == after[1]), None)
    if latest:
        end = len(keys) if after is None else exact if exact is not None else next(
            (index for index, key in enumerate(durable) if key >= after[1]), len(keys))
        start = max(0, end - count)
        return rows[start:end], keys[start:end], start > 0
    start = 0 if after is None else exact + 1 if exact is not None else next(
        (index for index, key in enumerate(durable) if key > after[1]), len(keys))
    return rows[start:start + count], keys[start:start + count], start + count < len(rows)


def _segment_window(db, segment, latest, after, count):
    """One segment, keyset-paged on its display index (position = ``display_order``): only the page's rows
    are read. Read oldest first, a carried copy of a message an earlier page showed is skipped."""
    def read(conn):
        raw = db._display_rows_from_conn(conn, segment, limit=count, latest=latest,
                                         after=None if after is None else after[0])
        # Superseded originals of the page's rows: removed rows with the same identity, which carries the
        # timestamp, so the (session_id, active, timestamp) index finds them without a segment scan.
        identities = list({row['display_identity'] for row in raw if row['display_identity'] is not None})
        stamps = list({row['timestamp'] for row in raw if row['timestamp'] is not None})
        first = dict(conn.execute(
            "SELECT display_identity, MIN(id) FROM messages INDEXED BY idx_messages_session_active WHERE "
            f"session_id = ? AND active = 0 AND timestamp IN ({', '.join('?' * len(stamps))}) AND compacted = 0 "
            f"AND display_identity IN ({', '.join('?' * len(identities))}) GROUP BY display_identity",
            (segment, *stamps, *identities)).fetchall()) if identities and stamps else {}
        return raw, first
    raw, first = db._read_retrying_ioerr(read)
    rows, keys = [], []
    for row in raw:
        key = (row['display_order'], min(row['display_order'], first.get(row['display_identity'], row['id'])))
        if after is not None and not latest and key[1] <= after[1]:
            continue
        message = db._row_to_message_dict(row, warn_context='session.transcript', summary_flag=True)
        message['_row_id'] = message['id']
        rows.append(message)
        keys.append(key)
    return rows, keys, len(raw) == count


def _window(db, target, latest, after, count):
    """``(rows, keys, more beyond)``: up to *count* display rows past *after* in reading order, in display order."""
    from hermes_state import SessionResumeTooLargeError
    if len(db._resume_lineage_ids(target)) > 1:
        try:
            db.assert_resume_safe(target)
        except SessionResumeTooLargeError:
            pass  # past the resume's guard: the current segment only
        else:
            return _lineage_window(db, target, latest, after, count)
    if not db._ensure_display_order(target):
        raise RuntimeStoreError('storage_unavailable', detail='the transcript has no display index')
    return _segment_window(db, target, latest, after, count)


def _take(rows, latest, limit):
    """How many rows of the window this page keeps: *limit* from the reading end, plus the tool results
    that belong with the call at its far edge (read oldest first: the results after it; newest first:
    back to the call itself)."""
    taken = min(limit, len(rows))
    if latest:
        while taken < len(rows) and rows[-taken].get('role') == 'tool':
            taken += 1
    else:
        while taken < len(rows) and rows[taken].get('role') == 'tool':
            taken += 1
    return taken


def _fit(project, rows, latest, taken):
    """``(projected rows, taken)``: halve the page from its far edge until its JSON fits ``PAGE_BYTES``;
    a single row still over it has its text cut and carries ``truncated: true``."""
    while True:
        page = rows[-taken:] if latest else rows[:taken]
        projected = project(page)
        if len(json.dumps(projected, default=str)) <= PAGE_BYTES:
            return projected, taken
        if taken == 1:
            for message in projected:
                for key in ('text', 'content'):
                    if isinstance(message.get(key), str) and len(message[key]) > PAGE_BYTES // 4:
                        message[key] = message[key][:PAGE_BYTES // 4] + '…'
                        message['truncated'] = True
            return projected, taken
        taken //= 2


def read_page(db, target, order, after, limit, *, profile_home, inline_images=False):
    """``(projected rows, next key)`` of one page; the rows in display order either way."""
    from gateway.session_controls import display_projection
    latest = order == 'latest'
    count = limit + _TOOL_RUN
    rows, keys, beyond = _window(db, target, latest, after, count)
    projected, taken = _fit(partial(display_projection, profile_home=profile_home, image_urls=inline_images),
                            rows, latest, _take(rows, latest, limit))
    if not (taken < len(rows) or beyond):
        return projected, None
    return projected, keys[-taken] if latest else keys[taken - 1]


async def transcript(connection, ref, params):
    _check_params(params, {'session_id', 'profile', 'cursor', 'limit', 'order', 'inline_images'})
    limit, cursor = params.get('limit', 100), params.get('cursor')
    order, inline_images = params.get('order', 'oldest'), params.get('inline_images', False)
    if (type(limit) is not int or not 1 <= limit <= MAX_PAGE_ROWS or not isinstance(cursor, (str, type(None)))
            or order not in ('oldest', 'latest') or type(inline_images) is not bool):
        raise RuntimeStoreError('invalid_params')
    after = None if cursor is None else _decode_cursor(cursor, ref.session_id, order)
    target = _readable_target(connection, ref, params)
    authority = connection.authority

    def read():
        rows, next_key = read_page(authority.db, target, order, after, limit,
                                   profile_home=authority.profile_id, inline_images=inline_images)
        return {'session_id': ref.session_id, 'rows': rows,
                'next_cursor': _encode_cursor(ref.session_id, order, next_key)}
    return await asyncio.to_thread(read)


# Rows that can hold a handoff (``ContextCompressor.classify_summary_content``'s prefixes, the merged
# delimiter, or the stored flag): live, archived, or superseded by a later compaction that carried or
# unwrapped them (a removed row whose ``message_uid`` a later row still has; a rewound row has none).
# ``compaction_summary`` confirms each in Python.
_HANDOFF_CANDIDATES_SQL = (
    "SELECT * FROM messages m WHERE m.session_id IN (SELECT value FROM json_each(:ids))"
    " AND m.role IN ('user', 'assistant') AND (m._compressed_summary = 1"
    " OR instr(m.content, '[CONTEXT COMPACTION') > 0 OR instr(m.content, '[CONTEXT SUMMARY]') > 0)"
    " AND (m.active = 1 OR m.compacted = 1 OR EXISTS (SELECT 1 FROM messages later"
    " WHERE later.session_id IN (SELECT value FROM json_each(:ids)) AND later.message_uid = m.message_uid"
    " AND later.id > m.id)) ORDER BY m.id")


def stored_compactions(db, target):
    """The lineage's compactions, oldest first. Not the display dedupe: a later compaction that reuses or
    unwraps a summary-bearing carrier keeps the human message's identity, so the display read shows one
    copy and an older handoff would be lost. Here a handoff's copies (a later generation or a rotated
    segment carrying it) share its summary text and collapse into its first row; distinct compactions
    never do. Reads only the rows that can hold a handoff."""
    from agent.compaction_display import compaction_summary
    with db._read_ctx() as conn:
        rows = conn.execute(_HANDOFF_CANDIDATES_SQL, {'ids': json.dumps(db._resume_lineage_ids(target))}).fetchall()
    found, seen = [], set()
    for row in rows:
        message = db._row_to_message_dict(row, warn_context='session.compactions', summary_flag=True)
        summary = compaction_summary(message)
        if summary is not None and summary['summary'] not in seen:
            seen.add(summary['summary'])
            found.append({'timestamp': message.get('timestamp'), **summary})
    return found


async def compactions(connection, ref, params):
    _check_params(params, {'session_id', 'profile'})
    target = _readable_target(connection, ref, params)
    db = connection.authority.db
    return {'session_id': ref.session_id,
            'compactions': await asyncio.to_thread(stored_compactions, db, target)}
