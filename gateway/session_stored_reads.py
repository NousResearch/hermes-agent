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
            or any(type(part) is not int or not 0 <= part < 2 ** 63 for part in key)):
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
        window = keys[start:end]
        return rows[start:end], window, start > 0, window[0] if window else None
    start = 0 if after is None else exact + 1 if exact is not None else next(
        (index for index, key in enumerate(durable) if key > after[1]), len(keys))
    window = keys[start:start + count]
    return rows[start:start + count], window, start + count < len(rows), window[-1] if window else None


# The current display slot of the message whose first durable row is :durable: the newest shown copy of
# its identity (the live one first; a carried tail's copy when the original is removed), else the shown
# summary carrier that wraps it (same ``message_uid``; a carrier's content, so its identity, is new), else
# that row itself while it is shown. Each read uses its own index: the row by primary key, its identity's
# shown copies on idx_messages_display_identity, its live carrier on idx_messages_session_uid.
_CURSOR_ROW_SQL = ("SELECT display_identity, display_order, active, compacted, message_uid FROM messages"
                   " WHERE id = :durable AND session_id = :sid")
_CURSOR_COPY_SQL = (
    "SELECT display_order FROM messages INDEXED BY idx_messages_display_identity WHERE session_id = :sid"
    " AND display_identity = :identity AND (active = 1 OR compacted = 1) AND display_order IS NOT NULL"
    " ORDER BY active DESC, id DESC LIMIT 1")
_CURSOR_CARRIER_SQL = (
    "SELECT display_order FROM messages INDEXED BY idx_messages_session_uid WHERE session_id = :sid"
    " AND message_uid = :uid AND active = 1 AND _compressed_summary = 1 AND display_order IS NOT NULL"
    " ORDER BY id DESC LIMIT 1")


def _current_position(conn, segment, durable):
    """The slot ``_CURSOR_*_SQL`` find for *durable*, or None (the message was rewound or deleted)."""
    row = conn.execute(_CURSOR_ROW_SQL, {'sid': segment, 'durable': durable}).fetchone()
    if row is None:
        return None
    if row['display_identity'] is not None:
        copy = conn.execute(_CURSOR_COPY_SQL, {'sid': segment, 'identity': row['display_identity']}).fetchone()
        if copy is not None:
            return copy[0]
    if row['message_uid']:
        carrier = conn.execute(_CURSOR_CARRIER_SQL, {'sid': segment, 'uid': row['message_uid']}).fetchone()
        if carrier is not None:
            return carrier[0]
    shown = (row['active'] == 1 or row['compacted'] == 1) and row['display_order'] is not None
    return row['display_order'] if shown else None


def _placeholders(values):
    return ', '.join('?' * len(values))


def _durable_firsts(conn, segment, raw):
    """``{row id: first durable id}`` for the page rows a compaction re-wrote: a carried copy (its removed
    original has the same identity, which carries the timestamp: the (session_id, active, timestamp)
    index finds it) and a summary carrier (the first row with its ``message_uid``; only a carrier folds
    by uid, so a pruned tool result keeps its own identity, as the display dedupe keeps it)."""
    identities = list({row['display_identity'] for row in raw if row['display_identity'] is not None})
    stamps = list({row['timestamp'] for row in raw if row['timestamp'] is not None})
    by_identity = dict(conn.execute(
        "SELECT display_identity, MIN(id) FROM messages INDEXED BY idx_messages_session_active WHERE "
        f"session_id = ? AND active = 0 AND timestamp IN ({_placeholders(stamps)}) AND compacted = 0 "
        f"AND display_identity IN ({_placeholders(identities)}) GROUP BY display_identity",
        (segment, *stamps, *identities)).fetchall()) if identities and stamps else {}
    uids = list({row['message_uid'] for row in raw if row['_compressed_summary'] and row['message_uid']})
    by_uid = dict(conn.execute(
        f"SELECT message_uid, MIN(id) FROM messages WHERE session_id = ? AND message_uid IN ({_placeholders(uids)})"
        " GROUP BY message_uid", (segment, *uids)).fetchall()) if uids else {}
    return {row['id']: min(row['display_order'], by_identity.get(row['display_identity'], row['id']),
                           by_uid.get(row['message_uid'], row['id']) if row['_compressed_summary'] else row['id'])
            for row in raw}


def _segment_window(db, segment, latest, after, count):
    """One segment, keyset-paged on its display index (position = ``display_order``): only the page's rows
    are read, in one read transaction (the cursor's slot, the page and its durable ids see one
    generation). The keyset starts at the cursor message's current slot; read oldest first, a re-written
    copy of a message an earlier page showed is skipped. The rows get the conversation normalization a
    resume applies (``_rows_to_conversation``: harness turns and their curator replies, stale tool-call
    markers), with the display row before the page as context so a prompt/reply pair split by the page
    edge is still one pair. Returns ``(rows, keys, more beyond, edge key)``; the edge key is the last row
    read in reading order, so rows the normalization drops there are not read again."""
    def read(conn):
        conn.execute('BEGIN')
        try:
            position = None
            if after is not None:
                found = _current_position(conn, segment, after[1])
                position = after[0] if found is None else found
            raw = db._display_rows_from_conn(conn, segment, limit=count, latest=latest, after=position)
            firsts = _durable_firsts(conn, segment, raw)
            before = db._display_rows_from_conn(conn, segment, limit=1, latest=True,
                                                after=min(row['display_order'] for row in raw)) if raw else []
            return raw, firsts, before
        finally:
            if conn.in_transaction:
                conn.execute('ROLLBACK')
    raw, firsts, before = db._read_retrying_ioerr(read)
    keys = {row['id']: (row['display_order'], firsts[row['id']]) for row in raw}
    kept = [row for row in raw if after is None or latest or keys[row['id']][1] > after[1]]
    context = {row['id'] for row in before}
    rows = [message for message in db._rows_to_conversation(
        [*before, *kept], session_id=segment, include_ancestors=False, repair_alternation=False, include_row_ids=True)
        if message['_row_id'] not in context]
    edge = None if not raw else keys[raw[0]['id'] if latest else raw[-1]['id']]
    return rows, [keys[message['_row_id']] for message in rows], len(raw) == count, edge


def _window(db, target, latest, after, count):
    """``(rows, keys, more beyond, edge key, segments read)``: up to *count* display rows past *after* in
    reading order, in display order (see ``_segment_window`` for the edge key)."""
    from hermes_state import SessionResumeTooLargeError
    lineage = db._resume_lineage_ids(target)
    if len(lineage) > 1:
        try:
            db.assert_resume_safe(target)
        except SessionResumeTooLargeError:
            pass  # past the resume's guard: the current segment only
        else:
            return (*_lineage_window(db, target, latest, after, count), lineage)
    if not db._ensure_display_order(target):
        raise RuntimeStoreError('storage_unavailable', detail='the transcript has no display index')
    return (*_segment_window(db, target, latest, after, count), [target])


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


def _cut(rows, latest, taken):
    """Half of a page of *taken* rows, from its far edge, at a tool-group boundary when one exists: the
    rows dropped never start with a tool result whose call stays (oldest first), and the rows kept never
    start with one whose call goes (newest first)."""
    keep = max(1, taken // 2)
    if latest:
        while keep > 1 and rows[-keep].get('role') == 'tool':
            keep -= 1
    else:
        while keep > 1 and rows[keep].get('role') == 'tool':
            keep -= 1
    return keep


# What survives of a row too large for a page on its own: who said it, when, and as much text as fits.
_BOUNDED_KEYS = ('role', 'timestamp', 'row_id', 'display_kind', 'name', 'tool_call_id')


def _size(value):
    return len(json.dumps(value, default=str))


def _bounded(message, budget):
    """*message* cut to *budget* bytes of serialized JSON: its identity keys and the head of its text,
    marked ``truncated``. Every other field (reasoning, tool arguments, metadata) is dropped."""
    text = next((message[key] for key in ('text', 'content') if isinstance(message.get(key), str)), '')
    kept = {key: message[key] for key in _BOUNDED_KEYS if message.get(key) is not None}
    kept['truncated'] = True
    chars = len(text)
    while True:
        trial = {**kept, 'text': text[:chars]}
        if _size(trial) <= budget or chars == 0:
            return trial
        chars //= 2


def _call_context(db, page, segments):
    """The calls of the tool results on *page* whose call is not before them on it (it sits on an earlier
    page), as one empty assistant row: projected first, it labels those results and shows nothing itself.
    Each result is paired with its own call occurrence, as ``_rows_to_conversation`` pairs them: the one
    whose durable tool-call uid it carries, else the nearest earlier call with its id (provider ids repeat)."""
    seen, wanted = set(), {}
    for row in page:
        if row.get('role') == 'assistant':
            seen.update(call.get('id') for call in row.get('tool_calls') or () if isinstance(call, dict))
        elif row.get('role') == 'tool' and row.get('tool_call_id') and row['tool_call_id'] not in seen:
            wanted.setdefault(row['tool_call_id'], (row['_row_id'], row.get('_tool_call_uid')))
    if not wanted:
        return []
    with db._read_ctx() as conn:
        found = conn.execute(
            "SELECT tool_calls, tool_call_uids FROM messages WHERE session_id IN (SELECT value FROM json_each(?))"
            " AND role = 'assistant' AND tool_calls IS NOT NULL AND id < ? ORDER BY id DESC LIMIT 16",
            (json.dumps(segments), max(first for first, _ in wanted.values()))).fetchall()
    calls = {}
    for raw_calls, raw_uids in found:  # nearest first
        try:
            listed, uids = json.loads(raw_calls), json.loads(raw_uids) if raw_uids else {}
        except (TypeError, ValueError):
            continue
        for call in listed if isinstance(listed, list) else ():
            call_id = call.get('id') if isinstance(call, dict) else None
            if call_id not in wanted or call_id in calls:
                continue
            uid = wanted[call_id][1]
            if uid and not (isinstance(uids, dict) and uids.get(call_id) == uid):
                continue  # another occurrence of a reused provider id (``tool_call_uids``: id -> uid)
            calls[call_id] = call
    return [{'role': 'assistant', 'content': '', 'tool_calls': list(calls.values())}] if calls else []


def _fit(project, rows, latest, taken):
    """``(projected rows, taken)``: halve the page from its far edge (``_cut``) until its JSON fits
    ``PAGE_BYTES``; a single row still over it is ``_bounded``."""
    while True:
        page = rows[-taken:] if latest else rows[:taken]
        projected = project(page)
        if _size(projected) <= PAGE_BYTES:
            return projected, taken
        if taken == 1:
            budget = PAGE_BYTES // max(1, len(projected)) - 2
            return [_bounded(message, budget) for message in projected], taken
        taken = _cut(rows, latest, taken)


def read_page(db, target, order, after, limit, *, profile_home, inline_images=False):
    """``(projected rows, next key)`` of one page; the rows in display order either way."""
    from gateway.session_controls import display_projection
    latest = order == 'latest'
    count = limit + _TOOL_RUN
    rows, keys, beyond, edge, segments = _window(db, target, latest, after, count)

    def project(page):
        return display_projection(_call_context(db, page, segments) + page, profile_home, image_urls=inline_images)
    projected, taken = _fit(project, rows, latest, _take(rows, latest, limit))
    if taken < len(rows):
        return projected, keys[-taken] if latest else keys[taken - 1]
    return projected, edge if beyond else None


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
    segment carrying it) are clones of its stored row, same ``message_uid`` and same stored content, and
    collapse into its first row; two compactions never share both, even with the same prose. Reads only
    the rows that can hold a handoff."""
    from agent.compaction_display import compaction_summary
    lineage = json.dumps(db._resume_lineage_ids(target))  # its own read: never inside the one below
    with db._read_ctx() as conn:
        rows = conn.execute(_HANDOFF_CANDIDATES_SQL, {'ids': lineage}).fetchall()
    found, seen = [], set()
    for row in rows:
        occurrence = (row['message_uid'], row['content'])
        if occurrence in seen:
            continue
        message = db._row_to_message_dict(row, warn_context='session.compactions', summary_flag=True)
        summary = compaction_summary(message)
        if summary is not None:
            seen.add(occurrence)
            found.append({'timestamp': message.get('timestamp'), **summary})
    return found


async def compactions(connection, ref, params):
    _check_params(params, {'session_id', 'profile'})
    target = _readable_target(connection, ref, params)
    db = connection.authority.db
    return {'session_id': ref.session_id,
            'compactions': await asyncio.to_thread(stored_compactions, db, target)}
