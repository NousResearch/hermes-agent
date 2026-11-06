"""``session.transcript``: a stored conversation's display rows, read without attaching (#150091), paged
both ways, polled forward from a cursor, with its compaction handoffs as rows; with ``subagent_id``, the
same read of a delegated child of the session (``gateway/session_transcript_subagent.py``).

A picker previews a conversation before it opens it. ``session.resume`` would subscribe the caller,
and ``session.history`` / ``session.events.since`` serve an attached session. This read builds no
agent, subscribes no viewer and writes no row.

A page is bounded three ways: ``limit`` display rows (plus at most ``_TOOL_RUN`` tool results kept
with the call that made them), ``PAGE_BYTES`` of projected JSON, and the store work behind it. A
single-segment conversation (every in-place compaction) is paged in SQL on its durable display
index, so only the page's bodies are read. A rotated lineage needs the one display dedupe over every
segment; it is read whole only while it is under ``sessions.max_resume_messages`` (the resume's own
guard), and past that only its current segment is paged, as a resume past it shows only live rows.

Forward polling (``order: 'oldest'``): the cursor also holds the segment it read and
``transcript_cursor_state`` of the rows it covered. While both still hold, only appends happened and the
read continues after the cursor; otherwise (an in-place compaction, a rotation, a rewind, a rewrite or
deletion of a covered row) the reply is ``reset`` and starts again from the first page. An oldest-order
cursor is never null, so a caught-up reader keeps it and polls; an empty page means caught up.
"""
import asyncio
import base64
import binascii
import json
from functools import partial

from gateway.session_busy_controls import authorize
from gateway.session_transcript_rows import project_with_handoffs
from hermes_state_runtime import RuntimeStoreError

MAX_PAGE_ROWS = 500
# Projected JSON per page: well under the gateway clients' 8 MiB frame (hermes_cli/gateway_client.py),
# with room for the reply envelope; the resume reply's archive budget is 3 MiB.
PAGE_BYTES = 2 * 1024 * 1024
# Tool results kept on their call's page beyond ``limit`` (a tool row is labelled from its call).
_TOOL_RUN = 64
_CURSOR_TAG = 't3'
_CURSOR_MAX_CHARS = 1024
_INT_MAX = 2 ** 63
# Reads retried while a non-append change lands during one (each attempt is one bounded page).
_STABLE_READS = 4
TRANSCRIPT_FIELDS = frozenset({'session_id', 'profile', 'subagent_id', 'cursor', 'limit', 'order', 'inline_images'})


def handlers(connection):
    return {'session.transcript': partial(transcript, connection)}


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


def _uint(value):
    return type(value) is int and 0 <= value < _INT_MAX


def _encode_cursor(binding, key, poll):
    """*binding*: ``(session_id, subagent_id, order)``. *key*: the page edge ``(position, durable)``, None
    for the start (an oldest-order read of an empty transcript). *poll*: ``(segment, through, state)`` of an
    oldest-order read, else None."""
    segment, through, state = poll if poll is not None else (None, None, None)
    payload = [_CURSOR_TAG, *binding, list(key) if key is not None else [], segment, through,
               list(state) if state is not None else None]
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')


def _decode_cursor(cursor, binding):
    """``{key, segment, through, state}`` of a cursor this read issued for *binding*; anything else is
    ``invalid_params``."""
    try:
        if len(cursor) > _CURSOR_MAX_CHARS:
            raise ValueError('cursor too long')
        tag, session_id, subagent_id, order, key, segment, through, state = json.loads(
            base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
    except (TypeError, ValueError, binascii.Error):
        raise RuntimeStoreError('invalid_params', field='cursor') from None
    oldest = binding[2] == 'oldest'
    if ((tag, session_id, subagent_id, order) != (_CURSOR_TAG, *binding) or not isinstance(key, list)
            or len(key) not in ((0, 2) if oldest else (2,)) or not all(_uint(part) for part in key)
            or (oldest and not (isinstance(segment, str) and segment and _uint(through) and isinstance(state, list)
                                and len(state) == 3 and all(_uint(part) for part in state)))
            or (not oldest and (segment, through, state) != (None, None, None))):
        raise RuntimeStoreError('invalid_params', field='cursor')
    return {'key': tuple(key) or None, 'segment': segment, 'through': through,
            'state': tuple(state) if state is not None else None}


# A row's key is ``(position, durable)``: where the read orders it, and the id of its message's first durable
# row, a superseded original's included (a compaction that carries the protected tail writes new rows and
# marks the originals removed). A later compaction never changes ``durable``, so a page boundary holds.


def _lineage_window(db, target, latest, after, count, lineage):
    """A rotated lineage: the one display dedupe over every segment (``get_messages_as_conversation
    (include_ancestors=True, include_compacted=True)``), keyed by first durable id. The page starts
    next to the row the cursor names, wherever a compaction since moved it."""
    first_ids = {}
    rows = db._dedupe_display_generations(db._fetch_conversation_rows(
        lineage, '', with_session_id=False, with_compacted=True),
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
    original has the same identity) and a summary carrier (the first row with its ``message_uid``; only a
    carrier folds by uid, so a pruned tool result keeps its own identity, as the display dedupe keeps it).
    A copy and a carrier keep the original's timestamp (``_insert_message_rows`` stamps it on the live
    dict for exactly this), so both reads stay on the (session_id, active, timestamp) index."""
    identities = list({row['display_identity'] for row in raw if row['display_identity'] is not None})
    stamps = list({row['timestamp'] for row in raw if row['timestamp'] is not None})
    by_identity = dict(conn.execute(
        "SELECT display_identity, MIN(id) FROM messages INDEXED BY idx_messages_session_active WHERE "
        f"session_id = ? AND active = 0 AND timestamp IN ({_placeholders(stamps)}) AND compacted = 0 "
        f"AND display_identity IN ({_placeholders(identities)}) GROUP BY display_identity",
        (segment, *stamps, *identities)).fetchall()) if identities and stamps else {}
    carriers = [row for row in raw if row['_compressed_summary'] and row['message_uid'] and row['timestamp'] is not None]
    uids, carrier_stamps = list({row['message_uid'] for row in carriers}), list({row['timestamp'] for row in carriers})
    by_uid = dict(conn.execute(
        "SELECT message_uid, MIN(id) FROM messages INDEXED BY idx_messages_session_active WHERE session_id = ?"
        f" AND active = 0 AND timestamp IN ({_placeholders(carrier_stamps)}) AND message_uid IN ({_placeholders(uids)})"
        " GROUP BY message_uid", (segment, *carrier_stamps, *uids)).fetchall()) if carriers else {}
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
    # Oldest order continues only from a cursor whose rows did not change shape (``_poll``), so the page
    # is every slot past it: a durable id is not monotonic in display order (a handoff's new row sits
    # before the carried copies of older messages).
    kept = raw
    context = {row['id'] for row in before}
    rows = [message for message in db._rows_to_conversation(
        [*before, *kept], session_id=segment, include_ancestors=False, repair_alternation=False, include_row_ids=True)
        if message['_row_id'] not in context]
    edge = None if not raw else keys[raw[0]['id'] if latest else raw[-1]['id']]
    return rows, [keys[message['_row_id']] for message in rows], len(raw) == count, edge


def _window(db, target, latest, after, count, lineage):
    """``(rows, keys, more beyond, edge key, segments read)``: up to *count* display rows past *after* in
    reading order, in display order (see ``_segment_window`` for the edge key). *lineage*: the segments
    the transcript displays (a conversation's resume lineage, a delegated child's own)."""
    from hermes_state import SessionResumeTooLargeError
    if len(lineage) > 1:
        try:
            db.assert_resume_safe(target)
        except SessionResumeTooLargeError:
            pass  # past the resume's guard: the current segment only
        else:
            return (*_lineage_window(db, target, latest, after, count, lineage), lineage)
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
_BOUNDED_KEYS = ('row_kind', 'role', 'timestamp', 'row_id', 'display_kind', 'name', 'tool_call_id', 'compacted_messages')


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


def _call_uids(calls, uids):
    """Each call's own durable uid, in order: ``tool_call_uids`` maps a provider id to its uid, or to a list
    of uids, one per occurrence, when folded turns repeat the id in one row (``index_tool_call_uids``)."""
    from agent.message_sanitization import coalesce_tool_call_id
    seen, out = {}, []
    for call in calls:
        call_id = coalesce_tool_call_id(call)
        uid = uids.get(call_id) if isinstance(uids, dict) else None
        if isinstance(uid, list):
            nth = seen[call_id] = seen.get(call_id, -1) + 1
            uid = uid[nth] if nth < len(uid) else None
        out.append(uid if isinstance(uid, str) and uid else None)
    return out


def _paired_call(calls, uids, result_id, result_uid):
    """The call of *calls* (one stored row) that a result with *result_id* / *result_uid* answers: the one
    whose own uid it carries, else (a result without a uid) the row's last call naming its id."""
    from agent.message_sanitization import tool_call_id_variants, tool_result_id_variants
    variants = set(tool_result_id_variants(result_id))
    matches = [(call, uid) for call, uid in zip(calls, _call_uids(calls, uids))
               if variants & set(tool_call_id_variants(call))]
    if result_uid:
        return next((call for call, uid in matches if uid == result_uid), None)
    return matches[-1][0] if matches else None


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
        listed = [call for call in listed if isinstance(call, dict)] if isinstance(listed, list) else []
        for result_id, (_, result_uid) in wanted.items():
            if result_id not in calls and (call := _paired_call(listed, uids, result_id, result_uid)) is not None:
                calls[result_id] = call
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


def read_page(db, target, order, after, limit, *, profile_home, inline_images=False, lineage=None):
    """``(projected rows, next key, through)`` of one page, the rows in display order either way;
    *through* is the highest stored row id the page read. A window whose rows all project to nothing
    (hidden scaffolding) is read past, so an empty page is the end."""
    from gateway.session_controls import display_projection
    latest = order == 'latest'
    count = limit + _TOOL_RUN
    lineage = lineage or db._resume_lineage_ids(target)
    through = 0
    while True:
        rows, keys, beyond, edge, segments = _window(db, target, latest, after, count, lineage)

        def project(page):
            return project_with_handoffs(db, page, segments, lambda shown: display_projection(
                _call_context(db, page, segments) + shown, profile_home, image_urls=inline_images))
        projected, taken = _fit(project, rows, latest, _take(rows, latest, limit))
        page = rows[len(rows) - taken:] if latest else rows[:taken]
        through = max([through, *(row['_row_id'] for row in page if isinstance(row.get('_row_id'), int))])
        if taken < len(rows):
            next_key, more = keys[-taken] if latest else keys[taken - 1], True
        else:
            # Latest order: null past the oldest page. Oldest order: the edge, or where the read started,
            # so a caught-up reader keeps a cursor to poll from.
            next_key = (edge if beyond else None) if latest else (edge if edge is not None else after)
            more = beyond
        if projected or not more:
            return projected, next_key, through
        after = next_key


def _poll(db, target, lineage, order, held, limit, **page):
    """``(rows, next key, poll state, reset)``. Latest order pages back from the cursor's message wherever
    a compaction moved it. Oldest order continues after the cursor only while its segment and
    ``transcript_cursor_state`` still hold, else starts again (``reset``). The state stored for the next
    poll is read between two snapshots of the rows that existed before the page read; a non-append change
    landing in between reads the page again, so a cursor never vouches for rows it did not see."""
    if order == 'latest':
        rows, key, _ = read_page(db, target, order, held and held['key'], limit, lineage=lineage, **page)
        return rows, key, None, held is None
    for _ in range(_STABLE_READS):
        mark = db.get_active_message_watermark(target)
        before = db.transcript_cursor_state(target, mark, lineage)
        holds = (held is not None and held['segment'] == target
                 and db.transcript_cursor_state(target, held['through'], lineage) == held['state'])
        after = held['key'] if holds else None
        rows, key, through = read_page(db, target, order, after, limit, lineage=lineage, **page)
        through = max(through, held['through'] if holds else 0)
        state = db.transcript_cursor_state(target, through, lineage)
        if db.transcript_cursor_state(target, mark, lineage) == before:
            return rows, key if key is not None else after, (target, through, state), not holds
    raise RuntimeStoreError('storage_unavailable', detail='the transcript kept changing during the read')


def read_transcript(db, target, lineage, binding, held, limit, *, profile_home, inline_images):
    """The reply fields ``rows``, ``next_cursor`` and ``reset`` of one page of physical *target*."""
    rows, key, poll, reset = _poll(db, target, lineage, binding[2], held, limit,
                                   profile_home=profile_home, inline_images=inline_images)
    next_cursor = _encode_cursor(binding, key, poll) if key is not None or poll is not None else None
    return {'rows': rows, 'next_cursor': next_cursor, 'reset': reset}


def _transcript_params(params):
    """``(limit, order, inline_images, subagent_id, cursor)`` of a valid request."""
    if set(params) - TRANSCRIPT_FIELDS or not isinstance(params.get('session_id'), str) or not params['session_id']:
        raise RuntimeStoreError('invalid_params')
    limit, cursor, subagent_id = params.get('limit', 100), params.get('cursor'), params.get('subagent_id')
    order, inline_images = params.get('order', 'oldest'), params.get('inline_images', False)
    if (type(limit) is not int or not 1 <= limit <= MAX_PAGE_ROWS or not isinstance(cursor, (str, type(None)))
            or order not in ('oldest', 'latest') or type(inline_images) is not bool
            or not (subagent_id is None or (isinstance(subagent_id, str) and subagent_id))):
        raise RuntimeStoreError('invalid_params')
    return limit, order, inline_images, subagent_id, cursor


async def transcript(connection, ref, params):
    limit, order, inline_images, subagent_id, cursor = _transcript_params(params)
    binding = (ref.session_id, subagent_id, order)
    held = None if cursor is None else _decode_cursor(cursor, binding)
    if subagent_id is not None:
        from gateway.session_transcript_subagent import subagent_transcript
        return await subagent_transcript(connection, ref, params, binding, held, limit, inline_images)
    target = _readable_target(connection, ref, params)
    authority = connection.authority

    def read():
        db = authority.db
        return {'session_id': ref.session_id, **read_transcript(
            db, target, db._resume_lineage_ids(target), binding, held, limit,
            profile_home=authority.profile_id, inline_images=inline_images)}
    return await asyncio.to_thread(read)
