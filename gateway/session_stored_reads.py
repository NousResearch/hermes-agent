"""``session.transcript``: a stored conversation's display rows, read without attaching (#150091), paged
both ways, polled forward from a cursor, with its compaction handoffs as rows; with ``subagent_id``, the
same read of a delegated child of the session (``gateway/session_transcript_subagent.py``).

A picker previews a conversation before it opens it. ``session.resume`` would subscribe the caller,
and ``session.history`` / ``session.events.since`` serve an attached session. This read builds no
agent, subscribes no viewer and writes no row.

A page is bounded three ways: ``limit`` rows, handoff rows included (plus at most ``_TOOL_RUN`` tool
results kept with the call that made them), ``PAGE_BYTES`` of projected JSON, and the store work behind
it. A row's handoffs and the row itself can fall on two pages; the cursor then names the row and how many
of them were sent. A
single-segment conversation (every in-place compaction) is paged in SQL on its durable display
index, so only the page's bodies are read. A rotated lineage needs the one display dedupe over every
segment; it is read whole only while it is under ``sessions.max_resume_messages`` (the resume's own
guard), and past that only its current segment is paged, as a resume past it shows only live rows. A
cursor holds which of the two its page read (``_scope``); a read in the other answers ``reset``. A
newest-first cursor of a current segment holds that segment too: once a compaction rotates it, ``reset``.

Forward polling (``order: 'oldest'``): the cursor also holds the segment it read and
``transcript_cursor_state`` of the rows it covered. While both still hold, only appends happened and the
read continues after the cursor; otherwise (an in-place compaction, a rotation, a rewind, a rewrite or
deletion of a covered row or of any earlier segment's row) the reply is ``reset`` and starts again from
the first page. The page and the state its cursor stores are read in one read transaction
(``SessionDB.transcript_read_snapshot``), so a cursor vouches for exactly the rows its page holds. Without
a pooled reader there is no such transaction: a read counts only when no write landed meanwhile, and
one that never settles answers ``reset`` (``_unproven``). An oldest-order cursor is never null, so a caught-up reader keeps it and polls; an empty page means caught up.
"""
import asyncio
import base64
import binascii
import json
from functools import partial

from gateway.session_busy_controls import authorize
from gateway.session_transcript_rows import (
    ROW_KIND_COMPACTION, ROW_KIND_MESSAGE, page_handoffs, project_entries, shown_occurrences,
)
from hermes_agent.hermes_state_runtime import RuntimeStoreError

MAX_PAGE_ROWS = 500
# Projected JSON per page: well under the gateway clients' 8 MiB frame (hermes_cli/gateway_client.py),
# with room for the reply envelope; the resume reply's archive budget is 3 MiB.
PAGE_BYTES = 2 * 1024 * 1024
# Tool results kept on their call's page beyond ``limit`` (a tool row is labelled from its call).
_TOOL_RUN = 64
_CURSOR_TAG = 't4'
# An earlier cursor format, without the scope: still decoded (and validated), and its scope never matches, so
# a poll from it resets.
_CURSOR_TAG_UNSCOPED = 't3'
_CURSOR_MAX_CHARS = 1024
_INT_MAX = 2 ** 63
# Reads retried while a page's read transaction ends early (each attempt is one bounded page).
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


def _encode_cursor(binding, key, poll, whole):
    """*binding*: ``(session_id, subagent_id, order)``. *key*: the page edge ``(position, durable)``, None
    for the start (an oldest-order read of an empty transcript), or ``(position, durable, n)`` when the
    page ended inside that row's group (its handoffs and the row, in reading order; *n*: ``_sent``). *poll*: ``(segment, through, state)`` of an
    oldest-order read, ``(segment, None, None)`` of a tip-only latest-order one, else None. *whole*: the scope the page was read in (``_scope``); a key means a
    slot only in it."""
    segment, through, state = poll if poll is not None else (None, None, None)
    payload = [_CURSOR_TAG, *binding, list(key) if key is not None else [], segment, through,
               list(state) if state is not None else None, whole]
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')


def _decode_cursor(cursor, binding):
    """``{key, segment, through, state, whole}`` of a cursor this read issued for *binding*; anything else
    is ``invalid_params``. An unscoped cursor (``_CURSOR_TAG_UNSCOPED``) has ``whole`` None."""
    try:
        if len(cursor) > _CURSOR_MAX_CHARS:
            raise ValueError('cursor too long')
        payload = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
        if isinstance(payload, list) and payload[:1] == [_CURSOR_TAG_UNSCOPED]:
            payload = [_CURSOR_TAG, *payload[1:], None]
        tag, session_id, subagent_id, order, key, segment, through, state, whole = payload
    except (TypeError, ValueError, binascii.Error):
        raise RuntimeStoreError('invalid_params', field='cursor') from None
    oldest = binding[2] == 'oldest'
    if ((tag, session_id, subagent_id, order) != (_CURSOR_TAG, *binding) or not isinstance(key, list)
            or not (whole is None or type(whole) is bool)
            or len(key) not in ((0, 2, 3) if oldest else (2, 3)) or not all(_uint(part) for part in key)
            or (len(key) == 3 and key[2] == 0)
            or (oldest and not (isinstance(segment, str) and segment and _uint(through) and isinstance(state, list)
                                and len(state) == 3 and all(_uint(part) for part in state)))
            or (not oldest and ((through, state) != (None, None)
                                or not (segment is None or (whole is False and isinstance(segment, str) and segment))))):
        raise RuntimeStoreError('invalid_params', field='cursor')
    return {'key': tuple(key) or None, 'segment': segment, 'through': through,
            'state': tuple(state) if state is not None else None, 'whole': whole}


# A row's key is ``(position, durable)``: where the read orders it, and the id of its message's first durable
# row, a superseded original's included (a compaction that carries the protected tail writes new rows and
# marks the originals removed). A later compaction never changes ``durable``, so a page boundary holds. A
# cursor key with a third part (``sent``) reads from that row again, inclusive.


def _lineage_window(db, target, latest, after, count, lineage):
    """A rotated lineage: the one display dedupe over every segment (``get_messages_as_conversation
    (include_ancestors=True, include_compacted=True)``), keyed by first durable id. The page starts
    next to the row the cursor names, wherever a compaction since moved it. The last part is the rows
    before the window."""
    first_ids = {}
    rows = db._dedupe_display_generations(db._fetch_conversation_rows(
        lineage, '', with_session_id=False, with_compacted=True),
        drop_removed=True, first_ids=first_ids)
    rows = db._rows_to_conversation(rows, session_id=target, include_ancestors=True,
                                    repair_alternation=False, include_row_ids=True)
    keys = [(first_ids.get(row['_row_id'], row['_row_id']),) * 2 for row in rows]
    durable = [key[1] for key in keys]
    exact = None if after is None else next((index for index, key in enumerate(durable) if key == after[1]), None)
    inclusive = int(after is not None and len(after) > 2)
    if exact is not None:
        exact += inclusive if latest else -inclusive
    if latest:
        end = len(keys) if after is None else exact if exact is not None else next(
            (index for index, key in enumerate(durable) if key >= after[1]), len(keys))
        start = max(0, end - count)
        window = keys[start:end]
        return rows[start:end], window, start > 0, window[0] if window else None, rows[:start]
    start = 0 if after is None else exact + 1 if exact is not None else next(
        (index for index, key in enumerate(durable) if key > after[1]), len(keys))
    window = keys[start:start + count]
    return rows[start:start + count], window, start + count < len(rows), window[-1] if window else None, rows[:start]


def _segment_window(db, segment, latest, after, count):
    """One segment, keyset-paged on its display index (position = ``display_order``): only the page's rows
    are read, in one read transaction (the cursor's slot, the page and its durable ids see one
    generation). The keyset starts at the cursor message's current slot; read oldest first, a re-written
    copy of a message an earlier page showed is skipped. The rows get the conversation normalization a
    resume applies (``_rows_to_conversation``: harness turns and their curator replies, stale tool-call
    markers), with the display row before the page as context so a prompt/reply pair split by the page
    edge is still one pair. Returns ``(rows, keys, more beyond, edge key, first slot read)``; the edge key is the last row
    read in reading order, so rows the normalization drops there are not read again."""
    raw, firsts, before = db.transcript_segment_page(
        segment, latest=latest, after=None if after is None else tuple(after[:2]), count=count,
        inclusive=after is not None and len(after) > 2)
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
    start = min((row['display_order'] for row in raw), default=None)
    return rows, [keys[message['_row_id']] for message in rows], len(raw) == count, edge, start


def _scope(db, target, lineage):
    """What a read of *lineage* shows, judged on the caller's reads (inside ``_poll``'s snapshot, the
    page's generation): True, the whole lineage (``_lineage_window``: rotated, and under the resume's
    guard); False, only *target*, paged on its display index; None, only *target*, whose display index
    lacks a slot (a legacy segment, or a rewrite since the last backfill cleared one), so a page would skip
    that row: ``_whole_lineage`` backfills it, a write, outside any snapshot."""
    from hermes_agent.hermes_state import SessionResumeTooLargeError
    if len(lineage) > 1:
        try:
            db.assert_resume_safe(target)
        except SessionResumeTooLargeError:
            pass  # past the resume's guard: the current segment only
        else:
            return True
    return False if db.transcript_display_index_complete(target) else None


def _whole_lineage(db, target, lineage):
    """Whether *lineage* is read whole (``_scope``), with *target*'s display index backfilled when it lacks
    a slot."""
    whole = _scope(db, target, lineage)
    if whole is None and not db._ensure_display_order(target):
        raise RuntimeStoreError('storage_unavailable', detail='the transcript has no display index')
    return bool(whole)


def _window(db, target, latest, after, count, lineage, whole=None):
    """``(rows, keys, more beyond, edge key, segments read, earlier)``: up to *count* display rows past *after* (from
    it when the key has a third part) in reading order, in display order (see ``_segment_window`` for the
    edge key). *lineage*: the segments the transcript displays (a conversation's resume lineage, a
    delegated child's own). *whole*: ``_whole_lineage``, decided here when None. ``earlier(candidates)``: the
    handoff occurrences among *candidates* that a row before the window holds (``page_handoffs``): the
    rows before it of the whole lineage, or the segment's display rows before its first slot."""
    if whole is None:
        whole = _whole_lineage(db, target, lineage)
    if whole:
        *window, prior = _lineage_window(db, target, latest, after, count, lineage)
        ids = [row['_row_id'] for row in prior if type(row.get('_row_id')) is int]
        return (*window, lineage, lambda candidates: shown_occurrences(
            message for messages in db.transcript_handoff_rows(ids, lineage).values() for message in messages))
    *window, start = _segment_window(db, target, latest, after, count)
    return (*window, [target], lambda candidates: set() if start is None else shown_occurrences(
        db.transcript_handoffs_shown_before(target, start, [(message.get('message_uid'), message.get('content'))
                                                           for message in candidates])))


def _items(rows, keys, handoffs, latest, after):
    """What the window can put on the page, in reading order: ``(row index, slot)``. A row's group is one
    item per handoff shown before it (slot: its index in ``handoffs[row]``) and one for the row itself
    (slot: its handoff count), in reading order. When the cursor key has a third part (``_sent``) and the
    window starts at the row it names, the items of the row's group that were sent already are skipped."""
    groups = []
    for index in (range(len(rows) - 1, -1, -1) if latest else range(len(rows))):
        group = [(index, slot) for slot in range(len(handoffs[index]) + 1)]
        groups.append(group[::-1] if latest else group)
    if groups and after is not None and len(after) > 2 and keys[groups[0][0][0]][1] == after[1]:
        groups[0] = groups[0][len(groups[0]) - after[2]:] if latest else groups[0][after[2]:]
    return [item for group in groups for item in group]


def _sent(handoffs, latest, item):
    """Oldest first: how many items of its row's group *item* and the ones before it are. Newest first: how many
    are left after it, the older handoffs, which a later compaction's re-wrap never shifts (it adds its
    handoff next to the row, on the side already read). None when *item* ends the group (the row is done)."""
    index, slot = item
    if latest:
        return slot or None
    return None if slot == len(handoffs[index]) else slot + 1


def _take(roles, latest, limit):
    """How many items (``_items``) this page keeps: *limit* from the reading end, plus the tool results that
    belong with the call at its far edge (read oldest first: the results after it; newest first: back to
    the call itself). *roles*: each item's role, in reading order (a handoff's is ``'compaction'``)."""
    taken = min(limit, len(roles))
    while taken < len(roles) and roles[taken - 1 if latest else taken] == 'tool':
        taken += 1
    return taken


def _cut(roles, latest, taken):
    """Half of a page of *taken* items, from its far edge, at a tool-group boundary when one exists: the
    items dropped never start with a tool result whose call stays (oldest first), and the items kept never
    end with one whose call goes (newest first)."""
    keep = max(1, taken // 2)
    while keep > 1 and roles[keep - 1 if latest else keep] == 'tool':
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
    found = db.transcript_tool_calls_before(segments, max(first for first, _ in wanted.values()))
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


def _fit(project, roles, latest, taken):
    """``(projected rows, taken)``: halve the page from its far edge (``_cut``) until its JSON fits
    ``PAGE_BYTES``; a single item still over it is ``_bounded``. ``project(taken)`` projects the first
    *taken* items."""
    while True:
        projected = project(taken)
        if _size(projected) <= PAGE_BYTES:
            return projected, taken
        if taken == 1:
            budget = PAGE_BYTES // max(1, len(projected)) - 2
            return [_bounded(message, budget) for message in projected], taken
        taken = _cut(roles, latest, taken)


def _entries(rows, handoffs, chosen):
    """The chosen items in display order: ``(row_kind, handoff row or stored row)``."""
    return [(ROW_KIND_COMPACTION, handoffs[index][slot]) if slot < len(handoffs[index]) else (ROW_KIND_MESSAGE, rows[index])
            for index, slot in sorted(chosen)]


def read_page(db, target, order, after, limit, *, profile_home, inline_images=False, lineage=None, whole=None):
    """``(projected rows, next key, through)`` of one page, the rows in display order either way;
    *through* is the highest stored row id the page read. At most *limit* rows, handoff rows included,
    plus the tool results that stay with their call. A window whose rows all project to nothing (hidden
    scaffolding) is read past, so an empty page is the end. *whole*: ``_whole_lineage``, decided here
    when None."""
    from gateway.session_controls import display_projection
    latest = order == 'latest'
    count = limit + _TOOL_RUN
    lineage = lineage or db._resume_lineage_ids(target)
    through = 0
    while True:
        rows, keys, beyond, edge, segments, earlier = _window(db, target, latest, after, count, lineage, whole)
        handoffs = page_handoffs(db, rows, segments, earlier)
        items = _items(rows, keys, handoffs, latest, after)
        roles = [rows[index].get('role') if slot == len(handoffs[index]) else ROW_KIND_COMPACTION
                 for index, slot in items]

        def project(taken):
            entries = _entries(rows, handoffs, items[:taken])
            shown = [row for kind, row in entries if kind == ROW_KIND_MESSAGE]
            return project_entries(entries, lambda marked: display_projection(
                _call_context(db, shown, segments) + marked, profile_home, image_urls=inline_images))
        projected, taken = _fit(project, roles, latest, _take(roles, latest, limit))
        through = max([through, *(rows[index]['_row_id'] for index, _ in items[:taken]
                                  if isinstance(rows[index].get('_row_id'), int))])
        if taken < len(items):
            last = items[taken - 1]
            sent = _sent(handoffs, latest, last)
            next_key, more = (keys[last[0]] if sent is None else (*keys[last[0]], sent)), True
        else:
            # Latest order: null past the oldest page. Oldest order: the edge, or where the read started,
            # so a caught-up reader keeps a cursor to poll from.
            next_key = (edge if beyond else None) if latest else (edge if edge is not None else after)
            more = beyond
        if projected or not more:
            return projected, next_key, through
        after = next_key


def _read(db, target, lineage, order, held, limit, whole, page):
    """One page in scope *whole* (``_scope``): ``(rows, next key, poll state, reset, whole)``. A cursor
    issued in another scope (the lineage crossed the resume's guard since, or an unscoped cursor) holds
    nothing: its key names a slot of the other read, and its rows are not what this read shows."""
    if held is not None and held['whole'] is not whole:
        held = None
    if order == 'latest':
        if held is not None and not whole and held['segment'] != target:
            held = None  # a tip-only key names a slot of its segment: a rotation since starts again
        rows, key, _ = read_page(db, target, order, held and held['key'], limit, lineage=lineage,
                                 whole=whole, **page)
        return rows, key, None if whole else (target, None, None), held is None, whole
    holds = (held is not None and held['segment'] == target
             and db.transcript_cursor_state(target, held['through'], lineage) == held['state'])
    after = held['key'] if holds else None
    rows, key, through = read_page(db, target, order, after, limit, lineage=lineage, whole=whole, **page)
    through = max(through, held['through'] if holds else 0)
    state = db.transcript_cursor_state(target, through, lineage)
    return rows, key if key is not None else after, (target, through, state), not holds, whole


def _poll(db, target, lineage, order, held, limit, **page):
    """``(rows, next key, poll state, reset, whole)``. Latest order pages back from the cursor's message
    wherever a compaction moved it (in the lineage read whole; a current segment's cursor resets once a
    rotation replaces that segment). Oldest order continues after the cursor only while its segment and
    ``transcript_cursor_state`` still hold, else starts again (``reset``). Either order starts again when
    the scope (``_scope``) is not the cursor's. The scope, the check, the page and the state stored for
    the next poll are read in one read transaction, so a cursor vouches for exactly the rows its page
    holds, and a change committed meanwhile is the next poll's to see. A display index missing a slot
    there (a rewrite since the backfill) is backfilled outside it and read again. Without a pooled reader
    the reads are kept only when no write landed between them; after one retry, ``_unproven``, as when
    the index never stays whole."""
    _whole_lineage(db, target, lineage)  # may backfill a legacy display index: before the snapshot
    missing = False
    for attempt in range(_STABLE_READS):
        with db.transcript_read_snapshot() as consistent:
            whole = _scope(db, target, lineage)
            missing = whole is None
            if not missing:
                result = _read(db, target, lineage, order, held, limit, whole, page)
                if consistent():
                    return result
                if not consistent.pinned and attempt:
                    return _unproven(db, target, lineage, order, limit, result, whole, page)
        if missing:
            whole = _whole_lineage(db, target, lineage)  # outside the snapshot; then read again
    if missing:  # the last attempt still found a slot missing: the backfilled index, unproven
        result = _read(db, target, lineage, order, held, limit, whole, page) if order == 'latest' else None
        return _unproven(db, target, lineage, order, limit, result, whole, page)
    raise RuntimeStoreError('storage_unavailable', detail='the transcript read kept ending early')


# A cursor state no transcript has: the next poll from it resets.
_UNPROVEN_STATE = (_INT_MAX - 1,) * 3


def _unproven(db, target, lineage, order, limit, result, whole, page):
    """A store without a pooled reader saw a write during the read and again on the retry, so no page of
    it is proven to match a cursor state. Latest order stores no state: its last read stands. Oldest
    order answers ``reset`` with the first page and a state no transcript has, so the next poll resets
    again rather than trust it: a change is never silently missed or doubled."""
    if order == 'latest':
        return result
    rows, key, through = read_page(db, target, order, None, limit, lineage=lineage, whole=whole, **page)
    return rows, key, (target, through, _UNPROVEN_STATE), True, whole


def read_transcript(db, target, lineage, binding, held, limit, *, profile_home, inline_images):
    """The reply fields ``rows``, ``next_cursor`` and ``reset`` of one page of physical *target*."""
    rows, key, poll, reset, whole = _poll(db, target, lineage, binding[2], held, limit,
                                          profile_home=profile_home, inline_images=inline_images)
    next_cursor = _encode_cursor(binding, key, poll, whole) if key is not None or binding[2] == 'oldest' else None
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
