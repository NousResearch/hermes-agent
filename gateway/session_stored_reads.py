"""Reads of a stored conversation that never attach the caller (#150091): ``session.transcript``
pages its display rows, ``session.compactions`` lists the handoff summaries its compactions wrote.

A picker previews a conversation before it opens it. ``session.resume`` would subscribe the caller,
and ``session.history`` / ``session.events.since`` serve an attached session. Both reads take
``session.resume``'s own display read of the compression lineage (live rows, compaction-archived rows,
each logical message once: ``get_messages_as_conversation(include_compacted=True)``) and its
``display_messages`` projection; neither builds an agent, subscribes a viewer or writes a row.
"""
import asyncio
import base64
import binascii
import json
from functools import partial

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError

MAX_PAGE_ROWS = 500
_CURSOR_TAG = 't1'


def handlers(connection):
    return {'session.transcript': partial(transcript, connection),
            'session.compactions': partial(compactions, connection)}


def _readable_target(connection, ref, params):
    """The current physical segment of the conversation *ref* names, once the caller may read it.

    The check is ``session.resume``'s (``SessionAuthority.authorize`` for ``session:read``: profile,
    owner or operator, read-only history mounts). Like ``session.info``, it may bring a stored
    conversation's route back in memory; that builds no agent and subscribes nobody. An old local
    row (no receipt and no route, so ``authorize`` finds nothing to mount) is read by the rule the
    native picker lists it by, and is never adopted the way a resume adopts it."""
    authority = connection.authority
    try:
        authorize(connection, ref, params, 'session:read')
    except RuntimeStoreError as exc:
        from gateway.session_listing import legacy_readable
        if exc.reason != 'not_found' or not legacy_readable(authority, connection.actor, ref.session_id):
            raise
        return authority.db.get_compression_tip(ref.session_id) or ref.session_id
    from gateway.session_local_recovery import transcript_target
    return transcript_target(authority, ref)


def _check_params(params, fields):
    if set(params) - fields or not isinstance(params.get('session_id'), str) or not params['session_id']:
        raise RuntimeStoreError('invalid_params')


def _encode_cursor(key):
    if key is None:
        return None
    return base64.urlsafe_b64encode(json.dumps([_CURSOR_TAG, key]).encode()).decode().rstrip('=')


def _decode_cursor(cursor):
    try:
        tag, key = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
    except (TypeError, ValueError, binascii.Error):
        raise RuntimeStoreError('invalid_params', field='cursor') from None
    if tag != _CURSOR_TAG or type(key) is not int or key < 0:
        raise RuntimeStoreError('invalid_params', field='cursor')
    return key


def _display_rows(db, target):
    """``(rows, keys)``: the lineage's display rows in display order, and each one's sort key, the id of
    its logical message's FIRST stored row. A compaction or rotation copies a message to a new row,
    never a new first row, so a key outlives every later compaction."""
    first_ids = {}
    rows = db.get_messages_as_conversation(target, include_ancestors=True, include_compacted=True,
                                           include_row_ids=True, display_sort_ids=first_ids)
    return rows, [first_ids.get(row['_row_id'], row['_row_id']) for row in rows]


def read_page(db, target, after, limit):
    """``(stored rows, next key)``: *limit* display rows after the row keyed *after* (from the start
    when None), plus the tool results that follow the last one (a tool row is labelled from its call)."""
    rows, keys = _display_rows(db, target)
    # The row keyed *after* is skipped; when a rewind removed it, the page starts at the next key.
    start = 0 if after is None else next(
        (index + (key == after) for index, key in enumerate(keys) if key >= after), len(keys))
    end = min(start + limit, len(rows))
    while end < len(rows) and rows[end].get('role') == 'tool':
        end += 1
    return rows[start:end], (keys[end - 1] if end < len(rows) else None)


async def transcript(connection, ref, params):
    _check_params(params, {'session_id', 'profile', 'cursor', 'limit'})
    limit, cursor = params.get('limit', 100), params.get('cursor')
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE_ROWS or not isinstance(cursor, (str, type(None))):
        raise RuntimeStoreError('invalid_params')
    after = None if cursor is None else _decode_cursor(cursor)
    target = _readable_target(connection, ref, params)
    authority = connection.authority

    def read():
        from gateway.session_controls import display_projection
        rows, next_key = read_page(authority.db, target, after, limit)
        return {'session_id': ref.session_id, 'rows': display_projection(rows, authority.profile_id),
                'next_cursor': _encode_cursor(next_key)}
    return await asyncio.to_thread(read)


async def compactions(connection, ref, params):
    _check_params(params, {'session_id', 'profile'})
    target = _readable_target(connection, ref, params)
    db = connection.authority.db

    def read():
        from agent.compaction_display import compaction_summary_text
        found = []
        for row in db.get_messages_as_conversation(target, include_ancestors=True, include_compacted=True):
            summary = compaction_summary_text(row)
            if summary is not None:
                found.append({'timestamp': row.get('timestamp'), 'summary': summary})
        return {'session_id': ref.session_id, 'compactions': found}
    return await asyncio.to_thread(read)
