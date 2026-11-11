"""Desktop's message reactions (``message.react``) on the owner's transcript.

The sidecar handler looked the session up in its own table, which never holds an authority
session, and the ``message.*`` namespace never falls back to it, so every reaction was rolled back
as "Could not react". The write is the same ``SessionDB.set_message_reaction`` (Tapback semantics,
rows of the session's visible resume lineage only), on the conversation's current physical session.
"""
from functools import partial

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError

_FIELDS = frozenset({'session_id', 'profile', 'row_id', 'newest_role', 'emoji', 'author'})


def _parse(params):
    row_id, newest_role, emoji = params.get('row_id'), params.get('newest_role'), params.get('emoji')
    author = params.get('author') or 'user'
    if (set(params) - _FIELDS or (row_id is not None and type(row_id) is not int)
            or (row_id is None and newest_role not in {'user', 'assistant'})
            or (emoji is not None and (not isinstance(emoji, str) or not emoji.strip()))
            or author not in {'user', 'agent'}):
        raise RuntimeStoreError('invalid_params')
    return row_id, newest_role, emoji.strip() if emoji else None, author


def _write(db, target, row_id, newest_role, emoji, author):
    if row_id is None:
        row_id = db.latest_message_row_id(target, role=newest_role)
        if row_id is None:
            return None, None
    return row_id, db.set_message_reaction(target, row_id, emoji, author=author)


async def react(connection, ref, params):
    row_id, newest_role, emoji, author = _parse(params)
    authority = connection.authority
    if not ref.session_id or ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:submit')
    from gateway.session_ancillary import physical_target
    from gateway.session_runtime_workers import tracked_write
    # A reset or compression moves the conversation's rows onto a new physical session; the
    # reaction lineage check runs from that tip, as the sidecar's submit-row target did.
    target = physical_target(authority, ref)
    row_id, reactions = await tracked_write(authority, partial(
        _write, authority.db, target, row_id, newest_role, emoji, author))
    if reactions is None:
        raise RuntimeStoreError('not_found')
    return {'row_id': row_id, 'reactions': reactions}
