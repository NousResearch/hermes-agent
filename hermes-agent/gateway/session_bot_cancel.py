"""``bot_relay.cancel``: a ``message_agent`` sender withdraws its own delivery by ``delivery_id``.

The cancel runs on the delivery's owner under the same mailbox lock that admits the delivery's one
automatic retry (``gateway/session_bot.py::_maybe_retry``), so it always sees which admission is the
request now. A sender that picked an admission from the receipt file could race that retry: the
first attempt already failed, the retry not yet recorded, and "nothing to cancel" while it runs.

Whatever this finds, a retry not admitted yet is withdrawn for good (the receipt's ``retry`` marks
it refused ``cancelled``), so no later wake, re-delivery or owner restart admits it.
"""
import asyncio

from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, get_session_admission

_FIELDS = frozenset({'id', 'profile', 'author_id'})


async def cancel(connection, params):
    from gateway.session_authorities import require_serving
    from gateway.session_runtime_workers import track_mutation
    require_serving(connection.authority)
    return await asyncio.shield(track_mutation(connection.authority, _cancel(connection, params)))


async def _cancel(connection, params):
    from gateway.session_bot import (_delivery_id, _delivery_result, _home, _retry_admission, _retry_eligible,
                                     _retry_identity)
    from gateway.session_bot_mailbox import mailbox_lock, read_receipt, remember_receipt, write_receipt
    authority, actor = connection.authority, connection.actor
    if set(params) - _FIELDS:
        raise RuntimeStoreError('invalid_params')
    home = _home(authority, actor, params.get('profile'))
    try:
        key = _delivery_id(params.get('id'))
    except ValueError as exc:
        raise RuntimeStoreError('invalid_params') from exc
    async with mailbox_lock(authority):
        _, record = await read_receipt(home, key)
        if record is None or not record.get('admission_id'):
            raise RuntimeStoreError('not_found')
        # Only the delivery's own sender: the principal it was admitted on and the bot it named.
        if record['principal_id'] != actor.subject or (record.get('author') or {}).get('id') != params.get('author_id'):
            raise RuntimeStoreError('permission_denied')
        retry_id = _retry_admission(authority, record)
        current = retry_id or record['admission_id']
        row = get_session_admission(authority.db, admission_id=current)
        live = row is not None and row['status'] != 'terminal'
        withdrawn = retry_id is None and (live or _retry_eligible(authority, record))
        if withdrawn:
            record['retry'] = {**(record.get('retry') or {}), 'identity': _retry_identity(key), 'refused': 'cancelled'}
        status = 'finished'
        if live:
            answer = await connection.cancel(SessionRef(authority.profile_id, record['session_id']),
                                             {'admission_id': current})
            status = 'stopping'  # a running turn was told to stop; its receipt reports how it ended
            if answer['status'] == 'terminal':
                status = 'cancelled' if answer.get('outcome') == 'cancelled' else 'finished'
        elif withdrawn:
            status = 'cancelled'  # its first attempt failed; the retry it was owed never starts
        record.update(_delivery_result(authority, record))
        await write_receipt(home, record)
        remember_receipt(authority, home, record)
        return {'delivery_id': key, 'status': status, 'receipt_status': record['status']}
