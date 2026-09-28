"""Peer turns into the canonical Bot Chat under session authority.

``hermes peer dm`` (``/api/sessions/{id}/chat[/stream]``) and ``hermes peer run`` (``/v1/runs``)
aim at the profile's Bot Chat. Under session authority that chat is a LOCAL session whose only
writer is the authority FIFO — a Desktop showing it is a viewer, never a lease-holding executor —
so the API edge admits the turn through the same door as ``bot_relay.deliver``
(``gateway/session_bot.py``) and reads the reply from that admission's receipt. Binding the
session as an API conversation (``run_api_turn``) would be a second writer and is refused
(``permission_denied``) by design. Without an authority the legacy live-owner mailbox applies.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import uuid

from gateway.session_contract import Principal
from hermes_state_runtime import RuntimeStoreError

_PENDING = ('queued', 'claimed')


def bot_chat_tip(authority, session_id):
    """The canonical Bot Chat's tip row when *session_id* is in its lineage, else None."""
    db = authority.db
    row = db.get_session_by_title('Bot Chat')
    if row is None:
        return None
    tip = db.get_compression_tip(row['id'])
    if not tip or db.get_compression_tip(session_id) != tip:
        return None
    return db.get_session(tip)


async def admit_peer_turn(authority, session_id, message, author):
    """``(profile home, receipt)`` when *session_id* is the Bot Chat and the turn was admitted to its
    FIFO; None when the target is any other session (the caller runs it as an API turn)."""
    from gateway.session_authorities import served_profile_name
    from gateway.session_bot import deliver
    tip = bot_chat_tip(authority, session_id)
    if tip is None:
        return None
    home = Path(authority.db.db_path).parent.resolve()
    # The API key already authenticated the peer; the turn enters the chat as its owner's own
    # input, the way owner recovery re-admits queued deliveries (``recover_bot_deliveries``).
    actor = Principal(tip['user_id'], authority.profile_id,
                      frozenset({'session:submit', 'session:read'}), 'api-peer-dm')
    key = uuid.uuid4().hex
    params = dict(id=key, profile=served_profile_name(home), message=message)
    if author is not None:
        params['author'] = dict(author)
    try:
        record = await deliver(SimpleNamespace(authority=authority, actor=actor), params)
    except RuntimeStoreError as exc:
        record = {'status': 'failed', 'delivery_id': key, 'reason': exc.reason,
                  'error': f'Bot Chat admission refused: {exc.reason}'}
    return home, record


async def await_live_delivery(adapter, home, record, timeout, *, should_stop=None):
    """Wait for a live Bot Chat receipt through whichever door admitted it: the authority FIFO
    (in-process waiter) or the legacy owner mailbox (``await_delivery_async``)."""
    from gateway.session_authorities import active_authority
    authority = active_authority(adapter.gateway_runner) if getattr(
        adapter.gateway_runner, 'session_authority', None) is not None else None
    if authority is not None and record.get('admission_id'):
        return await await_peer_receipt(authority, record, timeout, should_stop=should_stop)
    from tools.bot_live_delivery import await_delivery_async
    return await await_delivery_async(home, record['delivery_id'], timeout, should_stop=should_stop) or record


async def await_peer_receipt(authority, record, timeout, *, should_stop=None):
    """The admission's current receipt after it settles, *timeout* lapses, or *should_stop* says so."""
    from gateway.session_bot import _result
    admission_id = record['admission_id']
    if record['status'] not in _PENDING:
        return record
    loop = asyncio.get_running_loop()
    # Register the waiter BEFORE re-reading: a settle between the two either shows in the read
    # or resolves this future; a settle before both would otherwise leave a future nobody pops.
    waiter = authority.waiters.setdefault(admission_id, loop.create_future())
    current = _receipt(authority, record)
    deadline = None if timeout is None else loop.time() + timeout
    while current['status'] in _PENDING and not waiter.done():
        if should_stop is not None and should_stop():
            break
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            break
        tick = 0.5 if should_stop is not None else remaining
        await asyncio.wait([waiter], timeout=tick if remaining is None else min(tick, remaining))
        current = _receipt(authority, record)
    return current


def _receipt(authority, record):
    from gateway.session_bot import _result
    from gateway.session_results import admission_result
    current = {**record, **_result(authority, record)}
    if current['status'] == 'failed' and not current.get('error'):
        saved = admission_result(authority.db, record['admission_id'])
        error = (saved or {}).get('result', {}).get('error')
        if error:
            from tools.bot_failure_reasons import classify_agent_error
            current.update(error=error, reason=classify_agent_error(error))
    return current
