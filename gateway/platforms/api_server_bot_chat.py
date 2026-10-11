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
    """The delivery's receipt after it settles for good, *timeout* lapses, or *should_stop* says so.

    Each pass re-reads which admission is live (``peer_wait_admission``): after a transient failure
    the owner admits ONE retry, whose id is the only future that fires again; the original's interim
    ``failed`` (retry still eligible) reads as pending, never as the answer."""
    from gateway.session_bot import peer_wait_admission
    if record['status'] not in _PENDING:
        return record
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout
    while True:
        live, interim = await asyncio.to_thread(peer_wait_admission, authority, record)
        # Register the waiter BEFORE re-reading: a settle between the two either shows in the read
        # or resolves this future; a settle before both would otherwise leave a future nobody pops.
        waiter = None if live is None or interim else authority.waiters.setdefault(live, loop.create_future())
        current = _receipt(authority, record)
        if live is None and current['status'] in _PENDING:
            continue  # The derived retry committed while the receipt was being read.
        if interim:
            current['status'] = 'claimed'  # the retry is being admitted
        if live is None or current['status'] not in _PENDING:
            await _owner_published(authority, record, deadline)
            return current
        if should_stop is not None and should_stop():
            return current
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            return current
        tick = 0.05 if interim else (0.5 if should_stop is not None else remaining)
        if waiter is None:
            await asyncio.sleep(tick if remaining is None else min(tick, remaining))
        else:
            await asyncio.wait([waiter], timeout=tick if remaining is None else min(tick, remaining))


async def _owner_published(authority, record, deadline):
    """Answer only once the owner's reply watcher published the outcome this waiter read from the
    FIFO: its receipt write runs off-loop, and a re-read of the delivery must not say ``queued``
    after the sender already holds the reply."""
    watcher = getattr(authority, '_bot_reply_watchers', {}).get(record['delivery_id'])
    if watcher is not None and not watcher[1].done():
        remaining = None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())
        await asyncio.wait([watcher[1]], timeout=remaining)


def _receipt(authority, record):
    from gateway.session_bot import _result
    return {**record, **_result(authority, record)}
