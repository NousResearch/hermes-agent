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
import logging
from pathlib import Path
from types import SimpleNamespace
import uuid

from gateway.session_contract import Principal
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

_PENDING = ('queued', 'claimed')
# Owner of a Bot Chat the gateway mints for a peer's create-if-missing (no native client asked).
_PEER_SUBJECT = 'api-peer'
# The whole body ``hermes peer dm`` sends (``hermes_cli/subcommands/peer.py::_ensure_bot_chat``).
_PEER_CREATE_FIELDS = frozenset({'title', 'source'})


def _authority(runner):
    from gateway.session_authorities import active_authority
    return active_authority(runner) if getattr(runner, 'session_authority', None) is not None else None


async def create_peer_bot_chat(adapter, body):
    """``POST /api/sessions`` titled "Bot Chat" (``hermes peer dm``'s create-if-missing) under session
    authority: the owner mints the chat through the step a first ``bot_relay.deliver`` takes
    (``session_bot._create_bot_chat``), so a headless gateway's Bot Chat is a LOCAL session its FIFO
    owns, not an API row no remote sender can ever bind (#139370). None when the ordinary handler
    answers: no authority, another title, an explicit id, or the chat exists (its title conflict).
    The owner mints on the profile's own policy, so a create carrying anything else (a model
    lock, route, options, system prompt) is refused by name before anything is written."""
    from tools.bot_mode_probe import BOT_CHAT_TITLE
    authority = _authority(adapter.gateway_runner)
    if (authority is None or str(body.get('title') or '').strip() != BOT_CHAT_TITLE
            or body.get('id') or body.get('session_id')):
        return None
    from aiohttp import web
    from gateway.platforms.api_server import _error_response
    if unsupported := sorted(set(body) - _PEER_CREATE_FIELDS):
        return _error_response(
            f"Bot Chat creation refused: bot_chat_parameters_unsupported ({', '.join(unsupported)}); "
            "the Bot Chat is created with the profile's own model and prompt", 400,
            code='bot_chat_parameters_unsupported')
    from gateway.session_bot import _create_bot_chat
    from gateway.session_runtime_workers import track_mutation
    actor = Principal(_PEER_SUBJECT, authority.profile_id,
                      frozenset({'session:create', 'session:read', 'session:submit'}), 'api-peer-dm')
    try:
        authority._require_admission_open()
        ref = await asyncio.shield(track_mutation(authority, _create_bot_chat(authority, actor)))
    except RuntimeStoreError as exc:
        return _error_response(f'Bot Chat creation refused: {exc.reason}', 503, code=exc.reason)
    if ref is None:
        return None
    row = await asyncio.to_thread(authority.db.get_session, ref.session_id)
    return web.json_response({'object': 'hermes.session', 'session': adapter._session_response(row)}, status=201)


def _refusal(authority, tip, exc):
    """``(reason, error)`` for a refused peer turn. A ``not_found`` on a Bot Chat that is no local
    chat is named: a remote sender never binds unowned history, so a bare code would leave the
    peer nothing to act on. The full clause stays in gateway.log."""
    reason = exc.reason
    if reason == 'not_found' and not str(tip.get('chat_id') or '').startswith('local-'):
        from hermes_state_local_migration import validate_legacy_row
        try:
            validate_legacy_row(authority.db, tip)
            reason, hint = 'bot_chat_unbound', 'open it once with `hermes chat -c "Bot Chat"` on that host'
        except RuntimeStoreError:
            from gateway.session_bot import displaceable_bot_chat
            reason = 'bot_chat_not_local'
            if displaceable_bot_chat(authority, tip) is not None:
                hint = "it is a conversation the gateway cannot own; it is moved aside as 'Bot Chat (<source> <id>)' on the next delivery"
            else:  # local history a peer may not adopt (a served remote login's Desktop row, another profile's): never moved
                hint = 'it is local history owned by another login or profile; open or rename it from the client that owns it'
        logger.warning('Peer DM into Bot Chat %s refused: %s (not_found: field=%s detail=%s)',
                       tip['id'], reason, exc.field, exc.detail)
        return reason, f"Bot Chat admission refused: {reason} (this profile's 'Bot Chat' is not a local chat; {hint})"
    return reason, f'Bot Chat admission refused: {reason}'


async def reset_local_title_owner(db, title, include_hidden, source=None):
    """``[row]`` for the exact-title lookup of a LOCAL conversation that was reset (``/new``, the
    compression-exhaustion auto-reset), else ``[]``. The listing shows a reset lineage as its latest,
    untitled segment, so ``hermes peer dm`` stopped finding its Bot Chat and its create collided
    with the title. The titled creation id still owns the FIFO and is what the chat door admits.
    It applies the listing's own filters: hidden only with ``include_hidden``, never an archived
    row (a deliberate archive stays put; the reaper-archive resurrection is the caller's), and
    ``source`` when one is given."""
    return await asyncio.to_thread(_reset_local_title_owner, db, title, include_hidden, source) if title else []


def _reset_local_title_owner(db, title, include_hidden, source=None):
    row = db.get_session_by_title(title)
    if (row is None or row.get('end_reason') != 'session_reset' or row.get('archived')
            or (row.get('hidden') and not include_hidden) or (source and row.get('source') != source)):
        return []
    from hermes_state_local import local_receipt
    try:
        lineage = local_receipt(db, row['id']).get('lineage') or []
    except RuntimeStoreError:
        return []
    return [row] if len(lineage) > 1 else []


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


async def admit_peer_turn(runner, session_id, message, author):
    """``(profile home, receipt)`` when *session_id* is the Bot Chat and the turn was admitted to its
    FIFO; None when the target is any other session (the caller runs it as an API turn)."""
    from gateway.session_authorities import served_profile_name
    from gateway.session_bot import deliver
    authority = _authority(runner)
    if authority is None:
        return None
    tip = bot_chat_tip(authority, session_id)
    if tip is None:
        return None
    home = Path(authority.db.db_path).parent.resolve()
    from gateway.session_bot import displaceable_bot_chat
    if displaceable_bot_chat(authority, authority.db.get_session_by_title('Bot Chat') or tip) is not None:
        # The title is held by an API conversation no FIFO can own (#140769, MSG-21): the owner
        # moves it aside and mints the local Bot Chat for the peer, as ``create_peer_bot_chat`` does.
        actor = Principal(_PEER_SUBJECT, authority.profile_id,
                          frozenset({'session:create', 'session:read', 'session:submit'}), 'api-peer-dm')
    else:
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
        reason, error = _refusal(authority, tip, exc)
        record = {'status': 'failed', 'delivery_id': key, 'reason': reason, 'error': error}
    return home, record


async def await_live_delivery(adapter, home, record, timeout, *, should_stop=None):
    """Wait for a live Bot Chat receipt through whichever door admitted it: the authority FIFO
    (in-process waiter) or the legacy owner mailbox (``await_delivery_async``)."""
    authority = _authority(adapter.gateway_runner)
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
        if live == record['admission_id'] and not interim and current['status'] == 'failed' \
                and not current.get('retry_admission_id'):
            continue  # The original settled failed after the live read: re-derive (its retry may be owed).
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
