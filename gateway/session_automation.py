"""Trusted producer admission; not a client-selectable internal input flag.

Reuse the private native route envelope and FIFO, retaining nonhuman turn semantics.
A producer ACK means committed input, not successful inference or outbound delivery.
"""
from copy import deepcopy
from datetime import UTC, datetime
import json
import logging

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, admit_session_input, list_session_admissions

logger = logging.getLogger(__name__)


def producer_identity(runner, event):
    identity = runner._completion_delivery_identity(event)
    if identity is None:
        raise RuntimeStoreError('invalid_params')
    return json.dumps(identity, separators=(',', ':'))


def completion_admission(runner, event):
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    if authority is None:
        return None
    entry = runner.session_store.lookup_by_session_key(str(event.get('session_key') or ''))
    if entry is None:
        return None
    from hermes_state_runtime import list_session_admissions
    identity = producer_identity(runner, event)
    for row in list_session_admissions(authority.db, session_id=_route_owner(authority, entry), pending_only=False):
        descriptor = row['payload'].get('local_automation_v1') or row['payload'].get('native_text_v1', {}).get('automation', {})
        if identity in descriptor.get('identities', [descriptor.get('identity')]):
            return row
    return None


async def deliver_heartbeat(runner, event):
    """Prepare, format and admit one liveness tick as one critical section with a completion's
    admit-and-withdraw (``admit_process_notice``): a completion that lands while this tick is on its
    way in withdraws it once it is in, and a tick checked after it sees it. True when admitted, None
    to drop it, False to requeue the original event."""
    from gateway.run import _format_gateway_process_notification
    async with _heartbeat_lock(runner):
        for _attempt in range(2):
            try:
                prepared = await prepare_heartbeat(runner, event)
            except Exception:
                # One tick's failure must not drop the rest of the drained events.
                logger.exception("Heartbeat preparation error; the tick is requeued")
                return False
            text = prepared and _format_gateway_process_notification(prepared)
            if not text:
                return None
            try:
                delivered = await runner._inject_watch_notification(text, prepared)
            except Exception:
                logger.exception("Watch notification injection error")
                return False
            if delivered is not False or not _replaced_tick_claimed(runner, prepared):
                return delivered
            # The tick it replaced started first and runs with its own output: admit this one alone.
        return False


def _replaced_tick_claimed(runner, prepared):
    from gateway.session_authorities import active_authority
    from hermes_state_runtime import get_session_admission
    replaced = prepared.get('_replaces')
    authority = active_authority(runner) if replaced else None
    row = authority and get_session_admission(authority.db, admission_id=replaced)
    return bool(row) and row['status'] != 'queued'


async def admit_process_notice(runner, authority, adapter, event, evt):
    """Admit a process notice. A completion and the withdrawal of its process's waiting ticks hold
    the heartbeat lock together; a tick already holds it (``deliver_heartbeat``)."""
    identity = producer_identity(runner, evt)
    if evt.get('type') == 'heartbeat':
        # The tick's raw output rides its stored admission: a later tick folds from the ledger.
        event.metadata['heartbeat_tick'] = {'output': str(evt.get('output') or ''),
                                            'omitted': int(evt.get('omitted') or 0)}
        if evt.get('_replaces'):
            return await authority.admit_automation(adapter, event, identity, replaces=evt['_replaces'])
    if evt.get('type') != 'completion':
        return await authority.admit_automation(adapter, event, identity)
    async with _heartbeat_lock(runner):
        receipt = await authority.admit_automation(adapter, event, identity)
        await withdraw_process_ticks(runner, evt)
        return receipt


async def prepare_heartbeat(runner, event):
    """The liveness tick to admit for ``event``, or None to drop it because its process's completion
    is already in the session: a later tick would be a stale "still running" turn. The caller holds
    the heartbeat lock.

    Otherwise a tick of the same process still queued is replaced: this one carries its stored
    output, and ``_replaces`` names it, so its cancellation commits in this tick's own admission
    (``admit_automation``) and a refused replacement leaves it queued. A tick's output is never sent
    again, so a busy session gets one waiting turn holding every tick's output instead of one turn
    per tick. A queued tick that started or settled first ran with its own output, and this one
    queues behind it. A retry of a tick already admitted is dropped."""
    import asyncio
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    entry = runner.session_store.lookup_by_session_key(str(event.get('session_key') or '')) if authority else None
    if entry is None:
        return event
    owner, process = _route_owner(authority, entry), str(event.get('session_id') or '')
    if await asyncio.to_thread(_process_completed, runner, authority, owner, event):
        logger.debug("Dropped heartbeat %s of process %s: its completion is already admitted",
                     event.get('event_id'), process)
        return None
    from hermes_state_runtime import find_session_admission
    if await asyncio.to_thread(find_session_admission, authority.db, principal_id='automation:' + entry.session_key,
                               session_id=owner, request_id=producer_identity(runner, event)):
        return None  # a retry of a tick already admitted (its acknowledgement lost)
    pending = await asyncio.to_thread(list_session_admissions, authority.db, session_id=owner)
    waiting = next((row for row in pending if _queued_tick(row, process, entry.session_key)), None)
    earlier = waiting and _descriptor(waiting['payload']).get('tick')
    if not isinstance(earlier, dict):
        return event
    from tools.process_registry_notifications import join_heartbeat_outputs
    return {**event, **join_heartbeat_outputs(earlier, event), '_replaces': waiting['admission_id']}


def _descriptor(payload):
    return payload.get('local_automation_v1') or (payload.get('native_text_v1') or {}).get('automation') or {}


def _tick_metadata(metadata):
    """A liveness tick's stored output (``heartbeat_tick``): what a later tick folds in."""
    tick = metadata.get('heartbeat_tick')
    if tick is None:
        return {}
    if (not isinstance(tick, dict) or set(tick) != {'output', 'omitted'} or not isinstance(tick['output'], str)
            or type(tick['omitted']) is not int or tick['omitted'] < 0):
        raise RuntimeStoreError('invalid_params')
    return {'tick': dict(tick)}


def _replace_queued(admission_id, session_id, replaced):
    """Write guard: cancel the replaced queued tick in the replacement's own transaction; refuse the
    replacement when that tick was claimed or settled first (its output already ran)."""
    def guard(conn):
        changed = conn.execute("UPDATE session_admissions SET status='terminal',outcome='cancelled' "
                               "WHERE admission_id=? AND target_session_id=? AND status='queued'",
                               (admission_id, session_id)).rowcount
        if changed != 1:
            raise RuntimeStoreError('stale_generation')
        replaced[:] = [admission_id]
    return guard


async def withdraw_process_ticks(runner, event):
    """A process's completion is in the session: its ticks still queued there would run after it as
    stale "still running" turns, and the completion carries the final output, so they are cancelled.
    A tick that already started is left alone. Never raises: the completion is already admitted, and
    a retry of it takes the lost-acknowledgement path, not this one. The caller holds the heartbeat
    lock (``admit_process_notice``)."""
    import asyncio
    from gateway.session_authorities import active_authority
    try:
        authority = active_authority(runner)
        entry = runner.session_store.lookup_by_session_key(str(event.get('session_key') or '')) if authority else None
        if entry is None:
            return
        finished = {identity[1] for identity in _identities(
            [producer_identity(runner, event), *(event.get('_automation_identities') or ())])
            if identity[0] == 'completion' and len(identity) > 1}
        pending = await asyncio.to_thread(
            list_session_admissions, authority.db, session_id=_route_owner(authority, entry))
        for row in pending:
            if any(_queued_tick(row, process, entry.session_key) for process in finished):
                await _withdraw(authority, row)
    except Exception:
        logger.warning("Could not withdraw the waiting heartbeats of finished process %s",
                       event.get('session_id'), exc_info=True)


def _heartbeat_lock(runner):
    # A tick's check-and-admit and a completion's admit-and-withdraw: interleaved, a tick checked
    # before the completion landed is admitted after its withdrawal and runs as a stale turn.
    import asyncio
    lock = runner.__dict__.get('_heartbeat_lock')
    if lock is None:
        lock = runner.__dict__['_heartbeat_lock'] = asyncio.Lock()
    return lock


def _process_completed(runner, authority, owner, event):
    """Whether the completion of tick ``event``'s process is admitted into the session ``owner``: by
    its exact identity (``started_at`` names the process incarnation), or as a sibling in a batch,
    pending or settled: only the batch's primary is its ``request_id``."""
    completion = runner._completion_delivery_identity(
        {'type': 'completion', 'session_id': event.get('session_id'), 'started_at': event.get('started_at')})
    if completion is None:
        return False
    identity = json.dumps(completion, separators=(',', ':'))
    with authority.db._read_ctx() as conn:
        if conn.execute('SELECT 1 FROM session_admissions WHERE target_session_id=? AND request_id=?',
                        (owner, identity)).fetchone():
            return True
    for row in list_session_admissions(authority.db, session_id=owner, pending_only=False):
        descriptor = row['payload'].get('local_automation_v1') or row['payload'].get('native_text_v1', {}).get('automation', {})
        if identity in (descriptor.get('identities') or ()):
            return True
    return False


def _identities(raws):
    parsed = []
    for raw in raws:
        try:
            identity = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(identity, list) and identity:
            parsed.append(identity)
    return parsed


def _queued_tick(row, process, route):
    """Whether *row* is the gateway's own queued liveness tick of *process* on *route*: the route's
    producer admitted ``["heartbeat", process, event_id]``. A client input whose id merely looks like
    one, and the ``/heartbeat`` wake (four parts), never match."""
    if row['status'] != 'queued' or row['principal_id'] != 'automation:' + route:
        return False
    return any(len(identity) == 3 and identity[:2] == ['heartbeat', process]
               for identity in _identities([row['request_id']]))


async def _withdraw(authority, row):
    """Cancel the gateway's own queued automation *row*; False when it was claimed or settled first."""
    from gateway.session_contract import Principal
    # The row's own principal, with the operator access a LOCAL route requires of anyone but its viewer.
    actor = Principal(row['principal_id'], authority.profile_id,
                      frozenset({'session:submit', 'session:control', 'session:operator'}), 'producer-withdraw')
    try:
        receipt = await authority.cancel_queued(
            actor, SessionRef(authority.profile_id, row['target_session_id']), row['admission_id'])
    except RuntimeStoreError:
        return False
    return receipt.outcome == 'cancelled'


def _route_owner(authority, entry):
    # Compression moves the route's transcript to a child; admissions stay on the logical root.
    return (entry.origin.chat_id if entry.origin.platform == Platform.LOCAL
            else authority.logical_owner(entry.session_id))


def _owner(runner, event, authority):
    route = event.metadata.get('gateway_session_key') or runner.session_store._generate_session_key(event.source)
    entry = runner.session_store.lookup_by_session_key(route)
    if entry is None or entry.suspended:
        raise RuntimeStoreError('not_found')
    if runner.session_store._generate_session_key(event.source) != route:
        raise RuntimeStoreError('admission_conflict')
    expected = event.metadata.get('gateway_session_id')
    if not expected:
        raise RuntimeStoreError('admission_conflict')
    if expected != entry.session_id:
        # A completed child may follow compression, but never /new or an unrelated resume.
        if authority.db.get_compression_tip(expected) != entry.session_id:
            raise RuntimeStoreError('admission_conflict')
    return entry


def automation_notification_metadata(metadata):
    """Validate trusted producer category; omit the default from old fingerprints."""
    category = metadata.get('notification_category', 'result')
    if category not in ('result', 'diagnostic'):
        raise RuntimeStoreError('invalid_params')
    return {'notification_category': category} if category == 'diagnostic' else {}


_DISPLAY_TEXT_LIMIT = 2000


def automation_display_metadata(metadata):
    """Validate a producer's card (an async result, or a liveness tick's one-line receipt): both keys
    or neither, kind from the closed producer set."""
    if 'display_kind' not in metadata and 'display_text' not in metadata:
        return {}
    from gateway.response_filters import PRODUCER_CARD_KINDS
    kind, text = metadata.get('display_kind'), metadata.get('display_text')
    if (kind not in PRODUCER_CARD_KINDS or not isinstance(text, str) or not text.strip()
            or len(text) > _DISPLAY_TEXT_LIMIT):
        raise RuntimeStoreError('invalid_params')
    return {'display_kind': kind, 'display_text': text}

# notification_origin / original_trigger_message_id are producer debug context (#52694) and are not
# persisted; notification_category="diagnostic" rides the snapshot so the replayed wake still mutes,
# and display_kind/display_text so the persisted row is the card the producer named; heartbeat_tick is
# a liveness tick's own output, persisted so a later tick of its process folds from the ledger.
# gateway_proven_session_id / gateway_completion_ids are main's delivery-time ownership proof for the
# classic resolver (802fd5782f3, fd4e277c849). They are accepted and not persisted (a sibling set in the
# fingerprint would make the same completion's retry a new identity); execute_admission re-stamps the
# proof at execution from the target the pre-claim check validated, and the turn's resolver reads it.
# notification_event_type is producer coalescing context (gateway-only, like notification_origin): a
# completion wake carries it so the busy queue can join same-turn completions. It is accepted and not
# persisted; rejecting it here turned a refused automation wake's code from admission_conflict into
# invalid_params and dropped the wake entirely.
_AUTOMATION_METADATA = frozenset({'gateway_session_key', 'gateway_session_id', 'automation_identities',
    'turn_author', 'notification_origin', 'original_trigger_message_id', 'notification_category',
    'display_kind', 'display_text', 'gateway_proven_session_id', 'gateway_completion_ids', 'heartbeat_tick',
    'notification_event_type'})


def admission_display(payload):
    """An automation admission's async-result card (plus its model text) for the claim's
    ``message.start``: a viewer that did not submit the turn (the classic CLI, idle while a
    background process it started finishes in the gateway) paints the same header the submitting
    surface would. ``{}`` for every other input."""
    descriptor = payload.get('local_automation_v1') or (payload.get('native_text_v1') or {}).get('automation') or {}
    kind, text = descriptor.get('display_kind'), descriptor.get('display_text')
    if not kind or not isinstance(text, str) or not text.strip():
        return {}
    card = {'display_kind': kind, 'display_text': text, 'text': str(payload.get('text') or '')}
    if descriptor.get('notification_category') == 'diagnostic':
        card['notification_category'] = 'diagnostic'
    return card


def _route_identity(source):
    return {key: value for key, value in source.to_dict().items() if key != 'message_id'}


def committed_route_envelope(authority, entry):
    """The route's last committed native envelope with connector provenance (what a producer turn
    borrows), or ``None`` while no human message was admitted there."""
    from hermes_state_runtime import list_session_admissions
    # Admissions belong to the logical root; ``entry.session_id`` is only the transcript tip.
    prior = list_session_admissions(authority.db, session_id=authority.logical_owner(entry.session_id),
                                    pending_only=False)
    envelope = next((r['payload']['native_text_v1'] for r in reversed(prior)
                     if 'native_text_v1' in r['payload']), None)
    return envelope if envelope is not None and 'provenance' in envelope else None


def snapshot_automation(authority, adapter, event, identity):
    runner = authority.runner
    if (not event.internal or event.message_type != MessageType.TEXT or event.is_command()
            or not isinstance(event.text, str) or not identity
            or event.media_urls or event.prompt_response or event.source.platform == Platform.API_SERVER
            or set(event.metadata) - _AUTOMATION_METADATA):
        raise RuntimeStoreError('invalid_params')
    notification = (automation_notification_metadata(event.metadata) | automation_display_metadata(event.metadata)
                    | _tick_metadata(event.metadata))
    entry = _owner(runner, event, authority)
    if event.source.platform == Platform.LOCAL:
        return snapshot_local_automation(authority, adapter, event, identity, entry)
    from gateway.session_envelope import restore_native
    owner = authority.logical_owner(entry.session_id)
    envelope = committed_route_envelope(authority, entry)
    if envelope is None:
        raise RuntimeStoreError('not_found')
    # Persisted origins deliberately omit relay trust. Borrow only an exact
    # committed source's private proof, revalidated against the live connector.
    # Recheck only the source/provenance: a completed input's retained media is already released.
    restored = restore_native({'text': event.text, 'native_text_v1': {
        key: value for key, value in envelope.items() if key != 'media'}}, runner)
    # Synthetic wakes clear ``message_id`` (#52694, #123097) while the committed human source keeps
    # it, so compare the routing/provenance identity, never the platform reply anchor.
    if (_route_identity(restored.source) != _route_identity(event.source)
            or runner._adapter_for_source(restored.source) is not adapter):
        raise RuntimeStoreError('admission_conflict')
    provenance = deepcopy(envelope['provenance'])
    source = deepcopy(envelope['source'])
    # Never anchor the wake to the stale human message; ``to_dict`` already omits a null anchor.
    source.pop('message_id', None)
    # Producer timestamps and platform reply IDs change on retry. Neither belongs
    # in the identity/fingerprint of the same immutable completion.
    envelope = {'source': source, 'route': entry.session_key,
        'timestamp': datetime.fromtimestamp(0, UTC).isoformat(),
        'event': {'message_id': identity}, 'provenance': provenance,
        'automation': {'identity': identity, 'owner': owner}}
    envelope['automation'].update(notification)
    if getattr(event, '_heartbeat_session_id', None):
        envelope['automation']['heartbeat'] = event._heartbeat_session_id
    identities = event.metadata.get('automation_identities')
    if identities:
        envelope['automation']['identities'] = sorted(set(identities))
    return {'text': event.text, 'native_text_v1': envelope}, entry


def check_automation_route(runner, payload, session_id, available_source, adapter):
    from gateway.session_envelope import restore_native
    event = restore_native(payload, runner)
    envelope = payload['native_text_v1']
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    if authority is None:  # a late preflight for a profile this runtime no longer serves
        raise RuntimeStoreError('not_found')
    entry = _owner(runner, event, authority)
    if (entry.session_id != session_id
            or envelope['automation']['owner'] != authority.logical_owner(session_id)
            or runner.session_store._generate_session_key(available_source) != entry.session_key
            or adapter is None or runner._adapter_for_source(event.source) is not adapter):
        raise RuntimeStoreError('admission_conflict')
    return event.source, entry.session_key


def snapshot_local_automation(authority, adapter, event, identity, entry):
    from gateway.session_local_recovery import restore_local_session
    ref = restore_local_session(authority, event.source.chat_id)
    live = authority.sessions[ref.session_id]
    if (live.source is not event.source or not adapter.authorize_source(event.source)
            or authority.runner._adapter_for_source(live.source) is not adapter):
        raise RuntimeStoreError('permission_denied')
    descriptor = {'identity': identity, 'owner': ref.session_id,
                  'route': entry.session_key, 'target': entry.session_id}
    descriptor.update(automation_notification_metadata(event.metadata))
    descriptor.update(automation_display_metadata(event.metadata))
    descriptor.update(_tick_metadata(event.metadata))
    if event.metadata.get('turn_author') is not None:
        from agent.turn_author import parse_turn_author
        descriptor['turn_author'] = parse_turn_author(event.metadata['turn_author'])
        if descriptor['turn_author'] is None:
            raise RuntimeStoreError('invalid_params')
    if event.metadata.get('automation_identities'):
        descriptor['identities'] = sorted(set(event.metadata['automation_identities']))
    if getattr(event, '_heartbeat_session_id', None):
        descriptor['heartbeat'] = event._heartbeat_session_id
    return {'text': event.text, 'local_automation_v1': descriptor}, entry


def check_local_automation(authority, ref, row, *, route=None):
    live = authority.sessions.get(ref.session_id)
    route = route or (live.route if live is not None else None)
    payload = row['payload']
    descriptor = payload['local_automation_v1']
    if (not route or set(payload) != {'text', 'local_automation_v1'}
            or row['principal_id'] != 'automation:' + route
            or descriptor['owner'] != ref.session_id or descriptor['route'] != route
            or descriptor['identity'] != row['request_id']):
        raise RuntimeStoreError('permission_denied')
    entry = authority.runner.session_store.lookup_by_session_key(route)
    if (entry is None or entry.suspended or (entry.session_id != descriptor['target']
            and authority.db.get_compression_tip(descriptor['target']) != entry.session_id)):
        raise RuntimeStoreError('admission_conflict')


def restore_local_automation(authority, ref, row):
    check_local_automation(authority, ref, row)
    live = authority.sessions[ref.session_id]
    descriptor = row['payload']['local_automation_v1']
    entry = authority.runner.session_store.lookup_by_session_key(live.route)
    event = MessageEvent(text=row['payload']['text'], source=live.source, internal=True,
        message_id=descriptor['identity'], metadata={'gateway_session_key': live.route,
            'gateway_session_id': entry.session_id})
    event.metadata.update(automation_notification_metadata(descriptor))
    event.metadata.update(automation_display_metadata(descriptor))
    if descriptor.get('turn_author') is not None:
        event.metadata['turn_author'] = deepcopy(descriptor['turn_author'])
    if descriptor.get('heartbeat'):
        event._heartbeat_session_id = descriptor['heartbeat']
    return event


async def admit_automation(authority, adapter, event, identity, replaces=None):
    """``replaces``: a queued automation row of the same session this input supersedes (a liveness
    tick carrying the output of the one still waiting). Its cancellation commits in this admission's
    own transaction, so neither lands without the other."""
    authority._require_admission_open()
    payload, entry = snapshot_automation(authority, adapter, event, identity)
    from gateway.session_authority import LiveSession
    sid = (payload.get('local_automation_v1') or payload['native_text_v1']['automation'])['owner']
    ref = SessionRef(authority.profile_id, sid)
    authority.sessions.setdefault(ref.session_id, LiveSession(event.source, entry.session_key))
    from functools import partial
    from gateway.session_runtime_workers import tracked_write
    admitted, replaced = partial(authority._admitted, ref, event), []
    if replaces is not None:
        def admitted(row, admitted=admitted):
            if replaced:  # the guard ran in the committed write; an exact replay runs no guard
                authority._settle_cancelled(ref, replaces)
            return admitted(row)
    return await tracked_write(authority, partial(
        admit_session_input, authority.db, epoch=authority.epoch, principal_id='automation:' + entry.session_key,
        session_id=ref.session_id, request_id=identity, payload=deepcopy(payload),
        _authorize_write=authority._admission_gate(replaces and _replace_queued(replaces, ref.session_id, replaced))),
        then=admitted, ordered=True, after=partial(authority._schedule_admitted, ref))


_ROUTE_WAKE_PRODUCERS = frozenset({'kanban-wake', 'loop-tick'})


async def move_route_wakes(runner, session_key, old_session_id):
    """After the route rotated (/new, /reset, compression-exhaustion reset), carry the old owner's
    still-queued route wakes to the fresh session, as main's #114456 kept a parked internal wake
    across /new. Left behind, the old drain refuses them (``admission_conflict``) and they stay
    queued forever. Completions pinned to the closed conversation are not moved (they fail closed,
    #55578), nor anything behind an ``unknown`` head (the operator's to resume)."""
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    entry = runner.session_store.lookup_by_session_key(session_key) if authority is not None else None
    if entry is None or not old_session_id:
        return
    import asyncio
    from functools import partial
    from gateway.session_authority import LiveSession
    from gateway.session_contract import Principal
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_runtime import list_session_admissions
    old = SessionRef(authority.profile_id, authority.logical_owner(old_session_id))
    new = SessionRef(authority.profile_id, authority.logical_owner(entry.session_id))
    if old.session_id not in authority.sessions or old == new:
        return
    rows = await asyncio.to_thread(list_session_admissions, authority.db, session_id=old.session_id)
    if any(row['status'] == 'unknown' for row in rows):
        return
    actor = Principal('messaging-control', authority.profile_id, frozenset({'session:submit', 'session:control'}), 'reset')
    for row in rows:
        automation = row['payload'].get('native_text_v1', {}).get('automation', {})
        identity = json.loads(row['request_id']) if automation and row['request_id'].startswith('[') else None
        if (row['status'] != 'queued' or row['principal_id'] != 'automation:' + session_key
                or not isinstance(identity, list) or identity[:1] not in ([p] for p in _ROUTE_WAKE_PRODUCERS)):
            continue
        try:
            await authority.cancel_queued(actor, old, row['admission_id'])
        except RuntimeStoreError:
            continue  # claimed meanwhile: it runs where it is
        payload = deepcopy(row['payload'])
        payload['native_text_v1']['automation']['owner'] = new.session_id
        authority.sessions.setdefault(new.session_id, LiveSession(authority.sessions[old.session_id].source, session_key))
        await tracked_write(authority, partial(
            admit_session_input, authority.db, epoch=authority.epoch, principal_id=row['principal_id'],
            session_id=new.session_id, request_id=row['request_id'], payload=payload,
            _authorize_write=authority._admission_gate()), then=partial(authority._admitted, new, None),
            ordered=True, after=partial(authority._schedule_admitted, new))


async def reset_and_rehome_messages(runner, session_key, old_session_id):
    """The compression-exhaustion auto-reset: rotate the route to a fresh session and move the old
    owner's still-queued native ``messaging:*`` rows onto it, same order and request ids. Left
    behind they pause the old FIFO (``admission_conflict``) and never run, while a later message
    runs ahead of them. /new and /stop cancel these rows instead (the user asked for a clean break);
    the exhausted turn itself is started, so it is never moved.

    The rotation (sessions.json) and the move (one SQLite transaction) run in one worker step
    ordered with native admissions, not atomically across the two stores: a redelivery that
    resolves the route to the fresh session commits after the move and dedupes onto the moved row,
    and one that reserved the old route before the rotation is retargeted at commit
    (``_admit_native_write``). Returns ``(entry, release)``. Until the caller runs ``release()``
    (once, after its conversation cleanup and topic binding sync, also on failure) the fresh
    session's ``claim_gate`` is held, so no scheduler (a redelivery, a moved route wake, this
    move) starts a turn there against stale state; ``release()`` then schedules its FIFO."""
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    old = SessionRef(authority.profile_id, authority.logical_owner(old_session_id)) if authority else None
    live = authority.sessions.get(old.session_id) if old and old_session_id else None
    if live is None:
        return await runner.async_session_store.reset_session(session_key), lambda: None
    import asyncio
    from gateway.session_authority import LiveSession
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_runtime import rehome_queued_inputs

    def rotate_and_move():
        entry = runner.session_store.reset_session(session_key)
        if entry is None:
            return None, None, []
        new = SessionRef(authority.profile_id, authority.logical_owner(entry.session_id))
        if new == old:
            return entry, None, []
        if authority.db.get_session(new.session_id) is None:
            authority.db.create_session(new.session_id, source=live.source.platform.value)
        # An admission that reserved the old route before this rotation commits after it.
        authority.rotated_owners[old.session_id] = session_key
        return entry, new, rehome_queued_inputs(
            authority.db, epoch=authority.epoch, old_session_id=old.session_id,
            new_session_id=new.session_id, principal_prefix='messaging:')

    async def fence(result):
        # In the write's own task step, before any later ordered admission schedules the fresh FIFO.
        new = result[1]
        if new is None:
            return result, None
        fresh = authority.sessions.setdefault(new.session_id, LiveSession(live.source, session_key))
        await fresh.claim_gate.acquire()
        return result, fresh

    async with live.mutation_lock:
        write = asyncio.ensure_future(tracked_write(authority, rotate_and_move, fence, ordered=True))
        try:
            (entry, new, moved), fresh = await asyncio.shield(write)
        except asyncio.CancelledError:
            # The rotation still commits and takes the fence: lift it, or the fresh FIFO never runs.
            def lift(done):
                gate = None if done.cancelled() or done.exception() else done.result()[1]
                if gate is not None:
                    gate.claim_gate.release()
                    authority._schedule(done.result()[0][1])
            write.add_done_callback(lift)
            raise
    if fresh is None:
        return entry, lambda: None
    if moved:
        authority._publish_pending(old)
        authority._publish_pending(new)

    released = []

    def release():  # once: a second call must not lift a claim the drain took meanwhile
        if not released:
            released.append(True)
            fresh.claim_gate.release()
            if authority.sessions.get(new.session_id) is fresh:
                authority._schedule(new)
    return entry, release


async def admit_route_wake(runner, adapter, source, text, producer, metadata=None):
    """Commit a producer wake (Kanban push wake, /loop tick) for ``source``'s route into the
    authority FIFO, like heartbeats and process completions, instead of a classic turn that
    interrupts or races the admitted one. ``False`` when no authority serves the scope or the
    route holds no committed human turn to borrow provenance from: the caller keeps the adapter
    path (nothing is queued there either). Raises ``RuntimeStoreError`` on a refusal."""
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    key = runner._session_key_for_source(source)
    entry = runner.session_store.lookup_by_session_key(key) if authority is not None else None
    if authority is None or entry is None:
        return False
    import uuid
    identity = json.dumps([producer, key, uuid.uuid4().hex], separators=(',', ':'))
    if source.platform == Platform.LOCAL:
        # A local route has no messaging provenance to borrow: its wake rides local_automation_v1 on
        # the live session's own source object (``snapshot_local_automation`` checks identity).
        live = authority.sessions.get(source.chat_id)
        if live is None or live.route != key:
            return False
        event = MessageEvent(text=text, message_type=MessageType.TEXT, source=live.source, internal=True,
                             message_id=identity, metadata={'gateway_session_key': key,
                                                            'gateway_session_id': entry.session_id,
                                                            **(metadata or {})})
        await authority.admit_automation(adapter, event, identity)
        return True
    from gateway.session_envelope import restore_native
    envelope = committed_route_envelope(authority, entry)
    if envelope is None:
        return False
    # The route's committed source, which ``snapshot_automation`` re-derives and compares.
    restored = restore_native({'text': text, 'native_text_v1': {
        k: v for k, v in envelope.items() if k != 'media'}}, runner)
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=restored.source, internal=True,
                         message_id=identity, metadata={'gateway_session_key': key,
                                                        'gateway_session_id': entry.session_id,
                                                        **(metadata or {})})
    await authority.admit_automation(adapter, event, identity)
    return True
