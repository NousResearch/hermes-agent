"""Bot Chat ingress into the existing authority FIFO; viewers never own delivery.

The mailbox is a delivery receipt, not an execution queue. A committed canonical
admission is the only consumer; unknown execution is never retried as inference.
"""
import asyncio
import logging
from pathlib import Path

from agent.turn_author import parse_turn_author

from gateway.session_contract import Principal, SessionRef
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tools.bot_live_delivery import _delivery_id, _locked, _read, _root, _write


_CANONICAL_RECEIPT_STATUSES = frozenset({
    'canonical', 'queued', 'claimed', 'ambiguous', 'settled', 'failed', 'cancelled',
})


def _canonical_record_error(path, record):
    """Return why an admission-backed receipt is unsafe for bulk recovery."""
    delivery_id = record.get('delivery_id')
    if not isinstance(delivery_id, str) or delivery_id != path.stem:
        return 'delivery id does not match filename'
    try:
        _delivery_id(delivery_id)
    except ValueError:
        return 'delivery id is invalid'
    required_strings = ('admission_id', 'profile_home', 'session_id', 'principal_id')
    missing = [key for key in required_strings
               if not isinstance(record.get(key), str) or not record[key]]
    if missing:
        return 'missing canonical fields: ' + ', '.join(missing)
    if not isinstance(record.get('message'), str):
        return 'message is not a string'
    status = record.get('status')
    if not isinstance(status, str) or status not in _CANONICAL_RECEIPT_STATUSES:
        return f'unknown canonical status {status!r}'
    return None


def _record_shape_error(path, record):
    """Validate the known canonical or legacy receipt shape for directory scans."""
    if 'admission_id' in record or record.get('status') == 'canonical' or 'principal_id' in record:
        return _canonical_record_error(path, record)
    if 'owner' in record:
        from tools.bot_live_delivery import _ticket_shape_error
        return _ticket_shape_error(path, record)
    return 'unknown receipt schema'


def _scan_records(root):
    """Bulk mailbox scan: isolate unreadable/malformed records; exact-id reads still fail closed."""
    for path in root.glob('*.json'):
        try:
            record = _read(path)
            problem = None if record is None else _record_shape_error(path, record)
        except (OSError, ValueError) as exc:
            record, problem = None, str(exc)
        if problem is not None:
            logging.getLogger(__name__).warning(
                "Skipping malformed Bot mailbox receipt %s (%s)", path, problem)
            continue
        if record is not None:
            yield path, record


def _home(authority, actor, profile):
    from gateway.session_authorities import served_profile_name
    home = Path(authority.db.db_path).parent.resolve()
    name = served_profile_name(home)
    if actor.profile_id != authority.profile_id or profile not in (name, 'hermes' if name == 'default' else name):
        raise RuntimeStoreError('profile_mismatch')
    if 'session:submit' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    return home


def _target(authority, actor):
    row = authority.db.get_session_by_title('Bot Chat')
    if row is None:
        ref = _create_bot_chat(authority, actor)
    else:
        tip = authority.db.get_compression_tip(row['id'])
        target = authority.db.get_session(tip)
        if target is None:
            raise RuntimeStoreError('not_found')
        from gateway.session_local_migration import resolve_local_target
        ref = resolve_local_target(authority, actor, target['id'])
    authority.authorize(actor, ref, 'session:submit')
    live = authority.sessions[ref.session_id]
    entry = authority.runner.session_store.lookup_by_session_key(live.route)
    if live.source.platform != Platform.LOCAL or entry is None:
        raise RuntimeStoreError('admission_conflict')
    return ref, live, entry


def _create_bot_chat(authority, actor):
    """No Bot Chat yet: mint it the way ``hermes chat --in ~ -c "Bot Chat" --create-if-missing``
    did on the CLI lane this door replaced, so a fresh profile's first delivery has somewhere
    to land instead of being recorded as failed. Same owner step as ``session.create`` with a
    title: no await between the lookup and the titled creation. Creation is the actor's own
    capability; a submit-only principal (owner recovery) still gets ``not_found``."""
    if 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('not_found')
    from gateway.session_local import create_local_session
    from gateway.session_local_title import title_new_session
    ref = create_local_session(authority, actor, {
        'request_id': 'bot-chat:' + authority.profile_id, 'source': 'cli', 'cwd': str(Path.home())})
    title_new_session(authority, ref, 'Bot Chat')
    return ref


def _admission_outcome(authority, admission_id, fallback=None):
    """``(status, reply)`` of one admission, read from its committed result, never transcript recency."""
    row = get_session_admission(authority.db, admission_id=admission_id)
    if row is None:
        raise RuntimeStoreError('storage_unavailable')
    # A terminal row without a result blob is still definitive when the outcome says the
    # input never ran (cancelled) or was refused/failed; ambiguous is reserved for the
    # durable unknown state and for missing evidence (completed/interrupted without result).
    status = {'queued': 'queued', 'started': 'claimed', 'unknown': 'ambiguous',
              'terminal': {'cancelled': 'cancelled', 'rejected': 'failed', 'failed': 'failed'}.get(row['outcome'], 'ambiguous')}[row['status']]
    from gateway.session_results import admission_result
    saved = admission_result(authority.db, admission_id)
    reply = (fallback or {}).get('reply', '')
    if saved is not None:
        reply = saved['result'].get('final_response', '')
        status = 'settled' if row['outcome'] == 'completed' else 'failed'
        # A successful bare silence marker is a delivery decision (same rule as the gateway and
        # the live Bot Chat completion): the turn stays in the target's transcript, the sender
        # never sees the marker as prose.
        from gateway.response_filters import is_intentional_silence_response
        if status == 'settled' and is_intentional_silence_response(reply):
            reply = ''
    elif fallback is not None and fallback.get('status') in {'settled', 'failed'}:
        status = fallback['status']
    return status, reply


def _retry_admission(authority, record):
    """The delivery's one retry admission id, read from the FIFO by its derived identity, so every
    reader (an in-memory record, the receipt file, an owner restart) follows the same retry."""
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT admission_id FROM session_admissions WHERE target_session_id=? AND request_id=?',
                           (record['session_id'], _retry_identity(record['delivery_id']))).fetchone()
    return row[0] if row else None


def _result(authority, record):
    """The delivery's receipt. Once the one transient-failure retry was admitted, the sender's
    outcome is the retry's: ``status``/``reply`` follow it (``retry_admission_id``).

    A ``failed`` receipt carries ``error`` + a typed ``reason`` (``tools.bot_failure_reasons``):
    the relay's Desktop forwards ``res.reason`` to the sender, which cannot re-derive it from a
    reply that lacks the provider text."""
    retry_id = _retry_admission(authority, record)
    current = retry_id or record['admission_id']
    if retry_id:
        status, reply = _admission_outcome(authority, retry_id)
    else:
        status, reply = _admission_outcome(authority, record['admission_id'], record)
        if status == 'failed' and _retry_due(authority, record):
            # Interim, never the answer: the one retry is admitted next (or by the restarted
            # owner's recovery while this one drains). Senders poll the published receipt file.
            status = 'claimed'
    error = reason = None
    if status == 'failed':
        error, reason = _failure(authority, current)
    return {k: v for k, v in dict(status=status, delivery_id=record['delivery_id'],
        profile_home=record['profile_home'], session_id=record['session_id'],
        admission_id=record['admission_id'], message=record['message'], reply=reply,
        retry_admission_id=retry_id, error=error, reason=reason).items() if v is not None}


def _failure(authority, admission_id):
    """``(error, reason)`` of a failed admission from its committed result: the raw provider error
    and the turn loop's typed verdict, classified like every other Bot lane. A refusal that never
    ran (no result) has no provider text: ``unknown``."""
    from gateway.session_results import admission_result
    from tools.bot_failure_reasons import classify_agent_error, turn_failure_text
    saved = (admission_result(authority.db, admission_id) or {}).get('result') or {}
    error = saved.get('error') or saved.get('final_response') or 'Bot Chat delivery failed'
    return error, classify_agent_error(turn_failure_text(saved.get('error'), saved.get('failure_reason')) or error)


def peer_wait_admission(authority, record):
    """``(admission_id, interim)`` a live waiter must follow; ``(None, False)`` once the delivery
    settled for good. The retry, once admitted, is the only admission whose future fires again.
    An original that failed while its one retry is still eligible is ``interim``: pending, never
    the answer, while the owner's receipt task admits the retry (or records why it would not;
    the receipt file is re-read for that, the caller's in-memory record is stale)."""
    retry_id = _retry_admission(authority, record)
    if retry_id:
        return (retry_id, False) if _admission_outcome(authority, retry_id)[0] in {'queued', 'claimed'} else (None, False)
    status = _admission_outcome(authority, record['admission_id'], record)[0]
    if status in {'queued', 'claimed'}:
        return record['admission_id'], False
    if status == 'failed' and _retry_due(authority, record):
        return record['admission_id'], True
    return None, False


def _retry_due(authority, record):
    """The failed original's one retry is still to be admitted. Its refusal is recorded on the
    receipt file; an in-memory record without the ``retry`` marker re-reads it there."""
    retry = record.get('retry')
    if retry is None:
        retry = (_read(_root(record['profile_home']) / f"{record['delivery_id']}.json") or {}).get('retry')
    return _retry_eligible(authority, {**record, 'retry': retry})


_UNCHECKED = object()


def _store(home, record, *, expect=_UNCHECKED):
    """Pin a receipt another process may still move (a new or legacy ticket) under the
    cross-process mailbox lock, held for this one write: never across an admission, a FIFO wait or
    a recovery scan, and taken off-loop (``asyncio.to_thread``). ``expect`` (the status read before,
    None = absent) makes it a compare-and-set that refuses when another writer moved the ticket
    first, e.g. a DM runner's ``cancel_queued_delivery``."""
    with _locked(home) as root:
        path = root / f"{record['delivery_id']}.json"
        if expect is not _UNCHECKED and (_read(path) or {}).get('status') != expect:
            return False
        _write(path, record)
        return True


def _publish(home, record):
    """Republish a pinned receipt in place. Once pinned (``ambiguous`` → canonical) only this owner
    writes it, and every foreign writer refuses it, so an atomic rename needs no cross-process lock;
    publishing in the same step the admission settles keeps the file ahead of any reply."""
    _write(_root(home) / f"{record['delivery_id']}.json", record)


def _mailbox_order(authority):
    """Orders this owner's read-admit-publish sequences per receipt in-process. An asyncio lock:
    a contender yields to the event loop instead of blocking it on the file lock."""
    order = getattr(authority, '_bot_mailbox_order', None)
    if order is None:
        order = authority._bot_mailbox_order = asyncio.Lock()
    return order


def _retry_identity(key):
    return 'bot:' + key + ':retry'


def _retry_eligible(authority, record):
    """Exactly the main-lane gate (``tools.bot_failure_reasons.result_retry_action``): the original
    admission SETTLED ``failed`` with a committed result whose error classifies transient (429/5xx/
    context overflow). queued/started/unknown admissions never qualify — unknown execution is never
    replayed — and the retry itself is never retried."""
    if (record.get('retry') or {}).get('refused') or _retry_admission(authority, record):
        return False
    row = get_session_admission(authority.db, admission_id=record['admission_id'])
    if row is None or row['status'] != 'terminal' or row['outcome'] != 'failed':
        return False
    from gateway.session_results import admission_result
    saved = admission_result(authority.db, record['admission_id'])
    if saved is None:
        return False
    from tools.bot_failure_reasons import RETRY_NONE, result_retry_action
    return result_retry_action(saved['result']) != RETRY_NONE


async def _maybe_retry(authority, home, record):
    """Admit the delivery's one retry under a DERIVED identity recorded on the same receipt, so a
    repeated deliver(), an owner restart or a second settle re-reads that admission instead of
    minting another. The identity is written before the admission: a death in that two-store window
    re-admits the SAME identity, which the FIFO answers with the existing row (never a second turn).

    No user-row adoption: a transient owner-side failure persists the DM row and then CLOSES it with the
    durable failed-turn boundary (``agent.conversation_loop._close_durable_failed_turn`` /
    ``_hmwa_close_failed_turn``, #107070), so ``agent.session_persistence.adopt_unanswered_turn`` would
    decline by contract (a plain assistant row follows the DM) and the retry is a fresh turn after that
    boundary. A failure that left the DM as an OPEN tail (the context-pressure classes skip the closer)
    is not retried: the owner execution has no adoption seam, and the retry's DM would merge into it."""
    if not _retry_eligible(authority, record):
        return
    identity = _retry_identity(record['delivery_id'])
    actor = Principal(record['principal_id'], authority.profile_id, frozenset({'session:submit', 'session:read'}),
                      'bot-delivery-retry')
    try:
        authority._require_admission_open()
        ref, live, entry = _target(authority, actor)
        if ref.session_id != record['session_id']:
            raise RuntimeStoreError('admission_conflict')
        if authority.db.latest_conversation_role(entry.session_id) == 'user':
            raise RuntimeStoreError('open_user_tail')
    except RuntimeStoreError as exc:
        if exc.reason == 'runtime_draining':
            return  # nothing recorded: the restarted owner's recovery re-evaluates the same gate
        record['retry'] = {'identity': identity, 'refused': exc.reason}
        record.update(_result(authority, record))  # the refusal makes the original's failure final
        _publish(home, record)
        return
    event = MessageEvent(text=record['message'], source=live.source, internal=True,
        message_id=identity, metadata={'gateway_session_key': live.route, 'gateway_session_id': entry.session_id})
    from gateway.session_automation import automation_notification_metadata
    event.metadata.update(automation_notification_metadata(record))
    if record.get('author') is not None:
        event.metadata['turn_author'] = dict(record['author'])
    record['retry'] = {'identity': identity}
    _publish(home, record)
    receipt = await authority.admit_automation(authority.runner._adapter_for_source(live.source), event, identity)
    record.update(_result(authority, record))
    if receipt.status in {'queued', 'started'}:
        _watch_reply(authority, home, record['delivery_id'], receipt.admission_id)
    _publish(home, record)


async def _record_reply(authority, home, key, future):
    try:
        await asyncio.shield(future)
    except RuntimeStoreError:
        # A paused FIFO is not a terminal Bot outcome; project the durable state below.
        pass
    async with _mailbox_order(authority):
        record = _read(_root(home) / f'{key}.json')
        record.update(_result(authority, record))
        _publish(home, record)
        admission_id, interim = peer_wait_admission(authority, record)
        if admission_id is not None and not interim:
            _watch_reply(authority, home, key, admission_id)
            return
        await _maybe_retry(authority, home, record)


def relay_operation(connection, operation, params):
    authority, actor = connection.authority, connection.actor
    from gateway.session_authorities import served_profile_name
    home = Path(authority.db.db_path).parent.resolve()
    name = served_profile_name(home)
    _home(authority, actor, name)
    if 'session:control' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    fields = {'roster': {'agents'}, 'outbox': set(), 'reply': {'id', 'reply', 'error', 'reason'}}
    if set(params) - fields[operation]:
        raise RuntimeStoreError('invalid_params')
    from tools.bot_relay import write_remote_roster, claim_pending_envelopes, write_reply
    if operation == 'roster':
        return {'count': write_remote_roster(home, params.get('agents'))}
    if operation == 'outbox':
        return {'envelopes': claim_pending_envelopes(home)}
    try:
        write_reply(home, params.get('id'), reply=params.get('reply', ''),
                    error=params.get('error', ''), reason=params.get('reason', ''))
    except ValueError as exc:
        raise RuntimeStoreError('admission_conflict') from exc
    return {'ok': True}


async def _migrate(authority, actor, home, root):
    records = list(_scan_records(root))
    legacy = [(path, record) for path, record in records
              if record and 'owner' in record and not record.get('admission_id')
              and record['status'] in {'queued', 'claimed'}]
    if not legacy:
        return
    ref, live, entry = _target(authority, actor)
    for path, record in sorted(legacy, key=lambda item: (item[1].get('sequence', item[1]['created_at']), item[0].name)):
        owner = record['owner']
        if owner['profile_home'] != str(home):
            continue
        if authority.db.get_compression_tip(owner['session_id']) != entry.session_id:
            continue
        from hermes_cli.active_sessions import active_session_liveness_guard
        with active_session_liveness_guard(owner['session_id'], registry_home=home) as active:
            if active:
                raise RuntimeStoreError('runtime_coordination_required')
        if record['status'] == 'claimed':
            await asyncio.to_thread(_store, home, dict(record, status='ambiguous', reason='unknown_execution'),
                                    expect='claimed')
            continue
        await _admit(authority, actor, home, root, _delivery_id(record['delivery_id']),
                     record['message'], ref, live, entry, author=parse_turn_author(record.get('author')), legacy=record,
                     notification_category=record.get('notification_category', 'result'))


async def recover_bot_deliveries(authority):
    """Rebuild derivative replies and queued legacy admissions at owner startup."""
    home = Path(authority.db.db_path).parent.resolve()
    root = _root(home)
    async with _mailbox_order(authority):
        records = list(_scan_records(root))
        for path, record in records:
            if not record or record.get('profile_home') != str(home) or not record.get('admission_id'):
                continue
            record.update(_result(authority, record))
            live, interim = peer_wait_admission(authority, record)
            if live is not None and not interim:
                _watch_reply(authority, home, record['delivery_id'], live)
            _publish(home, record)
            if live is None or interim:
                await _maybe_retry(authority, home, record)
        row = authority.db.get_session_by_title('Bot Chat')
        if row is None:
            return
        target = authority.db.get_session(authority.db.get_compression_tip(row['id']))
        if target is None:
            return
        from hermes_state_local import POLICY_PREFIX
        with authority.db._read_ctx() as conn:
            bound = conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                (POLICY_PREFIX + (target.get('chat_id') or target['id']),)).fetchone()
        if not bound:
            return  # Unowned history requires a native ticket, never the first remote sender.
        actor = Principal(target['user_id'], authority.profile_id,
                          frozenset({'session:submit', 'session:read'}), 'bot-owner-recovery')
        await _migrate(authority, actor, home, root)


def _watch_reply(authority, home, key, admission_id):
    future = authority.waiters.setdefault(admission_id, asyncio.get_running_loop().create_future())
    task = asyncio.create_task(_record_reply(authority, home, key, future))
    tasks = getattr(authority, '_bot_receipt_tasks', None)
    if tasks is None:
        tasks = authority._bot_receipt_tasks = set()
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def deliver(connection, params):
    authority, actor = connection.authority, connection.actor
    home = _home(authority, actor, params.get('profile'))
    if set(params) - {'id', 'profile', 'message', 'session_id', 'author', 'notification_category'}:
        raise RuntimeStoreError('invalid_params')
    from gateway.session_automation import automation_notification_metadata
    notification = automation_notification_metadata(params)
    category = notification.get('notification_category', 'result')
    try:
        key = _delivery_id(params.get('id'))
    except ValueError as exc:
        raise RuntimeStoreError('invalid_params') from exc
    message = params.get('message')
    if not isinstance(message, str) or not message.strip() or len(message) > 16200:
        raise RuntimeStoreError('invalid_params')
    author = parse_turn_author(params.get('author'))
    if params.get('author') is not None and (not isinstance(params['author'], dict) or author is None):
        raise RuntimeStoreError('invalid_params')
    authority._require_admission_open()
    # In-process order only: the cross-process file lock is taken for the pin write alone, off-loop
    # (``_store``), so a foreign holder or a slow admission never stalls this loop or another delivery.
    async with _mailbox_order(authority):
        path = _root(home) / f'{key}.json'
        record = _read(path)
        if record is None or not record.get('admission_id'):
            await _migrate(authority, actor, home, path.parent)
            record = _read(path)
        if record is not None and record.get('admission_id'):
            if (record['message'] != message or record['principal_id'] != actor.subject
                    or record.get('author') != author
                    or record.get('notification_category', 'result') != category):
                raise RuntimeStoreError('admission_conflict')
            authority.authorize(actor, SessionRef(authority.profile_id, record['session_id']), 'session:submit')
            await _maybe_retry(authority, home, record)
            return _result(authority, record)
        if record is not None:
            raise RuntimeStoreError('unknown_execution')
        ref, live, entry = _target(authority, actor)
        if params.get('session_id', entry.session_id) != entry.session_id:
            raise RuntimeStoreError('admission_conflict')
        return await _admit(authority, actor, home, path.parent, key, message, ref, live, entry, author=author,
                            notification_category=category)


async def _admit(authority, actor, home, root, key, message, ref, live, entry, author=None, legacy=None,
                 notification_category='result'):
    from gateway.session_automation import automation_notification_metadata
    notification = automation_notification_metadata({'notification_category': notification_category})
    event = MessageEvent(text=message, source=live.source, internal=True,
        message_id='bot:' + key, metadata={'gateway_session_key': live.route,
                                         'gateway_session_id': entry.session_id})
    event.metadata.update(notification)
    if author is not None:
        event.metadata['turn_author'] = dict(author)
    # Pin the physical target before committing. A process death in this
    # two-store window leaves an explicit unknown record, never a new target.
    record = dict(legacy or {}, delivery_id=key, profile_home=str(home), session_id=ref.session_id,
        principal_id=actor.subject, message=message, status='ambiguous')
    record.update(notification)
    if author is not None:
        record['author'] = dict(author)
    # Compare-and-set against the state this owner read: a DM runner may cancel a queued legacy
    # ticket between the scan and this pin, and that cancellation wins.
    if not await asyncio.to_thread(_store, home, record, expect=(legacy or {}).get('status')):
        if legacy is not None:
            return None  # that cancellation (or a late claim) stands; nothing was admitted
        raise RuntimeStoreError('admission_conflict')
    receipt = await authority.admit_automation(authority.runner._adapter_for_source(live.source), event, 'bot:' + key)
    record.update(status='canonical', admission_id=receipt.admission_id)
    record.update(_result(authority, record))  # publish the outcome senders poll, not a bare marker
    if receipt.status in {'queued', 'started'}:
        # Its task publishes the settled receipt once this delivery releases the in-process order.
        _watch_reply(authority, home, key, receipt.admission_id)
    _publish(home, record)
    return _result(authority, record)
