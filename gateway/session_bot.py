"""Bot Chat ingress into the existing authority FIFO; viewers never own delivery.

The mailbox is a delivery receipt, not an execution queue. A committed canonical
admission is the only consumer; unknown execution is never retried as inference.
"""
import asyncio
from pathlib import Path

from agent.turn_author import parse_turn_author

from gateway.session_contract import Principal, SessionRef
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tools.bot_live_delivery import _delivery_id, _locked, _read, _write


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
        raise RuntimeStoreError('not_found')
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


def _result(authority, record):
    row = get_session_admission(authority.db, admission_id=record['admission_id'])
    if row is None:
        raise RuntimeStoreError('storage_unavailable')
    status = {'queued': 'queued', 'started': 'claimed', 'unknown': 'ambiguous', 'terminal': 'ambiguous'}[row['status']]
    # Read only this admission's committed result, never transcript recency.
    from gateway.session_results import admission_result
    saved = admission_result(authority.db, record['admission_id'])
    reply = record.get('reply', '')
    if saved is not None:
        reply = saved['result'].get('final_response', '')
        status = 'settled' if row['outcome'] == 'completed' else 'failed'
    elif record.get('status') in {'settled', 'failed'}:
        status = record['status']
    return {k: v for k, v in dict(status=status, delivery_id=record['delivery_id'],
        profile_home=record['profile_home'], session_id=record['session_id'],
        admission_id=record['admission_id'], message=record['message'], reply=reply).items()}


async def _record_reply(authority, home, key, future):
    await asyncio.shield(future)
    with _locked(home) as root:
        path = root / f'{key}.json'
        record = _read(path)
        record.update(_result(authority, record))
        _write(path, record)


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
    records = [(path, _read(path)) for path in root.glob('*.json')]
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
            record.update(status='ambiguous', reason='unknown_execution')
            _write(path, record)
            continue
        await _admit(authority, actor, home, root, _delivery_id(record['delivery_id']),
                     record['message'], ref, live, entry, author=parse_turn_author(record.get('author')), legacy=record)


async def recover_bot_deliveries(authority):
    """Rebuild derivative replies and queued legacy admissions at owner startup."""
    home = Path(authority.db.db_path).parent.resolve()
    with _locked(home) as root:
        records = [(path, _read(path)) for path in root.glob('*.json')]
        for path, record in records:
            if not record or record.get('profile_home') != str(home) or not record.get('admission_id'):
                continue
            record.update(_result(authority, record))
            _write(path, record)
            if record['status'] in {'queued', 'claimed'}:
                _watch_reply(authority, home, record['delivery_id'], record['admission_id'])
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
    if set(params) - {'id', 'profile', 'message', 'session_id', 'author'}:
        raise RuntimeStoreError('invalid_params')
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
    with _locked(home) as root:
        path = root / f'{key}.json'
        record = _read(path)
        if record is not None and record.get('admission_id'):
            if (record['message'] != message or record['principal_id'] != actor.subject
                    or record.get('author') != author):
                raise RuntimeStoreError('admission_conflict')
            authority.authorize(actor, SessionRef(authority.profile_id, record['session_id']), 'session:submit')
            return _result(authority, record)
        await _migrate(authority, actor, home, root)
        record = _read(path)
        if record is not None and record.get('admission_id'):
            if (record['message'] != message or record['principal_id'] != actor.subject
                    or record.get('author') != author):
                raise RuntimeStoreError('admission_conflict')
            authority.authorize(actor, SessionRef(authority.profile_id, record['session_id']), 'session:submit')
            return _result(authority, record)
        if record is not None:
            raise RuntimeStoreError('unknown_execution')
        ref, live, entry = _target(authority, actor)
        if params.get('session_id', entry.session_id) != entry.session_id:
            raise RuntimeStoreError('admission_conflict')
        return await _admit(authority, actor, home, root, key, message, ref, live, entry, author=author)


async def _admit(authority, actor, home, root, key, message, ref, live, entry, author=None, legacy=None):
    path = root / f'{key}.json'
    event = MessageEvent(text=message, source=live.source, internal=True,
        message_id='bot:' + key, metadata={'gateway_session_key': live.route,
                                         'gateway_session_id': entry.session_id})
    if author is not None:
        event.metadata['turn_author'] = dict(author)
    # Pin the physical target before committing. A process death in this
    # two-store window leaves an explicit unknown record, never a new target.
    record = dict(legacy or {}, delivery_id=key, profile_home=str(home), session_id=ref.session_id,
        principal_id=actor.subject, message=message, status='ambiguous')
    if author is not None:
        record['author'] = dict(author)
    _write(path, record)
    receipt = await authority.admit_automation(authority.runner._adapter_for_source(live.source), event, 'bot:' + key)
    record.update(status='canonical', admission_id=receipt.admission_id)
    _write(path, record)
    if receipt.status in {'queued', 'started'}:
        _watch_reply(authority, home, key, receipt.admission_id)
    return _result(authority, record)