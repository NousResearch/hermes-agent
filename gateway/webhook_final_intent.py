"""Original webhook finals: durable notification intent, distinct from task outcome.

Only a real committed admission/result is an origin. Native receipt files remain the
recipient's authority; this ledger never owns inference or edits recipient history.
"""
import asyncio
from functools import partial
import hashlib
import json
from pathlib import Path
import time

from hermes_state_runtime import RuntimeStoreError, _epoch, get_session_admission

PREFIX = 'webhook.final.intent.v1.'
TERMINAL = frozenset({'settled', 'suppressed', 'refused', 'failed'})


def read_intent(authority, admission_id):
    row = authority.db._read_one('SELECT value FROM state_meta WHERE key=?', (PREFIX + admission_id,))
    return json.loads(row['value']) if row else None


def _source_fence(conn, value, native):
    """Check admission-backed lifetimes inside the same write as publication."""
    from hermes_state_terminal import ADMISSION_PREFIX
    ids = []
    if value.get('ledger_prefix', PREFIX) == PREFIX:
        ids.append(value['origin_id'])
    if value.get('source_admission_id'):
        ids.append(value['source_admission_id'])
    if native.get('webhook_root'):
        ids.append(native['webhook_root'])
    for admission_id in dict.fromkeys(ids):
        live = conn.execute('SELECT a.target_session_id FROM session_admissions a '
                            'JOIN sessions s ON s.id=a.target_session_id WHERE a.admission_id=?',
                            (admission_id,)).fetchone()
        if live is None:
            tombstone = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                     (ADMISSION_PREFIX + admission_id,)).fetchone()
            retired = json.loads(tombstone[0]).get('retired_at') if tombstone else None
            return retired or 0
    return None


def _save(authority, value, *, claim=False, native=None):
    def write(conn):
        _epoch(conn, authority.epoch)
        # Retirement can start before the next owner advances the epoch.
        if getattr(authority, 'retiring', False) or getattr(authority, 'retired', False):
            raise RuntimeStoreError('runtime_retired')
        key = value.get('ledger_prefix', PREFIX) + value['origin_id']
        prior = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        if prior:
            retained = json.loads(prior[0])
            if retained.get('payload_compacted_at'):
                raise RuntimeStoreError('permission_denied')
            for field in ('source_retired_at', 'created_at'):
                if field in retained:
                    value[field] = retained[field]
        if claim or not prior:
            retired = value.get('source_retired_at')
            if retired is None:
                retired = _source_fence(conn, value, native or value.get('native') or {})
            if retired is not None:
                from hermes_state_mutation_retirement import RETRY_HORIZON_S
                value.update(source_retired_at=retired, state='refused', reason='source_unauthorized',
                             uncertain=False, observed_at=time.time())
                if not prior and (not retired or retired < time.time() - RETRY_HORIZON_S):
                    from hermes_state_webhook_retirement import native_digest
                    captured = value.pop('native', None)
                    if captured is not None:
                        value['native_digest'] = native_digest(captured)
                    value.update(message='', payload_compacted_at=time.time())
        conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES (?,?)',
                     (key,
                      json.dumps(value, ensure_ascii=False, separators=(',', ':'))))
    authority.db._execute_write(write)
    return value


def _state(authority, value, state, reason=None):
    value.update(state=state, reason=reason, observed_at=time.time())
    return _save(authority, value)


def _source(authority, admission_id):
    from gateway.platforms.webhook_delivery import validate_destination
    from gateway.webhook_target import is_bot_chat
    row = get_session_admission(authority.db, admission_id=admission_id)
    if row is None:
        raise RuntimeStoreError('not_found')
    native = row.get('payload', {}).get('native_text_v1', {})
    if not is_bot_chat(native.get('webhook_delivery', {}).get('deliver')):
        return None
    if (native.get('source', {}).get('platform') != 'webhook'
            or row['status'] not in {'terminal', 'unknown'}):
        raise RuntimeStoreError('permission_denied')
    if native.get('automation') is not None:
        from gateway.webhook_completion import verify_root
        verify_root(authority, row, native)
    delivery = validate_destination(native['webhook_delivery'])
    return row, native, delivery


def _authorize(authority, native, adapter):
    from gateway.platforms.webhook_delivery import route_digest, validate_destination
    from gateway.session_envelope import restore_native, _validate_native
    if getattr(authority, 'retiring', False) or getattr(authority, 'retired', False):
        raise RuntimeStoreError('runtime_retired')
    proof = {key: value for key, value in native.items() if key != 'automation'}
    event = restore_native({'text': '', 'native_text_v1': proof}, authority.runner)
    _validate_native(authority.runner, event, native.get('provenance'))
    if (adapter is None or authority.runner._adapter_for_source(event.source) is not adapter
            or native['webhook_route'] != route_digest(adapter, native['source']['chat_id'])):
        raise RuntimeStoreError('admission_conflict')
    validate_destination(native['webhook_delivery'])
    # Epoch and owner are rechecked before a physical RPC, not only at publication.
    with authority.db._read_ctx() as conn:
        _epoch(conn, authority.epoch)


def _materialize(authority, row, native, delivery):
    from gateway.platforms.webhook_bot_chat import result_id, presentation, wrapped_message
    from gateway.session_results import admission_result
    from gateway.webhook_target import target_home
    home = Path(authority.db.db_path).resolve().parent
    saved = admission_result(authority.db, row['admission_id']) if row['status'] == 'terminal' else None
    policy = (saved or {}).get('webhook_presentation')
    if saved is None and row.get('outcome') in {'completed', 'interrupted'}:
        policy = dict(disposition='ambiguous', category='result', text='', reason='source_result_unavailable')
    if policy is None:
        policy = presentation((saved or {}).get('result', {}), row.get('outcome') or 'unknown')
    state = policy['disposition']
    message = ''
    try:
        if state == 'pending':
            message = wrapped_message(delivery['route'], policy['text'])
    except Exception:  # health: allow BLE001 -- forced-redactor failure must never publish raw content
        state = 'refused'
    pin = native.get('webhook_target')
    # An old obligation is NEVER assigned a historical pin from today's profile.
    target = pin['home'] if pin else str(target_home(delivery['deliver'], home))
    kind = 'completion_final' if native.get('automation') is not None else 'agent_final'
    delivery_id = result_id(home, row['admission_id']) if kind == 'agent_final' else hashlib.sha256(
        json.dumps(['webhook-completion-v1', str(home), row['admission_id']], separators=(',', ':')).encode()).hexdigest()
    return _save(authority, dict(v=1, origin_kind=kind, origin_id=row['admission_id'],
        producer_home=str(home), source_outcome=row.get('outcome') or 'unknown',
        route=delivery['route'], route_digest=native['webhook_route'], target=delivery['deliver'],
        target_home=target, target_pin=pin, message=message,
        message_digest=hashlib.sha256(message.encode()).hexdigest(), category=policy['category'], author=None,
        delivery_id=delivery_id, state=state,
        reason='redaction_failed' if state == 'refused' else policy.get('reason'),
        ever_admitted=False, uncertain=False, pre_admission=False, native_admission_id=None), native=native)


def _validate_receipt(value, receipt):
    if (not isinstance(receipt, dict) or receipt.get('delivery_id') != value['delivery_id']
            or receipt.get('profile_home') != value['target_home']
            or receipt.get('message') != value['message'] or receipt.get('author') != value['author']
            or receipt.get('notification_category', 'result') != value['category']
            or not isinstance(receipt.get('session_id'), str) or not receipt['session_id']
            or receipt.get('status') not in {'queued', 'claimed', 'settled', 'failed', 'cancelled', 'ambiguous'}):
        raise ValueError('receipt identity mismatch')
    admission = receipt.get('admission_id')
    if admission is not None and (not isinstance(admission, str) or not admission):
        raise ValueError('receipt admission malformed')
    if value['native_admission_id'] is not None and admission != value['native_admission_id']:
        raise ValueError('receipt admission changed')
    if value.get('native_session_id') is not None and receipt['session_id'] != value['native_session_id']:
        raise ValueError('receipt session changed')
    if admission is None and receipt.get('refused') is not True:
        from gateway.session_bot import _is_pre_admission_pin
        if not _is_pre_admission_pin(Path(value['delivery_id'] + '.json'), receipt):
            raise ValueError('receipt admission missing or malformed')


def _legacy_evidence(value, receipt):
    """Unpinned legacy finals may OBSERVE proven native history, never admit anew."""
    import sqlite3
    from gateway.webhook_target import capture_target
    pin = capture_target(value['target_home'])
    if not pin['store'] or not receipt.get('admission_id'):
        raise ValueError('legacy unpinned')
    path = Path(value['target_home']) / 'state.db'
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as conn:
        row = conn.execute('SELECT request_id,target_session_id,payload_json FROM session_admissions WHERE admission_id=?',
                           (receipt['admission_id'],)).fetchone()
    if row is None or row[:2] != ('bot:' + value['delivery_id'], receipt['session_id']):
        raise ValueError('legacy receipt not proven')
    payload = json.loads(row[2])
    if payload.get('text') != value['message']:
        raise ValueError('legacy message not proven')
    value['receipt_evidence_pin'] = pin


def _observe(authority, value, receipt):
    _validate_receipt(value, receipt)
    if value['target_pin'] is None and not value.get('receipt_evidence_pin'):
        _legacy_evidence(value, receipt)
    admission = receipt.get('admission_id')
    if admission:
        value.update(ever_admitted=True, native_admission_id=admission,
                     native_session_id=receipt['session_id'], pre_admission=False)
    else:
        value['pre_admission'] = receipt.get('refused') is not True
    value['native_status'] = receipt['status']
    state = 'refused' if receipt.get('refused') is True else {'cancelled': 'failed'}.get(receipt['status'], receipt['status'])
    return _state(authority, value, state, 'native_' + state if state in {'failed', 'refused', 'ambiguous'} else None)


def _handoff(authority, value, native, adapter):
    from tools.bot_live_delivery import read_delivery_result, find_canonical_live_owner, deliver_to_live_owner, PreDialError
    from gateway.webhook_target import refresh_target
    try:
        pin = value['target_pin'] or value.get('receipt_evidence_pin')
        if pin:
            current = refresh_target(pin)
            if value['target_pin'] is not None and current != pin:
                value['target_pin'] = current
                _save(authority, value)  # freeze owner-created first store BEFORE dial
        receipt = read_delivery_result(value['target_home'], value['delivery_id'])
        if receipt is not None:
            return _observe(authority, value, receipt)
        if value['ever_admitted'] or (value['uncertain'] and not value['pre_admission']):
            return _state(authority, value, 'ambiguous', 'missing_native_evidence')
        if value['target_pin'] is None:
            return _state(authority, value, 'refused', 'legacy_unpinned')
        # A withdrawn native pre-admission pin proves no admission; only its owner withdraws it.
        value.update(pre_admission=False, uncertain=False)
        if value['target_pin']['store'] is None:
            return _state(authority, value, 'pending', 'owner_unavailable')
        try:
            owner = find_canonical_live_owner(value['target_home'])
        except (ValueError, OSError):
            owner = None
        if owner is None:
            return _state(authority, value, 'pending', 'owner_unavailable')
        _authorize(authority, native, adapter)
        value['uncertain'] = True
        _save(authority, value, claim=True, native=native)  # claim before, never during, physical I/O
        if value['state'] == 'refused':
            return value
        try:
            receipt = deliver_to_live_owner(value['target_home'], owner, value['message'],
                delivery_id=value['delivery_id'], notification_category=value['category'],
                expected_incarnation=value['target_pin'])
        except PreDialError:
            value['uncertain'] = False
            return _state(authority, value, 'pending', 'owner_unavailable')
        except Exception:  # health: allow BLE001 -- post-dial exceptions can contain private payloads
            return _state(authority, value, 'ambiguous', 'handoff_unverified')
        return _observe(authority, value, receipt)
    except (ValueError, OSError, KeyError, TypeError):
        return _state(authority, value, 'refused', 'target_or_receipt_mismatch')


def _reconcile(authority, adapter, admission_id):
    value = read_intent(authority, admission_id)
    if value and value['state'] in TERMINAL:
        return value
    try:
        source = _source(authority, admission_id)
        if source is None:
            if value:
                return _state(authority, value, 'refused', 'source_unauthorized')
            return None
        row, native, delivery = source
        value = value or _materialize(authority, row, native, delivery)
        _authorize(authority, native, adapter)
    except (RuntimeStoreError, ValueError, KeyError, TypeError):
        if value:
            return _state(authority, value, 'refused', 'source_unauthorized')
        raise
    if (value['state'] in TERMINAL or value['source_outcome'] == 'unknown'
            or value.get('reason') == 'source_result_unavailable'):
        return value
    return _handoff(authority, value, native, adapter)


async def reconcile_final(authority, adapter, admission_id):
    from gateway.session_runtime_workers import track_mutation, tracked_write
    from gateway.session_authorities import async_owner_scope
    async def run():
        locks = getattr(authority, '_webhook_intent_locks', None)
        if locks is None:
            locks = authority._webhook_intent_locks = {}
        lock = locks.setdefault(admission_id, asyncio.Lock())
        try:
            async with lock:
                async with async_owner_scope(authority):
                    return await tracked_write(authority, partial(_reconcile, authority, adapter, admission_id))
        finally:
            # Waiters hold the same lock object until they acquire it.
            if not lock.locked() and not getattr(lock, '_waiters', None):
                locks.pop(admission_id, None)
    return await asyncio.shield(track_mutation(authority, run()))
