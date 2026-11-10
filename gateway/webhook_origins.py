"""Authenticated direct/hook preparation, never a fabricated agent admission.

The provider key is stable across changes in content or destination. Preparing is
committed before scripts/hooks; an interrupted side effect is not replay permission.
"""
import asyncio
from datetime import UTC, datetime
from functools import partial
import hashlib
import json
from pathlib import Path
import time

from hermes_state_runtime import RuntimeStoreError
from gateway.webhook_final_intent import _save, _state, _authorize, _handoff, TERMINAL

PREFIX = 'webhook.origin.v1.'


def read_origin(authority, key):
    row = authority.db._read_one('SELECT value FROM state_meta WHERE key=?', (PREFIX + key,))
    return json.loads(row['value']) if row else None


def origin_event(adapter, route_config, route_name, profile, payload, delivery_id):
    from gateway.platforms.event import MessageEvent, MessageType
    source = adapter._admission_source(route_name, profile, delivery_id)
    now = time.time()
    adapter._delivery_info[source.chat_id] = dict(deliver=route_config.get('deliver', 'log'), profile=profile,
        deliver_extra=adapter._render_delivery_extra(route_config.get('deliver_extra', {}), payload),
        route=route_name, mirror=route_config.get('mirror_to_session') is True)
    adapter._delivery_info_created[source.chat_id] = now
    adapter._delivery_info_order.append((now, source.chat_id))
    adapter._prune_delivery_info(now)
    return MessageEvent(text='', message_type=MessageType.TEXT, source=source, raw_message=payload,
        message_id=delivery_id, timestamp=datetime.fromtimestamp(0, UTC))


def _identity(authority, principal, request_id, native):
    home = str(Path(authority.db.db_path).resolve().parent)
    return hashlib.sha256(json.dumps(['webhook-origin-v1', home, native['provenance']['transport_home'],
        native['provenance']['platform'], principal, request_id], separators=(',', ':')).encode()).hexdigest()


def _begin(authority, key, principal, event, native, fingerprint):
    pin = native['webhook_target']
    return _save(authority, dict(v=1, ledger_prefix=PREFIX, origin_id=key, origin_kind='preparing',
        created_at=time.time(),
        producer_home=str(Path(authority.db.db_path).resolve().parent), principal=principal,
        request_id=event.message_id, fingerprint=fingerprint, native=native,
        route=native['webhook_delivery']['route'], route_digest=native['webhook_route'],
        target=native['webhook_delivery']['deliver'], target_home=pin['home'], target_pin=pin,
        source_outcome='unknown', state='preparing', reason=None, message='',
        message_digest=hashlib.sha256(b'').hexdigest(), category='result', author=None,
        delivery_id=hashlib.sha256(('webhook-origin-delivery-v1:' + key).encode()).hexdigest(), ever_admitted=False,
        uncertain=False, pre_admission=False, native_admission_id=None))


def _recover_preparing(authority, value):
    from gateway.platforms.webhook_ingress import _committed
    rows = _committed(authority, value['principal'], value['request_id'])
    if len(rows) == 1:
        from gateway.platforms.webhook_bot_chat import result_id
        value.update(origin_kind='linked_native_admission', source_admission_id=rows[0]['admission_id'],
                     delivery_id=result_id(value['producer_home'], rows[0]['admission_id']), source_outcome='admitted')
        return _state(authority, value, 'linked_native_admission')
    return _state(authority, value, 'origin_unknown', 'preparation_interrupted')


def _finish(authority, value, kind, text, *, outcome='completed', muted=False, category='result'):
    from gateway.platforms.webhook_bot_chat import presentation, wrapped_message
    policy = presentation({'final_response': text}, outcome, muted=muted, category=category)
    value.update(origin_kind=kind, source_outcome=outcome, category=policy['category'])
    message = ''
    try:
        if policy['disposition'] == 'pending':
            message = wrapped_message(value['route'], policy['text'])
    except Exception:  # health: allow BLE001 -- never retain raw output when redaction fails
        return _state(authority, value, 'refused', 'redaction_failed')
    value.update(message=message, message_digest=hashlib.sha256(message.encode()).hexdigest())
    return _state(authority, value, policy['disposition'], policy.get('reason'))


def _reconcile_origin(authority, adapter, key):
    value = read_origin(authority, key)
    if value is None or value['state'] in TERMINAL or value['state'] in {'origin_unknown', 'linked_native_admission'}:
        return value
    if value.get('source_retired_at') is not None:
        return _state(authority, value, 'refused', 'source_unauthorized')
    try:
        if value['native'].get('automation') is not None:
            from gateway.webhook_completion import verify_root
            verify_root(authority, {'target_session_id': value['native']['automation']['owner']}, value['native'])
        _authorize(authority, value['native'], adapter)
    except (RuntimeStoreError, ValueError, KeyError, TypeError):
        return _state(authority, value, 'refused', 'source_unauthorized')
    if value['state'] == 'preparing':
        return _recover_preparing(authority, value)
    return _handoff(authority, value, value['native'], adapter)


async def reconcile_origin(authority, adapter, key):
    from gateway.platforms.webhook_ingress import _delivery_turn
    from gateway.session_runtime_workers import track_mutation, tracked_write
    from gateway.session_authorities import async_owner_scope
    async def run():
        async with async_owner_scope(authority):
            value = read_origin(authority, key)
            if value is None:
                return None
            async with _delivery_turn((authority.profile_id, value['principal'], value['request_id'])):
                return await tracked_write(authority, partial(_reconcile_origin, authority, adapter, key))
    return await asyncio.shield(track_mutation(authority, run()))


async def reconcile_origins(authority):
    from gateway.webhook_reconcile import PASS_LIMIT, restored_adapter
    cursor = getattr(authority, '_webhook_origin_cursor', '')
    query = """SELECT key,value FROM state_meta WHERE key>=? AND key<? AND key>?
        AND json_extract(value,'$.state') NOT IN
        ('settled','suppressed','refused','failed','origin_unknown','linked_native_admission') ORDER BY key LIMIT ?"""
    def read(after):
        return authority.db._read_all(query, (PREFIX, PREFIX + '~', after, PASS_LIMIT))
    rows = await asyncio.to_thread(read, cursor)
    if not rows and cursor:
        rows = await asyncio.to_thread(read, '')
    authority._webhook_origin_cursor = rows[-1]['key'] if rows else ''
    for row in rows:
        value = json.loads(row['value'])
        adapter = restored_adapter(authority, {'text': '', 'native_text_v1': value['native']})
        await reconcile_origin(authority, adapter, value['origin_id'])


def _response(value):
    from aiohttp import web
    return web.json_response(dict(status=value['state'], origin_id=value['origin_id'],
        origin_kind=value['origin_kind'], delivery_id=value['delivery_id'],
        **({'source_admission_id': value['source_admission_id']} if value.get('source_admission_id') else {})), status=202)


async def _render(adapter, config, payload, event_type, route_name):
    from gateway.platforms.webhook_mcp_events import render_mcp_event_prompt
    if config.get('script'):
        keep, transformed = await asyncio.to_thread(adapter._route_processor.run_route_script, config['script'], payload)
        if not keep:
            return None, payload
        payload = transformed or payload
    prompt = (render_mcp_event_prompt(adapter, config, payload, event_type, route_name)
              or adapter._render_prompt(config.get('prompt', ''), payload, event_type, route_name))
    if config.get('skills'):
        prompt = adapter._apply_skills(prompt, config['skills'])
    return prompt, payload


async def _origin_source(authority, event):
    """Origins are text-only internal events; validate the live source before lookup."""
    from gateway.session_envelope import prepare_native_source, restore_native
    payload = await prepare_native_source(authority.runner, event)
    source = restore_native(payload).source
    identity = json.dumps([source.profile, source.platform.value, source.chat_id,
                           source.thread_id, source.user_id], separators=(',', ':'))
    return payload, 'messaging:' + identity


def _check_origin(value, native, fingerprint):
    # Only the captured destination incarnation is excluded. Current provenance,
    # source/principal, route and request bytes must still describe the same origin.
    from hermes_state_webhook_retirement import native_digest
    retained = value.get('native_digest') if value.get('payload_compacted_at') else native_digest(value['native'])
    if value['fingerprint'] != fingerprint or retained != native_digest(native):
        raise RuntimeStoreError('admission_conflict')


async def _prepare_once(authority, adapter, event, config, event_type, fingerprint):
    from gateway.platforms.webhook_ingress import _delivery_turn
    from gateway.session_envelope import pin_webhook_target
    from gateway.session_runtime_workers import tracked_write
    authority._require_admission_open()
    payload, principal = await _origin_source(authority, event)
    native = payload['native_text_v1']
    key = _identity(authority, principal, event.message_id, native)
    async with _delivery_turn((authority.profile_id, principal, event.message_id)):
        value = read_origin(authority, key)
        if value:
            _check_origin(value, native, fingerprint)
            return await tracked_write(authority, partial(_reconcile_origin, authority, adapter, key))
        # A pre-upgrade real admission must still dedupe before any side effects.
        from gateway.platforms.webhook_ingress import _webhook_retry, _committed
        prior = await _webhook_retry(authority, event, payload, _committed(authority, principal, event.message_id))
        if prior is not None:
            return dict(state='linked_native_admission', origin_id=key, origin_kind='linked_native_admission',
                        delivery_id=key, source_admission_id=prior.admission_id)
        pin_webhook_target(authority.runner, event, payload)
        value = await tracked_write(authority, partial(_begin, authority, key, principal, event, native, fingerprint))
        prompt, transformed = await _render(adapter, config, event.raw_message, event_type, value['route'])
        if prompt is None:
            return await tracked_write(authority, partial(_finish, authority, value, 'direct_template', ''))
        if config.get('deliver_only'):
            value = await tracked_write(authority, partial(_finish, authority, value, 'direct_template', prompt))
        else:
            value = await _hook_or_admit(authority, adapter, event, prompt, transformed, payload, principal, value)
        return await tracked_write(authority, partial(_reconcile_origin, authority, adapter, key))


async def _hook_or_admit(authority, adapter, event, prompt, transformed, payload, principal, value):
    from gateway.platforms.webhook_ingress import _admit_once
    from gateway.run_inbound_consumer import POST_ADMISSION_HOOK
    from gateway.session_runtime_workers import tracked_write
    from hermes_cli.lifecycle import has_hook
    event.text, event.raw_message = prompt, transformed
    event.source._webhook_target_pin = value['target_pin']
    hooks = [name for name in ('pre_gateway_dispatch', POST_ADMISSION_HOOK) if has_hook(name)]
    receipt = await _admit_once(authority, event, True, hooks, False, (payload, []))
    if receipt is None:
        return await tracked_write(authority, partial(_finish, authority, value, 'hook_reply',
            getattr(event, '_consumer_reply', None)))
    from gateway.platforms.webhook_bot_chat import result_id
    value.update(origin_kind='linked_native_admission', source_admission_id=receipt.admission_id,
                 delivery_id=result_id(value['producer_home'], receipt.admission_id), source_outcome='admitted')
    await tracked_write(authority, partial(_state, authority, value, 'linked_native_admission'))
    task = asyncio.create_task(adapter._finalize_delivery(event, authority, receipt))
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
    return value


async def _rendered_once(authority, adapter, event, prompt, config):
    from gateway.platforms.webhook_ingress import _delivery_turn, _webhook_retry, _committed
    from gateway.session_envelope import pin_webhook_target
    from gateway.session_runtime_workers import tracked_write
    authority._require_admission_open()
    payload, principal = await _origin_source(authority, event)
    native = payload['native_text_v1']
    key = _identity(authority, principal, event.message_id, native)
    fingerprint = hashlib.sha256(json.dumps([prompt, event.raw_message], sort_keys=True,
                                            separators=(',', ':')).encode()).hexdigest()
    async with _delivery_turn((authority.profile_id, principal, event.message_id)):
        value = read_origin(authority, key)
        if value:
            _check_origin(value, native, fingerprint)
            return await tracked_write(authority, partial(_reconcile_origin, authority, adapter, key))
        prior = await _webhook_retry(authority, event, payload, _committed(authority, principal, event.message_id))
        if prior is not None:
            return None
        pin_webhook_target(authority.runner, event, payload)
        value = await tracked_write(authority, partial(_begin, authority, key, principal, event, native, fingerprint))
        await _hook_or_admit(authority, adapter, event, prompt, event.raw_message, payload, principal, value)
        return await tracked_write(authority, partial(_reconcile_origin, authority, adapter, key))


async def admit_rendered_origin(adapter, config, route_name, profile, payload, prompt, delivery_id):
    """The existing in-memory coalescer hands off here; no second buffering mechanism."""
    from gateway.platforms.webhook_ingress import producer_scope
    from gateway.session_runtime_workers import track_mutation
    event = origin_event(adapter, config, route_name, profile, payload, delivery_id)
    async with producer_scope(adapter, event) as authority:
        await asyncio.shield(track_mutation(authority, _rendered_once(authority, adapter, event, prompt, config)))
    return 'accepted'


async def handle_origin(adapter, config, route_name, profile, payload, event_type, delivery_id, raw_body):
    from aiohttp import web
    from gateway.platforms.webhook_ingress import producer_scope
    from gateway.session_runtime_workers import track_mutation
    event = origin_event(adapter, config, route_name, profile, payload, delivery_id)
    try:
        async with producer_scope(adapter, event) as authority:
            # Shield the WHOLE preparation: cancellation must not release its lock while a script
            # thread or hook still owns side effects or publication.
            fingerprint = hashlib.sha256(json.dumps([event_type, hashlib.sha256(raw_body).hexdigest()],
                separators=(',', ':')).encode()).hexdigest()
            work = _prepare_once(authority, adapter, event, config, event_type, fingerprint)
            value = await asyncio.shield(track_mutation(authority, work))
            return _response(value)
    except RuntimeStoreError as exc:
        return web.json_response({'error': 'Origin conflict' if exc.reason == 'admission_conflict' else 'Origin unavailable'},
                                 status=409 if exc.reason == 'admission_conflict' else 503)
    except Exception:  # health: allow BLE001 -- scripts/hooks may include secrets in exceptions
        return web.json_response({'error': 'Origin preparation unavailable'}, status=503)
