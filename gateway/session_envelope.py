"""Owner-only durable native input codec; never a client-supplied wire format.

Ordinary context and local media are retained without serializing delegated trust.
Multiplex callbacks bind current connector ownership; roles require fresh preflight.
Relay snapshots bind the authenticated connector and exact source; replay rechecks both.
"""
from copy import copy, deepcopy
from datetime import datetime

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.session_ingress_media import capture_native_media, restore_native_media
from hermes_state_runtime import RuntimeStoreError

_EVENT_FIELDS = (
    'user_id', 'user_name', 'message_id', 'platform_update_id',
    'reply_to_message_id', 'reply_to_text', 'reply_to_author_id',
    'reply_to_author_name', 'reply_to_is_own_message', 'allow_gateway_control',
)
_CONTEXT_FIELDS = ('auto_skill', 'channel_prompt', 'channel_context')


def _validate_native(runner, event, provenance=None, fresh_roles=False):
    source = event.source
    if (source is None or not isinstance(event.text, str)
            or not isinstance(event.message_type, MessageType)
            or event.message_type == MessageType.COMMAND or event.is_command()
            or event.internal or event.metadata or event.prompt_response
            or (source.role_authorized and not fresh_roles)
            or (source.delivered_via_upstream_relay and not (provenance and 'relay' in provenance))
            or source.profile_route_rejected
            or (getattr(source, '_authorization_profile_home', None) is not None and provenance is None)
            or (getattr(runner.config, 'multiplex_profiles', False) and provenance is None)):
        raise RuntimeStoreError('invalid_params')
    if provenance is not None:
        from gateway.session_ingress_context import restore_provenance
        restore_provenance(runner, source, provenance)
    delegated = fresh_roles or bool(provenance and 'relay' in provenance)
    if not runner._is_user_authorized_for_source(source, allow_adapter_delegation=delegated):
        raise RuntimeStoreError('permission_denied')
    if any(value is not None and not isinstance(value, str)
           for value in (event.channel_prompt, event.channel_context)):
        raise RuntimeStoreError('invalid_params')
    skill = event.auto_skill
    if not (skill is None or isinstance(skill, str)
            or isinstance(skill, list) and all(isinstance(item, str) for item in skill)):
        raise RuntimeStoreError('invalid_params')
    if (not isinstance(event.media_urls, list) or not all(isinstance(item, str) for item in event.media_urls)
            or not isinstance(event.media_types, list) or not all(isinstance(item, str) for item in event.media_types)
            or not isinstance(event.media_text_inlined, list)
            or not all(item is None or type(item) is bool for item in event.media_text_inlined)
            or len(event.media_types) not in (0, len(event.media_urls))
            or len(event.media_text_inlined) not in (0, len(event.media_urls))):
        raise RuntimeStoreError('invalid_params')
    encoded_source = source.to_dict()
    encoded_source['is_bot'] = source.is_bot
    restored_source = SessionSource.from_dict(encoded_source)
    restored_source.is_bot = source.is_bot
    if provenance is not None:
        restore_provenance(runner, restored_source, provenance)
    adapter = runner._adapter_for_source(source)
    if adapter is None or runner._adapter_for_source(restored_source) is not adapter:
        raise RuntimeStoreError('not_found')
    return encoded_source


def snapshot_native(runner, event):
    from gateway.session_ingress_context import capture_provenance
    provenance = capture_provenance(runner, event)
    return _snapshot_native(runner, event, provenance)


async def prepare_native(runner, event):
    from dataclasses import replace
    from gateway.session_ingress_context import capture_provenance, reauthorize_roles
    provenance = capture_provenance(runner, event)
    # Freeze caller-owned identity and content before the first network yield.
    # Preserve dynamic trust fields too: validation must reject forged ones,
    # not silently discard them while freezing the caller's source identity.
    frozen = replace(event, source=copy(event.source))
    fresh_roles = await reauthorize_roles(runner, frozen.source, provenance)
    return _snapshot_native(runner, frozen, provenance, fresh_roles)


def _snapshot_native(runner, event, provenance, fresh_roles=False):
    encoded_source = _validate_native(runner, event, provenance, fresh_roles)
    envelope = {'source': encoded_source,
                'route': runner.session_store._generate_session_key(event.source),
                'event': deepcopy({name: getattr(event, name) for name in _EVENT_FIELDS}),
                'timestamp': event.timestamp.isoformat()}
    if event.source.platform.value == 'webhook':
        from gateway.platforms.webhook_delivery import route_digest, snapshot_destination
        adapter = runner._adapter_for_source(event.source)
        if provenance is None:
            raise RuntimeStoreError('permission_denied')
        delivery = snapshot_destination(adapter, adapter._delivery_info.get(event.source.chat_id))
        envelope['webhook_delivery'] = delivery
        envelope['webhook_route'] = route_digest(adapter, event.source.chat_id)
    if event.source.role_authorized:
        envelope['reauthorize'] = 'roles'
    if provenance is not None:
        envelope['provenance'] = provenance
    # Omit new defaults so an identical retry of an older text admission retains
    # its fingerprint. Explicit context (including an empty skill list) is exact.
    envelope['event'].update({name: deepcopy(getattr(event, name)) for name in _CONTEXT_FIELDS
                              if getattr(event, name) is not None})
    if event.message_type != MessageType.TEXT:
        envelope['message_type'] = event.message_type.value
    if event.media_urls:
        envelope['media'] = capture_native_media(event.media_urls)
        envelope['event'].update(media_types=list(event.media_types),
                                 media_text_inlined=list(event.media_text_inlined))
    # Keep the private dispatch key so previously committed text-only rows recover
    # under the same authority; optional media/context fields extend that envelope.
    payload = {'text': event.text, 'native_text_v1': envelope}
    # Only an exact persisted legacy retry retains its old fingerprint. New
    # ordinary admissions must bind homes too, including execution after restart.
    if provenance is not None and not getattr(runner.config, 'multiplex_profiles', False):
        from hermes_state_runtime import list_session_admissions
        from gateway.session_authorities import active_authority
        authority = active_authority(runner)
        entry = runner.session_store.lookup_by_session_key(envelope['route'])
        if entry is not None:
            legacy = deepcopy(payload)
            legacy['native_text_v1'].pop('provenance')
            for row in list_session_admissions(authority.db, session_id=entry.session_id, pending_only=False):
                if row['request_id'] == event.message_id and row['payload'] == legacy:
                    return legacy
    return payload


def restore_native(payload, runner=None):
    envelope = payload['native_text_v1']
    source = SessionSource.from_dict(envelope['source'])
    source.is_bot = envelope['source']['is_bot']
    # Private descriptor requests a fresh check, never supplies the verdict.
    source.role_authorized = envelope.get('reauthorize') == 'roles'
    if 'provenance' in envelope:
        from gateway.session_ingress_context import callback_runner, restore_provenance
        runner = runner or callback_runner()
        if runner is not None:
            restore_provenance(runner, source, envelope['provenance'])
    event = MessageEvent(text=payload['text'], source=source,
                        timestamp=datetime.fromisoformat(envelope['timestamp']),
                        message_type=MessageType(envelope.get('message_type', 'text')),
                        media_urls=restore_native_media(envelope.get('media', [])),
                        **deepcopy(envelope['event']))
    if 'automation' in envelope:
        event.internal = True
        if envelope['automation'].get('heartbeat'):
            event._heartbeat_session_id = envelope['automation']['heartbeat']
        event.metadata = {'gateway_session_key': envelope['route'],
                          'gateway_session_id': envelope['automation']['owner']}
    return event


async def check_native_route(runner, payload, session_id, available_source, adapter):
    """Read-only preflight: route/auth rejection must never consume a queued row."""
    if 'automation' in payload['native_text_v1']:
        from gateway.session_automation import check_automation_route
        return check_automation_route(runner, payload, session_id, available_source, adapter)
    event = restore_native(payload, runner)
    # Validate the stored sender without recapturing files or trusting the binding caller.
    from gateway.session_ingress_context import reauthorize_roles
    provenance = payload['native_text_v1'].get('provenance')
    fresh_roles = await reauthorize_roles(runner, event.source, provenance)
    _validate_native(runner, event, provenance, fresh_roles)
    route = payload['native_text_v1']['route']
    store = runner.session_store
    if (store._generate_session_key(event.source) != route
            or store._generate_session_key(available_source) != route):
        raise RuntimeStoreError('admission_conflict')
    entry = store.lookup_by_session_key(route)
    if entry is None or entry.session_id != session_id:
        raise RuntimeStoreError('admission_conflict')
    if adapter is None or runner._adapter_for_source(event.source) is not adapter:
        raise RuntimeStoreError('not_found')
    if event.source.platform.value == 'webhook':
        from gateway.platforms.webhook_delivery import route_digest, validate_destination
        envelope = payload['native_text_v1']
        validate_destination(envelope.get('webhook_delivery'))
        if envelope.get('webhook_route') != route_digest(adapter, event.source.chat_id):
            raise RuntimeStoreError('admission_conflict')
    return event.source, route
