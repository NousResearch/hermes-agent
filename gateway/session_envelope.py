"""Owner-only durable native input codec; never a client-supplied wire format.

Ordinary context and local media are retained without serializing delegated trust.
Multiplex callbacks bind current connector ownership; roles require fresh preflight.
Relay snapshots bind the authenticated connector and exact source; replay rechecks both.
"""
import asyncio
from copy import copy, deepcopy
from dataclasses import replace
from datetime import datetime
import json
import logging

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.session_ingress_media import capture_native_media, restore_native_media
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

_EVENT_FIELDS = (
    'user_id', 'user_name', 'message_id', 'platform_update_id',
    'reply_to_message_id', 'reply_to_text', 'reply_to_author_id',
    'reply_to_author_name', 'reply_to_is_own_message', 'allow_gateway_control',
)
# ``reply_anchor_override`` = '' is Telegram topic transfer's "do not quote the root DM" (#139136).
_CONTEXT_FIELDS = ('auto_skill', 'channel_prompt', 'channel_context', 'reply_expected', 'reply_anchor_override')
# Closed metadata contract: only the keys the Baileys adapter stamps on ordinary inbound
# (``WhatsAppAdapter._build_message_event``) cross native admission; any other key refuses.
_WHATSAPP_METADATA = frozenset({'whatsapp_native_type', 'whatsapp_native', 'whatsapp_from_owner'})
# Top-level ``nativeMetadata`` keys the bridge emits (scripts/whatsapp-bridge/bridge_helpers.js).
_WHATSAPP_NATIVE_KEYS = frozenset({'video', 'audio', 'sticker', 'location', 'contact', 'contacts',
                                   'reaction', 'poll', 'pollUpdate'})
_WHATSAPP_NATIVE_MAX_BYTES = 16 * 1024
_MEDIA_FIELDS = ('media_urls', 'media_types', 'media_text_inlined')


def _native_metadata(event):
    """JSON projection of the adapter metadata a native replay keeps; ``None`` refuses the event.
    An oversized ``whatsapp_native`` (vCards with photos) is dropped, never refused."""
    metadata = event.metadata
    if not metadata:
        return {}
    if (not isinstance(metadata, dict) or event.source.platform.value != 'whatsapp'
            or not set(metadata) <= _WHATSAPP_METADATA):
        return None
    kind, native = metadata.get('whatsapp_native_type'), metadata.get('whatsapp_native')
    if (kind is not None and not (isinstance(kind, str) and 0 < len(kind) <= 64)
            or metadata.get('whatsapp_from_owner') not in (None, True)
            or native is not None and not (isinstance(native, dict) and set(native) <= _WHATSAPP_NATIVE_KEYS)):
        return None
    projected = {key: metadata[key] for key in ('whatsapp_native_type', 'whatsapp_from_owner') if key in metadata}
    if native is not None:
        try:
            encoded = json.dumps(native, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return None
        if len(encoded.encode()) <= _WHATSAPP_NATIVE_MAX_BYTES:
            projected['whatsapp_native'] = json.loads(encoded)
    return projected


def _adapter_event(event, **changes):
    """An adapter's event at the envelope boundary: an absent media list (``None``) is no media.
    Any other non-list still reaches ``_validate_native`` and refuses."""
    changes.update({name: [] for name in _MEDIA_FIELDS if getattr(event, name) is None})
    return replace(event, **changes) if changes else event


def _refuse(event, reason, clause):
    """Name the refused clause in gateway.log and on the error; the wire reason stays stable."""
    source = event.source
    logger.warning('Native admission refused: %s (%s) platform=%s chat=%s message=%s', reason, clause,
                   getattr(getattr(source, 'platform', None), 'value', None), getattr(source, 'chat_id', None),
                   event.message_id)
    raise RuntimeStoreError(reason, field=clause)


def _media_clause(event):
    urls, types, inlined = event.media_urls, event.media_types, event.media_text_inlined
    if not isinstance(urls, list) or not all(isinstance(item, str) for item in urls):
        return 'media_urls'
    if not isinstance(types, list) or not all(isinstance(item, str) for item in types):
        return 'media_types'
    if not isinstance(inlined, list) or not all(item is None or type(item) is bool for item in inlined):
        return 'media_text_inlined'
    if len(types) not in (0, len(urls)):
        return 'media_types_length'
    if len(inlined) not in (0, len(urls)):
        return 'media_text_inlined_length'
    return None


def _event_clause(runner, event, provenance, fresh_roles):
    source = event.source
    if source is None:
        return 'source'
    if not isinstance(event.text, str):
        return 'text'
    if not isinstance(event.message_type, MessageType):
        return 'message_type'
    # The ingress router's own classifier: ``/tmp/x is broken`` is text, not a command.
    if event.message_type == MessageType.COMMAND or event.get_command():
        return 'command'
    if event.internal:
        return 'internal'
    if _native_metadata(event) is None:
        return 'metadata'
    if event.prompt_response:
        return 'prompt_response'
    if source.role_authorized and not fresh_roles:
        return 'role_authorized_without_fresh_roles'
    if source.delivered_via_upstream_relay and not (provenance and 'relay' in provenance):
        return 'upstream_relay_without_provenance'
    if source.profile_route_rejected:
        return 'profile_route_rejected'
    if getattr(source, '_authorization_profile_home', None) is not None and provenance is None:
        return 'authorization_home_without_provenance'
    if getattr(runner.config, 'multiplex_profiles', False) and provenance is None:
        return 'multiplex_without_provenance'
    return None


def _validate_native(runner, event, provenance=None, fresh_roles=False):
    source = event.source
    if (clause := _event_clause(runner, event, provenance, fresh_roles)) is not None:
        _refuse(event, 'invalid_params', clause)
    if provenance is not None:
        from gateway.session_ingress_context import restore_provenance
        restore_provenance(runner, source, provenance)
    delegated = fresh_roles or bool(provenance and 'relay' in provenance)
    if not delegated:
        # An adapter declaring ``authorization_is_upstream`` (A2A peers: bearer-token
        # authenticated in do_POST) owns the whole authorization, so its fresh native inputs
        # carry no relay seal to derive delegation from. Without this they fell to the
        # principal allowlist and — with no {PLATFORM}_ALLOWED_USERS — the admission
        # re-check denied what the inbound filter had just accepted (#139199).
        delegated = runner._adapter_flag(
            source.platform, "authorization_is_upstream", runner._adapter_profile_for_source(source))
    if not runner._is_user_authorized_for_source(source, allow_adapter_delegation=delegated):
        _refuse(event, 'permission_denied', 'user_not_authorized')
    for name in ('channel_prompt', 'channel_context', 'reply_anchor_override'):
        if getattr(event, name) is not None and not isinstance(getattr(event, name), str):
            _refuse(event, 'invalid_params', name)
    if event.reply_expected is not None and type(event.reply_expected) is not bool:
        _refuse(event, 'invalid_params', 'reply_expected')
    skill = event.auto_skill
    if not (skill is None or isinstance(skill, str)
            or isinstance(skill, list) and all(isinstance(item, str) for item in skill)):
        _refuse(event, 'invalid_params', 'auto_skill')
    if (clause := _media_clause(event)) is not None:
        _refuse(event, 'invalid_params', clause)
    encoded_source = source.to_dict()
    encoded_source['is_bot'] = source.is_bot
    restored_source = SessionSource.from_dict(encoded_source)
    restored_source.is_bot = source.is_bot
    if provenance is not None:
        restore_provenance(runner, restored_source, provenance)
    adapter = runner._adapter_for_source(source)
    if adapter is None or runner._adapter_for_source(restored_source) is not adapter:
        _refuse(event, 'not_found', 'adapter' if adapter is None else 'adapter_mismatch')
    return encoded_source


def snapshot_native(runner, event):
    from gateway.session_ingress_context import capture_provenance
    provenance = capture_provenance(runner, event)
    return _snapshot_native(runner, _adapter_event(event), provenance)


async def prepare_native(runner, event):
    from gateway.session_ingress_context import capture_provenance, reauthorize_roles
    provenance = capture_provenance(runner, event)
    # Freeze caller-owned identity and content before the first network yield.
    # Preserve dynamic trust fields too: validation must reject forged ones,
    # not silently discard them while freezing the caller's source identity.
    frozen = _adapter_event(event, source=copy(event.source))
    fresh_roles = await reauthorize_roles(runner, frozen.source, provenance)
    encoded_source = _validate_native(runner, frozen, provenance, fresh_roles)
    # Copy + SHA-256 + fsyncs of up to max_inbound_media_bytes stay off the owner loop; the
    # copied context carries the admission's capture lease.
    media = None
    if frozen.media_urls:
        from gateway.session_ingress_media import _lease_scope, abandon_capture, release_abandoned_capture
        from gateway.session_runtime_workers import track_mutation
        scope = _lease_scope.get()
        work = asyncio.to_thread(capture_native_media, frozen.media_urls)
        capture = (track_mutation(scope.authority, work) if scope is not None and scope.authority is not None
                   else asyncio.create_task(work))
        try:
            # Cancelling this waiter must not cancel the thread's completion observer: the thread
            # may publish a path after the caller's lease context has already exited.
            media = await asyncio.shield(capture)
        except asyncio.CancelledError:
            if scope is not None and scope.authority is not None:
                scope.abandoned = True
                def finished(done):
                    if not done.cancelled():
                        done.exception()  # observe a failed copy too; partial publishes still collect
                    track_mutation(scope.authority, release_abandoned_capture(scope))
                capture.add_done_callback(finished)
            raise
        except BaseException:
            await abandon_capture(scope)
            raise
    return _snapshot_native(runner, frozen, provenance, fresh_roles, encoded_source, media)


def _snapshot_native(runner, event, provenance, fresh_roles=False, encoded_source=None, media=None):
    if encoded_source is None:
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
    if event.inherit_channel_pin is True:  # synthetic goal/resume turn reuses the session's pins (#126109)
        envelope['event']['inherit_channel_pin'] = True
    if metadata := _native_metadata(event):
        envelope['event']['metadata'] = metadata
    if event.message_type != MessageType.TEXT:
        envelope['message_type'] = event.message_type.value
    if event.media_urls:
        envelope['media'] = media if media is not None else capture_native_media(event.media_urls)
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


def restore_native_source(payload, runner=None):
    """restore_native's source (provenance included) without re-verifying the captured media:
    admission needs only the route; execution's restore_native still checks every byte."""
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
    return source


def restore_native(payload, runner=None):
    envelope = payload['native_text_v1']
    source = restore_native_source(payload, runner)
    timestamp = datetime.fromisoformat(envelope['timestamp'])
    if timestamp.tzinfo is not None and timestamp.timestamp() == 0:
        # Epoch 0 pins the retry fingerprint of producers with no provider time (relay, webhook,
        # automation); the turn records local time, as main's arrival-stamped event did.
        timestamp = datetime.now()
    event = MessageEvent(text=payload['text'], source=source, timestamp=timestamp,
                        message_type=MessageType(envelope.get('message_type', 'text')),
                        media_urls=restore_native_media(envelope.get('media', [])),
                        **deepcopy(envelope['event']))
    if 'automation' in envelope:
        event.internal = True
        if envelope['automation'].get('heartbeat'):
            event._heartbeat_session_id = envelope['automation']['heartbeat']
        event.metadata = {'gateway_session_key': envelope['route'],
                          'gateway_session_id': envelope['automation']['owner']}
        from gateway.session_automation import automation_display_metadata, automation_notification_metadata
        event.metadata.update(automation_notification_metadata(envelope['automation']))
        event.metadata.update(automation_display_metadata(envelope['automation']))
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
