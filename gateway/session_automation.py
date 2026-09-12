"""Trusted producer admission; not a client-selectable internal input flag.

Reuse the private native route envelope and FIFO, retaining nonhuman turn semantics.
A producer ACK means committed input, not successful inference or outbound delivery.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError, admit_session_input


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
    sid = entry.origin.chat_id if entry.origin.platform == Platform.LOCAL else entry.session_id
    for row in list_session_admissions(authority.db, session_id=sid, pending_only=False):
        descriptor = row['payload'].get('local_automation_v1') or row['payload'].get('native_text_v1', {}).get('automation', {})
        if identity in descriptor.get('identities', [descriptor.get('identity')]):
            return row
    return None


def _owner(runner, event):
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
        from gateway.session_authorities import active_authority
        if active_authority(runner).db.get_compression_tip(expected) != entry.session_id:
            raise RuntimeStoreError('admission_conflict')
    return entry


def snapshot_automation(authority, adapter, event, identity):
    runner = authority.runner
    if (not event.internal or event.message_type != MessageType.TEXT or event.is_command()
            or not isinstance(event.text, str) or not identity
            or event.media_urls or event.prompt_response or event.source.platform == Platform.API_SERVER
            or set(event.metadata) - {'gateway_session_key', 'gateway_session_id', 'automation_identities', 'turn_author'}):
        raise RuntimeStoreError('invalid_params')
    entry = _owner(runner, event)
    if event.source.platform == Platform.LOCAL:
        return snapshot_local_automation(authority, adapter, event, identity, entry)
    from gateway.session_envelope import restore_native
    from hermes_state_runtime import list_session_admissions
    prior = list_session_admissions(authority.db, session_id=entry.session_id, pending_only=False)
    envelope = next((r['payload']['native_text_v1'] for r in reversed(prior)
                     if 'native_text_v1' in r['payload']), None)
    if envelope is None or 'provenance' not in envelope:
        raise RuntimeStoreError('not_found')
    # Persisted origins deliberately omit relay trust. Borrow only an exact
    # committed source's private proof, revalidated against the live connector.
    restored = restore_native({'text': event.text, 'native_text_v1': envelope}, runner)
    if (restored.source.to_dict() != event.source.to_dict()
            or runner._adapter_for_source(restored.source) is not adapter):
        raise RuntimeStoreError('admission_conflict')
    provenance = deepcopy(envelope['provenance'])
    source = deepcopy(envelope['source'])
    # Producer timestamps and platform reply IDs change on retry. Neither belongs
    # in the identity/fingerprint of the same immutable completion.
    envelope = {'source': source, 'route': entry.session_key,
        'timestamp': datetime.fromtimestamp(0, timezone.utc).isoformat(),
        'event': {'message_id': identity}, 'provenance': provenance,
        'automation': {'identity': identity, 'owner': entry.session_id}}
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
    entry = _owner(runner, event)
    if (entry.session_id != session_id or envelope['automation']['owner'] != session_id
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


def check_local_automation(authority, ref, row):
    live = authority.sessions[ref.session_id]
    payload = row['payload']
    descriptor = payload['local_automation_v1']
    if (set(payload) != {'text', 'local_automation_v1'}
            or row['principal_id'] != 'automation:' + live.route
            or descriptor['owner'] != ref.session_id or descriptor['route'] != live.route
            or descriptor['identity'] != row['request_id']):
        raise RuntimeStoreError('permission_denied')
    entry = authority.runner.session_store.lookup_by_session_key(live.route)
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
    if descriptor.get('turn_author') is not None:
        event.metadata['turn_author'] = deepcopy(descriptor['turn_author'])
    if descriptor.get('heartbeat'):
        event._heartbeat_session_id = descriptor['heartbeat']
    return event


async def admit_automation(authority, adapter, event, identity):
    authority._require_admission_open()
    payload, entry = snapshot_automation(authority, adapter, event, identity)
    from gateway.session_authority import LiveSession
    sid = payload.get('local_automation_v1', {}).get('owner', entry.session_id)
    ref = SessionRef(authority.profile_id, sid)
    authority.sessions.setdefault(ref.session_id, LiveSession(event.source, entry.session_key))
    row = admit_session_input(authority.db, epoch=authority.epoch, principal_id='automation:' + entry.session_key,
        session_id=ref.session_id, request_id=identity, payload=deepcopy(payload))
    event._gateway_accepted = True
    authority._publish_pending(ref)
    authority._schedule(ref)
    return authority._receipt(row)
