"""Root API output consent and producer, adapted from 018d3ea9 / a9577feb.

The signed request is transient. Only its immutable consent/owner commitment
enters the canonical payload; current authority is checked at every boundary.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import copy
import hashlib
import json
import os
import sqlite3
import time

from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox, RoomArtifactScope
from gateway.hosted_room_peer import HostedMemberDispatch, verify_room_grant
from gateway.session_peer_target import root_target, target_policy, grant_fence, require_current_grant
from hermes_state_runtime import RuntimeStoreError, _epoch, _json, _row

RIGHTS = ('artifact.read', 'artifact.ack')
_SCOPE_FIELDS = ('room_id', 'task_id', 'execution_generation', 'member_id', 'target_profile',
                 'home_install_id', 'target_install_id', 'authority_gateway_id', 'authority_epoch')
_OWNER_FIELDS = ('room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
                 'member_id', 'target_install_id', 'target_profile')


def _owner_identity(authority):
    return (authority, authority.db, authority.epoch, authority.instance_id,
            authority.runner.session_authorities, os.getpid())


def _require_outbox(adapter, authority, conn):
    from gateway.runtime_ownership import process_ownership
    if (authority.db._db_file_was_replaced()
            or not process_ownership.owns(Path(authority.profile_id))
            or getattr(adapter, '_peer_output_owner', None) != _owner_identity(authority)):
        raise RuntimeStoreError('room_output_unavailable')
    outbox = adapter._peer_output_outbox
    if (outbox.db_path.resolve() != Path(authority.db.db_path).resolve()
            or Path(conn.execute('PRAGMA database_list').fetchone()[2]).resolve() != outbox.db_path.resolve()
            or not outbox.blob_root.is_dir() or outbox.blob_root.is_symlink()):
        raise RuntimeStoreError('room_output_unavailable')
    # SELECT only: no constructors, migrations, reclamation, or readiness marker.
    conn.execute('SELECT scope_json, ack_message_event_id, blob_reclaimed_at, receipt_expires_at '
                 'FROM hosted_room_output_artifacts LIMIT 0')
    conn.execute('SELECT lineage_identity, max_generation, retired_generation '
                 'FROM hosted_room_output_generation_fences LIMIT 0')
    return outbox


def _registered_routes(adapter):
    from gateway.platforms.api_server_room_artifacts import (
        _handle_room_run_artifact, _handle_room_run_artifact_ack, _handle_room_run_artifact_discard)
    app = getattr(adapter, '_app', None)
    if app is None:
        return False
    expected = {('GET', '/v1/runs/{run_id}/artifacts/{artifact_id}'): _handle_room_run_artifact,
                ('POST', '/v1/runs/{run_id}/artifacts/ack'): _handle_room_run_artifact_ack,
                ('POST', '/v1/runs/{run_id}/artifacts/discard'): _handle_room_run_artifact_discard}
    found = {}
    for route in app.router.routes():
        key = (route.method, route.resource.canonical)
        if key in expected:
            handler = route.handler
            if getattr(handler, '__self__', None) is adapter and getattr(handler, '__func__', None) is expected[key]:
                found[key] = handler
    return found.keys() == expected.keys()


def peer_output_permissions(adapter, *, profile, catalog, connection):
    """Read-only real Output provider for the route owner's constrained seam."""
    try:
        authority, _, policy = target_policy(adapter, profile, connection=connection)
        admission = getattr(adapter, '_room_output_admission', None)
        if (not catalog.get('text') or catalog.get('execution_policy') != policy
                or not _registered_routes(adapter)
                or getattr(admission, '__self__', None) is not adapter
                or getattr(admission, '__func__', None) is not authorize_output_consent):
            return ()
        if connection is None:
            with authority.db._read_ctx() as conn:
                _require_outbox(adapter, authority, conn)
        else:
            _require_outbox(adapter, authority, connection)
        return RIGHTS
    except (RuntimeStoreError, sqlite3.Error, AttributeError, OSError):
        return ()


def initialize_peer_output(adapter):
    """Output setup after the root adapter is registered, before API recovery.

    No listener or worker is started here. Readiness never invokes this hook.
    """
    from gateway.runtime_ownership import process_ownership
    authority, _ = root_target(adapter)
    if not process_ownership.owns(Path(authority.profile_id)) or not _registered_routes(adapter):
        raise RuntimeStoreError('room_output_unavailable')
    adapter._peer_output_outbox = RoomArtifactOutbox(authority.db.db_path)
    adapter._peer_output_owner = _owner_identity(authority)
    adapter._room_output_admission = MethodType(authorize_output_consent, adapter)
    adapter._room_output_invitation_permissions = MethodType(peer_output_permissions, adapter)


@dataclass(frozen=True)
class OutputConsent:
    adapter: object
    owner: tuple
    record_json: str
    authorizer: object


def capture_output_consent(adapter, token, dispatch, policy, *, connection=None):
    claims = verify_room_grant(adapter._room_grant_secret(), token, dispatch, permission='dispatch')
    if not set(RIGHTS) <= set(claims['permissions']):
        return None
    authorizer = getattr(adapter, '_room_output_admission', None)
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    authority, _ = root_target(adapter, dispatch.target_profile, connection=connection)
    _, catalog = _local_room_catalog(adapter, dispatch.target_profile, dispatch.target_install_id, _connection=connection)
    provider = getattr(adapter, '_room_output_invitation_permissions', None)
    if (getattr(provider, '__self__', None) is not adapter
            or getattr(provider, '__func__', None) is not peer_output_permissions
            or provider(profile=dispatch.target_profile, catalog=catalog, connection=connection) != RIGHTS
            or getattr(adapter, '_room_output_admission', None) is not authorizer
            or catalog['execution_policy'] != policy):
        raise RuntimeStoreError('room_output_unavailable')
    for right in RIGHTS:
        verify_room_grant(adapter._room_grant_secret(), token, dispatch, permission=right)
    scope = RoomArtifactScope.from_mapping({key: getattr(dispatch, key) for key in _SCOPE_FIELDS})
    if scope.home_install_id == scope.target_install_id:
        raise RuntimeStoreError('permission_denied')
    owner_scope = hashlib.sha256('\0'.join(str(claims[k]) for k in _OWNER_FIELDS).encode()).hexdigest()
    record = dict(version=1, scope=scope.as_mapping(), claims=claims, dispatch=dispatch.as_mapping(),
                  policy=policy, profile_id=authority.profile_id, owner_epoch=authority.epoch,
                  instance_id=authority.instance_id, run_owner_scope=owner_scope)
    return OutputConsent(adapter, _owner_identity(authority), _json(record), authorizer)


def consent_record(evidence, adapter, authority, dispatch, owner_scope):
    if (type(evidence) is not OutputConsent or evidence.adapter is not adapter
            or evidence.owner != _owner_identity(authority)):
        raise RuntimeStoreError('permission_denied')
    record = json.loads(evidence.record_json)
    if record['dispatch'] != dispatch.as_mapping() or record['run_owner_scope'] != owner_scope:
        raise RuntimeStoreError('permission_denied')
    return record


def authorize_output_consent(adapter, authority, shared, conn, token, dispatch, policy, evidence):
    """Route's NEW-write consumer; reuse its held shared/owner grant fence."""
    current = capture_output_consent(adapter, token, dispatch, policy, connection=conn)
    if current != evidence:
        raise RuntimeStoreError('room_output_consent_changed')
    if current is not None:
        claims = json.loads(current.record_json)['claims']
        require_current_grant(shared, claims)
        require_current_grant(conn, claims)
    return current is not None


def admitted_peer_scope(adapter, authority, row, conn):
    data = row['payload'].get('api_turn_v1', {})
    consent = data.get('output_consent')
    if consent is None:
        return None
    current, _, policy = target_policy(adapter, connection=conn)
    _require_outbox(adapter, authority, conn)
    if current is not authority or not _registered_routes(adapter):
        raise RuntimeStoreError('room_output_unavailable')
    provider = getattr(adapter, '_room_output_invitation_permissions', None)
    if (getattr(provider, '__self__', None) is not adapter
            or getattr(provider, '__func__', None) is not peer_output_permissions):
        raise RuntimeStoreError('room_output_unavailable')
    dispatch = HostedMemberDispatch.from_mapping(data['settings']['room_dispatch'])
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    _, catalog = _local_room_catalog(adapter, dispatch.target_profile, dispatch.target_install_id, _connection=conn)
    # Accepted Output consent is independent of passive Files readiness. Only
    # reconstruct that bit; all other catalog fields, consent and owner checks
    # below remain exact. This imports no lower Files/selection implementation.
    from gateway.hosted_room_peer import _catalog_digest
    accepted_bit = not catalog['attachments']
    if not catalog['text'] or dispatch.capability_digest not in (
            catalog['catalog_digest'], _catalog_digest(dict(catalog, attachments=accepted_bit))):
        raise RuntimeStoreError('room_capability_catalog_changed')
    scope = RoomArtifactScope.from_mapping({k: getattr(dispatch, k) for k in _SCOPE_FIELDS})
    if (row['principal_id'] != 'api' or row['owner_epoch'] != authority.epoch
            or consent['version'] != 1 or consent['scope'] != scope.as_mapping()
            or consent['dispatch'] != dispatch.as_mapping() or consent['policy'] != policy
            or data['settings']['room_execution_policy'] != policy
            or consent['owner_epoch'] != authority.epoch or consent['instance_id'] != authority.instance_id
            or consent['profile_id'] != authority.profile_id or consent['run_owner_scope'] != data.get('run_owner_scope')
            or scope.home_install_id == scope.target_install_id
            or not set(RIGHTS) <= set(consent['claims']['permissions'])):
        raise RuntimeStoreError('permission_denied')
    stored = conn.execute('SELECT payload_digest FROM session_admissions WHERE admission_id=?',
                          (row['admission_id'],)).fetchone()
    if stored is None or stored[0] != output_run_binding(row)['payload_digest']:
        raise RuntimeStoreError('storage_unavailable')
    return scope


def peer_output_binding(authority, ref, row):
    if row['payload'].get('api_turn_v1', {}).get('output_consent') is None:
        return None
    row = copy.deepcopy(row)
    from gateway.config import Platform
    from gateway.session_hosted_output import HostedOutputBinding
    from gateway.session_managed_worker import managed_policy
    adapter = authority.runner.adapters.get(Platform.API_SERVER)
    if adapter is None or managed_policy(authority, ref) is not None:
        raise RuntimeStoreError('room_output_unavailable')
    with authority.db._read_ctx() as conn:
        scope = admitted_peer_scope(adapter, authority, row, conn)
    live = authority.sessions.get(ref.session_id)

    from gateway.session_admission import admission_fingerprint
    payload_digest = admission_fingerprint(canonical_target=ref.session_id,
        payload={'input': row['payload'], 'intent': row['intent']})

    class PeerOutputBinding(HostedOutputBinding):
        @contextmanager
        def fence(self):
            with grant_fence(adapter) as (owner, shared):
                if owner is not authority:
                    raise RoomArtifactError('Group Chat output owner changed')
                yield shared

        def check_write(self, conn, scope, *, shared=None):
            if not self.active or os.getpid() != self.owner_pid or scope != self.scope:
                raise RoomArtifactError('Group Chat output producer is no longer active')
            if (ref.profile_id != authority.profile_id or live is None
                    or authority.sessions.get(ref.session_id) is not live
                    or live.source.platform != Platform.API_SERVER
                    or authority.runner._adapter_for_source(live.source) is not adapter
                    or live.event_stream.execution != {
                        'authority_epoch': authority.epoch, 'execution_generation': row['generation'],
                        'admission_id': row['admission_id']}):
                raise RoomArtifactError('Group Chat output execution changed')
            _epoch(conn, authority.epoch)
            saved = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?',
                                 (row['admission_id'],)).fetchone()
            if (saved is None or saved['status'] != 'started' or saved['generation'] != row['generation']
                    or saved['target_session_id'] != ref.session_id or saved['request_id'] != row['request_id']
                    or saved['payload_digest'] != payload_digest or _row(saved)['payload'] != row['payload']
                    or admitted_peer_scope(adapter, authority, _row(saved), conn) != scope):
                raise RoomArtifactError('Group Chat output admission changed')
            claims = row['payload']['api_turn_v1']['output_consent']['claims']
            if time.time() >= claims['expires_at'] or shared is None:
                raise RoomArtifactError('Group Chat output consent expired')
            require_current_grant(shared, claims)
            require_current_grant(conn, claims)

        def outbox(self):
            if not self.active or os.getpid() != self.owner_pid:
                raise RoomArtifactError("Group Chat output producer is unavailable")
            self.used = True
            binding = self
            class ProducerOutbox(RoomArtifactOutbox):
                def put_bytes(self, **kwargs):
                    with binding.fence() as shared:
                        self.authorize_write = lambda conn, checked: binding.check_write(conn, checked, shared=shared)
                        return super().put_bytes(**kwargs)
            outbox = copy.copy(adapter._peer_output_outbox)
            outbox.__class__ = ProducerOutbox
            return outbox

    binding = PeerOutputBinding(authority, ref, copy.deepcopy(row), scope, 0, os.getpid())
    with binding.fence() as shared, authority.db._read_ctx() as conn:
        binding.check_write(conn, scope, shared=shared)
    return binding


def output_run_binding(row, result=None):
    from gateway.session_admission import admission_fingerprint
    return {**({"result_digest": hashlib.sha256(_json({k: v for k, v in result.items()
                if k != "peer_output_binding"}).encode()).hexdigest()} if result is not None else {}),
            **{k: row[k] for k in ("admission_id", "request_id", "target_session_id", "owner_epoch", "generation")},
            "payload_digest": admission_fingerprint(canonical_target=row["target_session_id"],
                payload={"input": row["payload"], "intent": row["intent"]})}


def canonical_peer_artifact_fields(adapter, authority, row, result, conn):
    if (row['status'] != 'terminal' or row['outcome'] != 'completed' or not result.get('artifacts')
            or row['payload'].get('api_turn_v1', {}).get('output_consent') is None):
        return {}
    if result.get('failed') or result.get('error') or result.get('interrupted'):
        return {}
    scope = admitted_peer_scope(adapter, authority, row, conn)
    if (scope is None or result.get('artifact_scope') != scope.as_mapping()
            or result.get('peer_output_binding') != output_run_binding(row, result)):
        raise RuntimeStoreError('storage_unavailable')
    from gateway.hosted_room_artifacts import validate_terminal_artifact_manifest
    validate_terminal_artifact_manifest(result['artifacts'])
    return {'room_artifact_scope': scope.as_mapping(), 'artifacts': result['artifacts']}
