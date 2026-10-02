"""Peer output producer bound to one accepted canonical API admission."""
import hashlib
import json
import os
from pathlib import Path

from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox
from gateway.hosted_room_peer_output import OUTPUT_CAPABILITY, output_contract, output_scope


def output_available(adapter):
    from gateway.session_authorities import active_authority
    return dict(OUTPUT_CAPABILITY) if active_authority(getattr(adapter, 'gateway_runner', None)) is not None else None


def check_binding(binding, conn, scope):
    from gateway.hosted_room_peer import HostedMemberDispatch
    from gateway.session_api import _BINDING_PREFIX
    dispatch = HostedMemberDispatch.from_mapping(json.loads(binding.peer_dispatch_json))
    if output_contract(dispatch.document_output) is None or output_scope(dispatch) != scope or binding.row['principal_id'] != 'api':
        raise RoomArtifactError('Peer output consent changed')
    saved = conn.execute('SELECT payload_json FROM session_admissions WHERE admission_id=?',
                         (binding.row['admission_id'],)).fetchone()
    if (saved is None or json.loads(saved[0]).get('api_turn_v1', {}).get('settings', {}).get('room_dispatch') != dispatch.as_mapping()):
        raise RoomArtifactError('Peer output admission changed')
    from gateway.platforms.api_server_run_scope import room_run_scope_key
    if json.loads(saved[0])['api_turn_v1'].get('run_owner_scope') != room_run_scope_key(dispatch.as_mapping()):
        raise RoomArtifactError('Peer output admission has no authenticated room owner')
    record = conn.execute('SELECT value FROM state_meta WHERE key=?', (_BINDING_PREFIX + binding.ref.session_id,)).fetchone()
    if (record is None or json.loads(record[0]).get('room_identity') != [dispatch.home_install_id, dispatch.room_id,
                                                                    dispatch.member_id, dispatch.target_profile]):
        raise RoomArtifactError('Peer output session owner changed')


def peer_binding(authority, ref, row):
    from gateway import hosted_rooms
    from gateway.config import Platform
    from gateway.hosted_room_peer import HostedMemberDispatch
    from gateway.session_hosted_output import HostedOutputBinding
    raw = row['payload'].get('api_turn_v1', {}).get('settings', {}).get('room_dispatch')
    if not isinstance(raw, dict) or raw.get('document_output') is None:
        return None
    dispatch = HostedMemberDispatch.from_mapping(raw)
    live = authority.sessions.get(ref.session_id)
    if (row['principal_id'] != 'api' or live is None or live.source.platform != Platform.API_SERVER
            or dispatch.target_install_id != hosted_rooms.local_authority_gateway_id()
            or Path(authority.db.db_path).resolve().parent != Path(authority.profile_id).resolve()):
        raise RoomArtifactError('Peer output owner changed')
    binding = HostedOutputBinding(authority, ref, row, output_scope(dispatch), None, os.getpid(),
                                  peer_dispatch_json=json.dumps(dispatch.as_mapping(), sort_keys=True))
    with authority.db._read_ctx() as conn:
        binding.check_write(conn, binding.scope)
    return binding


class PeerDocumentOutbox(RoomArtifactOutbox):
    """Apply the negotiated document budget inside the ordinary outbox writer."""
    def put_bytes(self, *, scope, data, source_name, name=None):
        safe_name = self._safe_name(name or source_name)
        kind, mime = self._classify(safe_name, data)
        limits = OUTPUT_CAPABILITY
        if (kind not in limits['kinds'] or mime.startswith(('image/', 'audio/', 'video/'))
                or not 0 < len(data) <= limits['max_file_bytes']):
            raise RoomArtifactError('Peer sharing accepts file/PDF documents up to 5 MB each')
        authorize = self.authorize_write
        digest = hashlib.sha256(data).hexdigest()
        def bounded(conn, checked):
            if authorize is not None:
                authorize(conn, checked)
            old = conn.execute('SELECT 1 FROM hosted_room_output_artifacts WHERE scope_key=? AND sha256=? AND name=?',
                               (scope.key, digest, safe_name)).fetchone()
            count, size = conn.execute('SELECT COUNT(*),COALESCE(SUM(size),0) FROM hosted_room_output_artifacts WHERE scope_key=? AND acknowledged_at IS NULL',
                                       (scope.key,)).fetchone()
            if not old and (count >= limits['max_count'] or size + len(data) > limits['max_batch_bytes']):
                raise RoomArtifactError('Peer document sharing exceeds its 8-file / 6 MB turn budget')
        self.authorize_write = bounded
        try:
            return super().put_bytes(scope=scope, data=data, source_name=source_name, name=name)
        finally:
            self.authorize_write = authorize


def receipt_fields(row, result):
    from gateway.hosted_room_peer import HostedMemberDispatch
    from gateway.platforms.api_server_run_scope import room_run_scope_key
    from gateway.session_hosted_output import terminal_output_fields
    try:
        payload = row['payload'].get('api_turn_v1') or {}
        raw = (payload.get('settings') or {}).get('room_dispatch')
        if raw is None:
            return {}
        dispatch = HostedMemberDispatch.from_mapping(raw)
        if row['principal_id'] != 'api' or payload.get('run_owner_scope') != room_run_scope_key(dispatch.as_mapping()):
            if row['status'] == 'unknown':
                raise ValueError('unknown peer dispatch owner changed')
            return {}
        if dispatch.document_output is None:
            return {}
        if row['status'] == 'unknown':
            return {'peer_output_unresolved': output_scope(dispatch).as_mapping()}
        fields = terminal_output_fields(result)
        if fields and fields.get('artifact_scope', fields.get('peer_output_empty')) != output_scope(dispatch).as_mapping():
            return {}
        return fields
    except (ValueError, TypeError, KeyError) as error:
        raise PeerOutputStateUnavailable('canonical peer output state is unreadable') from error


def accepted_dispatch_digest(row):
    """Only a canonical authenticated admission can prove a remote output attempt."""
    from gateway.hosted_room_peer import HostedMemberDispatch
    from gateway.hosted_room_peer_output import dispatch_digest
    from gateway.platforms.api_server_run_scope import room_run_scope_key
    try:
        payload = row['payload']['api_turn_v1']
        dispatch = HostedMemberDispatch.from_mapping(payload['settings']['room_dispatch'])
        if (row['principal_id'] == 'api' and output_contract(dispatch.document_output) is not None
                and payload.get('run_owner_scope') == room_run_scope_key(dispatch.as_mapping())):
            return dispatch_digest(dispatch)
    except (ValueError, TypeError, KeyError):
        pass
    return None


class PeerOutputStateUnavailable(ValueError):
    """Canonical evidence cannot safely be projected as a terminal Run."""


def dispatch_evidence(row):
    """Opt-in read proof, including explicit null consent for legacy text Runs."""
    from gateway.hosted_room_peer import HostedMemberDispatch
    from gateway.hosted_room_peer_output import dispatch_digest
    from gateway.platforms.api_server_run_scope import room_run_scope_key
    try:
        payload = row['payload']['api_turn_v1']
        raw = payload['settings']['room_dispatch']
        if row['principal_id'] != 'api' or raw is None:
            return None
        dispatch = HostedMemberDispatch.from_mapping(raw)
        if payload.get('run_owner_scope') != room_run_scope_key(dispatch.as_mapping()):
            return None
        return {'dispatch_digest': dispatch_digest(dispatch), 'document_output': dispatch.document_output}
    except (ValueError, TypeError, KeyError) as error:
        raise PeerOutputStateUnavailable('canonical peer dispatch is unreadable') from error
