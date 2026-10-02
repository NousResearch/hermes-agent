"""Proof-v2 output reads and exact durable dispositions for accepted peer Runs."""
import base64
import json

from aiohttp import web

from gateway.hosted_room_artifacts import RoomArtifactOutbox
from gateway.hosted_room_peer import HostedMemberDispatch, verify_room_grant
from gateway.hosted_room_peer_output import output_contract, output_manifest, output_scope
from gateway.platforms.api_server_room_grants import _room_grant_claims
from gateway.session_results import _RESULT_PREFIX
from hermes_state_runtime import _epoch

_RECEIPT = 'gateway.peer-output-disposition.v1.'


def _snapshot(adapter, request, conn, *, permission='status'):
    from gateway.session_authorities import active_authority
    from gateway.hosted_rooms import local_authority_gateway_id
    if not request.get('verified_room_grant'):
        raise ValueError('output requires proof-v2')
    authority = active_authority(adapter.gateway_runner)
    if authority is None:
        raise ValueError('output owner unavailable')
    _epoch(conn, authority.epoch)
    claims = _room_grant_claims(adapter, request, permission=permission, conn=conn)
    from gateway.platforms.api_server_room_grants import _local_target
    from gateway.platforms.api_server import _api_request_profile
    _local_target(claims, _api_request_profile)
    rows = conn.execute("SELECT * FROM session_admissions WHERE principal_id='api' AND request_id=?",
                        (request.match_info['run_id'],)).fetchall()
    if len(rows) != 1:
        raise ValueError('output Run unavailable')
    row = rows[0]
    payload = json.loads(row['payload_json'])['api_turn_v1']
    dispatch = HostedMemberDispatch.from_mapping(payload['settings']['room_dispatch'])
    if (output_contract(dispatch.document_output) is None or dispatch.target_install_id != local_authority_gateway_id()
            or verify_room_grant(adapter._room_grant_secret(), adapter._room_grant_token(request),
                                 dispatch, permission=permission) != claims):
        raise ValueError('output consent changed')
    from gateway.platforms.api_server_run_scope import room_run_scope_key
    if payload['run_owner_scope'] != room_run_scope_key(claims):
        raise ValueError('output owner scope changed')
    scope = output_scope(dispatch)
    if row['status'] not in {'terminal', 'unknown'}:
        raise ValueError('output producer is still active')
    raw = conn.execute('SELECT value FROM state_meta WHERE key=?', (_RESULT_PREFIX + row['admission_id'],)).fetchone()
    result = json.loads(raw[0])['result'] if raw else {}
    manifest = result.get('artifacts')
    if manifest is not None:
        output_manifest(manifest)
        if result.get('artifact_scope') != scope.as_mapping():
            raise ValueError('output receipt changed scope')
    success = row['status'] == 'terminal' and row['outcome'] == 'completed' and not (
        result.get('failed') or result.get('error') or result.get('interrupted'))
    resolution = conn.execute('SELECT value FROM state_meta WHERE key=?', (_RECEIPT + scope.key,)).fetchone()
    return authority, scope, manifest, success, json.loads(resolution[0]) if resolution else None


def _read_snapshot(adapter, request, permission):
    from gateway.session_authorities import active_authority
    authority = active_authority(adapter.gateway_runner)
    if authority is None:
        raise ValueError('output owner unavailable')
    with authority.db._read_ctx() as conn:
        return _snapshot(adapter, request, conn, permission=permission)


def _same(expected, current):
    if expected[:4] != current[:4]:
        raise ValueError('output changed during request')


async def handle(adapter, request):
    operation = request.match_info['operation']
    permission = 'status' if operation == 'read' else 'stop'
    try:
        if operation not in {'read', 'ack', 'discard'}:
            raise ValueError('unsupported output operation')
        body = await request.json()
        expected = _read_snapshot(adapter, request, permission)
        authority, scope, manifest, success, resolution = expected
        common = {'artifact_scope': scope.as_mapping(), 'manifest_digest': manifest['manifest_digest'] if manifest else None}
        if not isinstance(body, dict) or any(body.get(k) != value for k, value in common.items()):
            raise ValueError('output commitment changed')
        outbox = RoomArtifactOutbox(authority.db.db_path)
        if operation == 'read':
            if set(body) != {*common, 'artifact_id'} or not success or manifest is None or resolution is not None:
                raise ValueError('output unavailable')
            item = next((item for item in manifest['items'] if item['artifact_id'] == body['artifact_id']), None)
            if item is None:
                raise ValueError('output item unavailable')
            metadata, data = outbox.read(scope, item['artifact_id'])
            if metadata != item:
                raise ValueError('output bytes changed')
            after = _read_snapshot(adapter, request, permission)
            _same(expected, after)
            if after[4] is not None:
                raise ValueError('output retired during read')
            return web.json_response({'metadata': metadata, 'data_base64': base64.b64encode(data).decode()})
        if operation == 'ack':
            if not success or manifest is None:
                raise ValueError('only completed output may be published')
            commitment = {**common, 'artifact_ids': [item['artifact_id'] for item in manifest['items']],
                          'message_event_id': 'dmessage:' + scope.task_id.removeprefix('dtask:')}
        else:
            commitment = common
        if body != commitment:
            raise ValueError('output disposition changed')
        receipt = {'operation': operation, **commitment}
        def authorize(conn, checked_scope):
            if checked_scope != scope:
                raise ValueError('output scope changed')
            current = _snapshot(adapter, request, conn, permission=permission)
            _same(expected, current)
            if current[4] not in (None, receipt):
                raise ValueError('output already has another disposition')
            # The commitment and outbox retirement share the SAME accepting transaction.
            conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                         (_RECEIPT + scope.key, json.dumps(receipt, sort_keys=True)))
        if resolution not in (None, receipt):
            raise ValueError('output disposition conflicts')
        outbox.authorize_write = authorize
        if resolution == receipt and outbox.retirement_complete(scope):
            authority.db._execute_write(lambda conn: authorize(conn, scope))
            changed = 0
        elif operation == 'ack':
            changed = outbox.acknowledge(scope, body['artifact_ids'], message_event_id=body['message_event_id'])
        else:
            changed = outbox.discard_durably(scope)
        return web.json_response({'acknowledged': True, 'changed': changed} if operation == 'ack'
                                 else {'discarded': True, 'removed': changed})
    except (ValueError, KeyError, TypeError):
        return web.json_response({'error': {'code': 'peer_output_refused'}}, status=409)


def http_routes(adapter):
    from gateway.platforms.api_server_room_proof import wrap
    return [('POST', '/v1/runs/{run_id}/artifacts/{operation}',
             wrap(adapter, lambda request: handle(adapter, request), max_bytes=32 * 1024))]
