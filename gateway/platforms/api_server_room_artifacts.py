"""Exact root Run read/ACK; selected 018d3ea9 / 00b9d2f7 contract.

All writer predicates use the shared-first held connections. The target checks
Home's immutable commitment, not a fictitious view of Home's journal.
"""
import copy
import hashlib
import json
import sqlite3
from types import MethodType

from aiohttp import web
from gateway.hosted_room_artifacts import RoomArtifactScope, validate_terminal_artifact_manifest
from gateway.session_peer_target import root_target, grant_fence, require_current_grant
from gateway.platforms.api_server_authority_runs import run_admission, run_projection
from gateway.session_results import _RESULT_PREFIX
from hermes_state_runtime import _json


class SharedGrantUnavailable(RuntimeError):
    """A foreign grant-store statement failed, not the owning SessionDB."""


def _http_routes(adapter):
    return [('GET', '/v1/runs/{run_id}/artifacts/{artifact_id}', MethodType(_handle_room_run_artifact, adapter)),
            ('POST', '/v1/runs/{run_id}/artifacts/ack', MethodType(_handle_room_run_artifact_ack, adapter)),
            ('POST', '/v1/runs/{run_id}/artifacts/discard', MethodType(_handle_room_run_artifact_discard, adapter))]


def _load_scope_and_status(adapter, request, *, permission, authority, shared, conn):
    from gateway.platforms.api_server_room_grants import _decode_request_grant, _local_target
    from gateway.platforms.api_server import _api_request_profile
    claims = _decode_request_grant(adapter, request, permission=permission)
    _local_target(claims, _api_request_profile)
    owner, _ = root_target(adapter, claims['target_profile'], connection=conn)
    if owner is not authority:
        raise ValueError('output owner changed')
    try:
        require_current_grant(shared, claims)
    except sqlite3.Error as exc:
        # Only the foreign statement crosses this boundary. Owner-origin SQL
        # must still reach its own sticky corruption classifier unchanged.
        raise SharedGrantUnavailable('shared Output grant storage unavailable') from exc
    require_current_grant(conn, claims)
    run_id = str(request.match_info['run_id'])
    owned = run_admission(adapter, run_id, connection=conn)
    if owned is None or owned[0] is not authority:
        raise ValueError('run not found')
    row = owned[1]
    from gateway.hosted_room_peer import HostedMemberDispatch, verify_room_grant
    dispatch = HostedMemberDispatch.from_mapping(row['payload']['api_turn_v1']['settings']['room_dispatch'])
    if verify_room_grant(adapter._room_grant_secret(), adapter._room_grant_token(request),
                         dispatch, permission=permission) != claims:
        raise ValueError('artifact grant policy changed')
    status = run_projection(adapter, run_id, connection=conn)
    if status['status'] != 'completed':
        raise ValueError('run not completed')
    scope = RoomArtifactScope.from_mapping(status.get('room_artifact_scope') or {})
    keys = ('room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
            'member_id', 'target_install_id', 'target_profile')
    if any(claims[k] != getattr(scope, k) for k in keys):
        raise ValueError('run scope changed')
    owner_scope = hashlib.sha256('\0'.join(str(claims[k]) for k in keys).encode()).hexdigest()
    if row['payload']['api_turn_v1'].get('run_owner_scope') != owner_scope:
        raise ValueError('run owner changed')
    manifest = validate_terminal_artifact_manifest(status.get('artifacts'))
    if not manifest:
        raise ValueError('output unavailable')
    saved = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?',
                                    (_RESULT_PREFIX + row['admission_id'],)).fetchone()[0])
    return dict(run_id=run_id, scope=scope.as_mapping(), row=row, result=saved['result'],
                manifest=status['artifacts'], ack=saved.get('peer_output_ack'),
                discard=saved.get('peer_output_discard'))


def _snapshot(adapter, request, permission):
    with grant_fence(adapter) as (authority, shared):
        value = authority.db._execute_write(lambda conn: _load_scope_and_status(adapter, request,
            permission=permission, authority=authority, shared=shared, conn=conn))
    return authority, value


def _same(expected, actual):
    # ACK metadata is additive in the same retained Run result owner.
    if {k: v for k, v in actual.items() if k not in ('ack', 'discard')} != {k: v for k, v in expected.items() if k not in ('ack', 'discard')}:
        raise ValueError('artifact Run commitment changed')


async def _handle_room_run_artifact(adapter, request):
    try:
        authority, expected = _snapshot(adapter, request, 'artifact.read')
        scope = RoomArtifactScope.from_mapping(expected['scope'])
        artifact_id = str(request.match_info['artifact_id'])
        item = next((x for x in expected['manifest']['items'] if x['artifact_id'] == artifact_id), None)
        if item is None or expected['discard'] is not None:
            raise ValueError('artifact not found')
        metadata, data = adapter._peer_output_outbox.read(scope, artifact_id)
        if metadata != item or len(data) != item['size'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('artifact changed')
        # Byte I/O is not a grant. Recheck both stores, owner, Run and source row.
        with grant_fence(adapter) as (current, shared):
            if current is not authority:
                raise ValueError('owner changed')
            def after_read(conn):
                actual = _load_scope_and_status(adapter, request, permission='artifact.read',
                    authority=authority, shared=shared, conn=conn)
                _same(expected, actual)
                if actual['discard'] is not None:
                    raise ValueError('artifact retired')
                row = conn.execute('SELECT * FROM hosted_room_output_artifacts '
                    'WHERE scope_key=? AND artifact_id=? AND acknowledged_at IS NULL', (scope.key, artifact_id)).fetchone()
                if row is None or adapter._peer_output_outbox._manifest(row) != item:
                    raise ValueError('artifact changed')
            authority.db._execute_write(after_read)
        return web.Response(body=data, content_type=item['mime'],
            headers={'X-Hermes-Artifact-SHA256': item['sha256'], 'Cache-Control': 'no-store'})
    except Exception:
        from gateway.platforms.api_server import _openai_error
        return web.json_response(_openai_error('Artifact not found.', code='artifact_not_found'), status=404)


async def _handle_room_run_artifact_ack(adapter, request):
    try:
        authority, expected = _snapshot(adapter, request, 'artifact.ack')
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {'artifact_ids', 'manifest_digest', 'message_event_id'}:
            raise ValueError('invalid acknowledgement')
        scope = RoomArtifactScope.from_mapping(expected['scope'])
        commitment = dict(artifact_ids=[x['artifact_id'] for x in expected['manifest']['items']],
            manifest_digest=expected['manifest']['manifest_digest'],
            message_event_id='dmessage:' + scope.task_id.removeprefix('dtask:'))
        if body != commitment:
            raise ValueError('acknowledgement commitment changed')
        with grant_fence(adapter) as (owner, shared):
            if owner is not authority:
                raise ValueError('owner changed')
            def authorize(conn, checked_scope, *, retirement=False):
                if checked_scope != scope:
                    raise ValueError('scope changed')
                actual = _load_scope_and_status(adapter, request, permission='artifact.ack',
                    authority=authority, shared=shared, conn=conn)
                _same(expected, actual)
                if (actual['discard'] is not None or actual['ack'] not in (None, commitment)
                        or (retirement and actual['ack'] != commitment)):
                    raise ValueError('retirement has no exact ACK commitment')
                return actual

            def commit_ack(conn, checked_scope):
                authorize(conn, checked_scope)
                key = _RESULT_PREFIX + expected['row']['admission_id']
                saved = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
                saved['peer_output_ack'] = commitment
                conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(saved), key))

            # Reauthorize after the body await, before even considering replay.
            current = authority.db._execute_write(lambda conn: authorize(conn, scope))
            outbox = copy.copy(adapter._peer_output_outbox)
            outbox.authorize_write = lambda conn, checked: authorize(conn, checked, retirement=True)
            retired = current['ack'] == commitment and outbox.retirement_complete(scope)
            if retired:
                changed = 0
            else:
                outbox.authorize_write = authorize
                outbox.commit_acknowledgement = commit_ack
                changed = outbox.acknowledge(scope, body['artifact_ids'], message_event_id=body['message_event_id'])
        return web.json_response({'acknowledged': True, 'changed': changed})
    except Exception:
        from gateway.platforms.api_server import _openai_error
        return web.json_response(_openai_error('Artifact acknowledgement was rejected.', code='invalid_artifact_ack'), status=409)


async def _handle_room_run_artifact_discard(adapter, request):
    """Historical artifact.ack retirement, bound to the canonical Run result."""
    from gateway.hosted_room_output_discard import (
        retire_exact, cleanup_exact, require_retired, require_record, OutputCleanupUnavailable)
    from tui_gateway.hosted_room_peer_http import peer_result_digest
    from gateway.platforms.api_server import _openai_error
    from gateway.session_peer_output import _require_outbox
    try:
        authority, expected = _snapshot(adapter, request, 'artifact.ack')
        # Never suspend while holding either shared or owning writer.
        body = await request.json()
        if (type(body) is not dict or set(body) != {'reason', 'result_digest'}
                or body['reason'] != 'verification_failed' or type(body['result_digest']) is not str):
            raise ValueError('invalid retirement commitment')
        scope = RoomArtifactScope.from_mapping(expected['scope'])
        with grant_fence(adapter) as (owner, shared):
            if owner is not authority:
                raise ValueError('output owner changed')

            def authorize(conn):
                actual = _load_scope_and_status(adapter, request, permission='artifact.ack',
                    authority=authority, shared=shared, conn=conn)
                _same(expected, actual)
                if (actual['ack'] is not None or body['result_digest'] != peer_result_digest(
                        run_projection(adapter, actual['run_id'], connection=conn))):
                    raise ValueError('output retirement commitment changed')
                return actual, _require_outbox(adapter, authority, conn)

            def retire(conn):
                actual, outbox = authorize(conn)
                record = actual['discard']
                if record is None:
                    blobs = retire_exact(outbox, conn, scope, expected['manifest']['items'], authorize=authorize)
                    record = dict(commitment=body, receipt=dict(discarded=True, removed=len(blobs)),
                                  state='pending', blobs=blobs)
                    key = _RESULT_PREFIX + expected['row']['admission_id']
                    saved = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
                    saved['peer_output_discard'] = record
                    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(saved), key))
                require_record(record, body, expected['manifest']['items'])
                if record['state'] == 'pending':
                    require_retired(conn, scope)
                return record

            record = authority.db._execute_write(retire)
            # Fence, exact receipt and cleanup intent are now durable. A failure
            # below is NOT completed-zero: retry the same authorized commitment.
            try:
                def cleanup(conn):
                    actual, outbox = authorize(conn)
                    if actual['discard'] != record:
                        raise ValueError('retirement receipt changed during cleanup')
                    if record['state'] == 'completed':
                        return  # Exact replay needs neither retired rows nor bytes.
                    cleanup_exact(outbox, conn, scope, expected['manifest']['items'],
                                  record['blobs'], authorize=authorize)
                    key = _RESULT_PREFIX + expected['row']['admission_id']
                    saved = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
                    saved['peer_output_discard'] = dict(record, state='completed', blobs=[])
                    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(saved), key))
                authority.db._execute_write(cleanup)
            except Exception as exc:
                # Let the shared fence roll back too; the first owner commit
                # remains a durable pending intent for an authenticated retry.
                raise OutputCleanupUnavailable('Output cleanup is unavailable') from exc
        return web.json_response(record['receipt'])
    except (sqlite3.Error, OSError, SharedGrantUnavailable, OutputCleanupUnavailable):
        return web.json_response(_openai_error('Artifact retirement storage is unavailable.',
            code='artifact_retirement_unavailable'), status=503, headers={'Retry-After': '1'})
    except Exception:
        return web.json_response(_openai_error('Artifact retirement was rejected.',
            code='invalid_artifact_retirement'), status=409)
