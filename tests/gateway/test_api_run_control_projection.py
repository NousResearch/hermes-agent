"""HTTP controls use exact durable ownership without depending on peer output display."""
import asyncio
import copy
import json
import sqlite3
import urllib.error
import urllib.request

import pytest

from gateway.hosted_room_proof import request_proof, verify_response, RESPONSE_HEADER, RESPONSE_NONCE_HEADER
from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_peer_output_faults import pair
from tests.gateway.test_session_group_peer_daemons import _join_pair


def _wire(p, invitation, path, *, body=None, headers=None, anonymous=False):
    method = 'GET' if body is None else 'POST'
    headers = {'Content-Type': 'application/json', **(headers or {})}
    payload = b'' if body is None else json.dumps(body, separators=(',', ':')).encode()
    if not anonymous:
        if invitation is None:
            headers['Authorization'] = 'Bearer target-gateway-owned-secret'
        else:
            authorization, key, mac, payload = request_proof(invitation['grant'],
                installation_id=invitation['catalog']['installation_id'], method=method,
                path=path, body=payload, headers=headers)
            headers['Authorization'] = authorization
    request = urllib.request.Request(p.proxy.target + path, method=method,
        data=payload if body is not None else None, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read()
        if invitation is not None and not anonymous:
            assert response.headers.get(RESPONSE_HEADER), response.status
            raw = verify_response(key, mac, response.status, raw,
                response.headers[RESPONSE_HEADER], response.headers[RESPONSE_NONCE_HEADER])
        try:
            result = json.loads(raw)
        except ValueError:
            result = {'unstructured': raw.decode(errors='replace')}
        return response.status, result


def _row(p):
    with sqlite3.connect(p.peer / 'state.db') as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT * FROM session_admissions WHERE principal_id='api'").fetchall()
        assert len(rows) == 1
        return dict(rows[0])


def _payload(p, value):
    with sqlite3.connect(p.peer / 'state.db') as db:
        db.execute("UPDATE session_admissions SET payload_json=? WHERE principal_id='api'",
                   (value if isinstance(value, str) else json.dumps(value),))


async def _start(p, hd, pd, peer):
    if not peer:
        status, _ = await asyncio.to_thread(_wire, p, None, '/v1/runs',
            body={'input': 'HOLD_ACTIVE'}, headers={'Idempotency-Key': 'control-projection'})
        assert status == 202
        assert await asyncio.to_thread(p.pm.active_event.wait, 20)
        return None
    async with websocket(p.home, hd) as hw, websocket(p.peer, pd) as pw:
        _, invitation = await _join_pair(hw, pw, peer_first=True)
        sent = await rpc(hw, 'groups.send', room_id='linked', event_id='control-projection',
            payload={'text': '@reviewer Create and share the checklist.', 'thread_id': 'thread'})
        assert sent['result']['accepted']
        assert await asyncio.to_thread(p.pm.shared_event.wait, 20)
        assert p.pm.shared['ok']
        return invitation


def _replay(p, invitation, original):
    if invitation is None:
        body, key = {'input': 'HOLD_ACTIVE'}, 'control-projection'
    else:
        dispatch = json.loads(original['payload_json'])['api_turn_v1']['settings']['room_dispatch']
        body = {'input': dispatch['prompt'], 'hosted_room_dispatch': dispatch}
        key = f"room:{dispatch['task_id']}:{dispatch['execution_generation']}"
    return _wire(p, invitation, '/v1/runs', body=body, headers={'Idempotency-Key': key})


def _private_artifacts(p):
    with sqlite3.connect(p.peer / 'state.db') as db:
        rows = db.execute('SELECT artifact_id,blob_name,acknowledged_at,blob_reclaimed_at '
                          'FROM hosted_room_output_artifacts ORDER BY artifact_id').fetchall()
    blobs = p.peer / 'hosted-room-artifact-outbox' / 'blobs'
    return rows, [(blobs / row[1]).read_bytes() for row in rows]


def _grant(p, invitation, **changes):
    from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret, issue_room_grant
    from gateway.platforms.api_server_run_scope import ROOM_RUN_SCOPE_FIELDS
    secret = gateway_room_grant_secret(p.peer)
    claims = decode_room_grant(secret, invitation['grant'], permission='status')
    fields = {name: claims[name] for name in (*ROOM_RUN_SCOPE_FIELDS, 'execution_policy_digest')}
    token = issue_room_grant(secret, grant_id='control-boundary', **(fields | changes))
    return {**invitation, 'grant': token}


@pytest.mark.parametrize('peer', [True, False], ids=['proof', 'bearer'])
@pytest.mark.parametrize('http_receipt', [True, False], ids=['http-receipt', 'canonical-only'])
def test_stop_and_resolution_survive_optional_projection_failure(tmp_path, peer, http_receipt):
    with pair(tmp_path, '') as p, daemon(p.root, p.home, p.he, barrier=False) as (_, hd):
        with daemon(p.root, p.peer, p.pe, barrier=False) as (proc, pd):
            invitation = asyncio.run(_start(p, hd, pd, peer))
            original = _row(p)
            assert original['status'] == 'started'
            calls = len(p.pm.requests)
            artifacts = _private_artifacts(p) if peer else None
            corrupted = json.loads(original['payload_json'])
            corrupted['api_turn_v1']['settings']['room_dispatch'] = 'unreadable dispatch'
            _payload(p, corrupted)
            if not http_receipt:
                with sqlite3.connect(p.peer / 'runs_idempotency.db') as db:
                    db.execute('DELETE FROM run_idempotency')
            path = '/v1/runs/' + original['request_id']
            if peer:
                for fault in ('missing_task', 'prompt_mismatch'):
                    changed = json.loads(original['payload_json'])
                    dispatch = changed['api_turn_v1']['settings']['room_dispatch']
                    if fault == 'missing_task':
                        del dispatch['task_id']
                    else:
                        dispatch['prompt'] += ' changed'
                    _payload(p, changed)
                    status, result = _wire(p, invitation, path)
                    assert status == 503 and result['error']['code'] == 'peer_output_state_unavailable', (fault, status, result)
                _payload(p, corrupted)
            status, stopped = _wire(p, invitation, path + '/stop', body={})
            assert status == 200 and stopped['status'] == 'stopping', (status, stopped)
            assert stopped['admission_id'] == original['admission_id']
            assert _wire(p, invitation, path)[0:2] == (503, {'error': {'code': 'peer_output_state_unavailable'}})
            proc.kill(); proc.wait(timeout=10)
        p.pm.release.set(); p.pm.active_release.set()
        with daemon(p.root, p.peer, p.pe, barrier=False) as (_, pd):
            assert _row(p)['status'] == 'unknown'
            body = {'admission_id': original['admission_id'], 'execution_generation': original['generation']}
            if peer:
                for changes in ({'permissions': ['status']}, {'member_id': 'another'},
                                {'target_profile': 'foreign'}, {'authority_epoch': 999}):
                    grant = _grant(p, invitation, **changes)
                    for suffix, data in [('stop', {}), ('resolve-unknown', body)]:
                        status, result = _wire(p, grant, path + '/' + suffix, body=data)
                        assert status in {401, 403, 404}, (changes, status, result)
                    assert _row(p)['status'] == 'unknown'
            for changed, code in [({'admission_id': 'another'}, 'not_found'),
                                  ({'execution_generation': original['generation'] + 1}, 'stale_generation')]:
                status, result = _wire(p, invitation, path + '/resolve-unknown', body={**body, **changed})
                assert status == 409 and result['error']['code'] == code, (status, result)
            status, result = _wire(p, invitation, path + '/stop', body={})
            assert status == 409 and result['error']['code'] == 'unknown_execution', (status, result)
            for suffix, response in [('approval', {'request_id': 'absent', 'execution_generation': original['generation'], 'choice': 'deny'}),
                                     ('clarify', {'request_id': 'absent', 'execution_generation': original['generation'], 'answer': 'no'})]:
                status, result = _wire(p, invitation, path + '/' + suffix, body=response)
                assert status == 503 and result['error']['code'] == 'peer_output_state_unavailable', (status, result)
            status, result = _wire(p, invitation, path + '/steer', body={'text': 'no'})
            assert status == (401 if peer else 503), (status, result)
            # Ordinary Runs have no independent idempotency identity after deleting
            # their HTTP ledger; only document Runs can recover an absent receipt.
            if http_receipt or peer:
                status, result = _replay(p, invitation, original)
                expected = 'peer_output_state_unavailable' if http_receipt else 'admission_conflict'
                assert status == (503 if http_receipt else 409) and result['error']['code'] == expected, (status, result)
                assert not result.get('not_admitted')
            assert _wire(p, invitation, path + '/stop', body={}, anonymous=True)[0] == 401
            status, result = _wire(p, invitation, path + '/resolve-unknown', body=body)
            assert status == 200 and result['status'] == 'terminal' and result['outcome'] == 'interrupted', (status, result)
            assert _row(p)['status'] == 'terminal'
            assert len(p.pm.requests) == calls
            assert _wire(p, invitation, path)[0] == 503
            status, result = _wire(p, invitation, path + '/stop', body={})
            assert status == 200 and result['status'] == 'cancelled', (status, result)
            assert not any(key in result for key in ('artifacts', 'peer_output_empty', 'discarded', 'acknowledged'))
            if peer:
                assert _private_artifacts(p) == artifacts
                for operation in ('read', 'ack', 'discard'):
                    assert _wire(p, invitation, path + '/artifacts/' + operation, body={})[0] == 409
                assert _private_artifacts(p) == artifacts
                async def cannot_end():
                    async with websocket(p.home, hd) as hw:
                        ended = await rpc(hw, 'groups.disband', room_id='linked')
                        assert 'error' in ended, ended
                asyncio.run(cannot_end())
                assert _private_artifacts(p) == artifacts


@pytest.mark.parametrize('peer', [True, False], ids=['proof', 'bearer'])
def test_unreadable_core_ownership_never_authorizes_http_control(tmp_path, peer):
    with pair(tmp_path, '') as p, daemon(p.root, p.home, p.he, barrier=False) as (_, hd):
        with daemon(p.root, p.peer, p.pe, barrier=False) as (proc, pd):
            invitation = asyncio.run(_start(p, hd, pd, peer))
            original = _row(p)
            proc.kill(); proc.wait(timeout=10)
        p.pm.release.set(); p.pm.active_release.set()
        with daemon(p.root, p.peer, p.pe, barrier=False) as (_, pd):
            path = '/v1/runs/' + original['request_id']
            body = {'admission_id': original['admission_id'], 'execution_generation': original['generation']}
            valid = json.loads(original['payload_json'])
            for receipt in (True, False):
                if not receipt:
                    with sqlite3.connect(p.peer / 'runs_idempotency.db') as db:
                        db.execute('DELETE FROM run_idempotency')
                for fault in ('payload', 'missing_scope', 'wrong_scope'):
                    corrupted = copy.deepcopy(valid)
                    if fault == 'missing_scope':
                        del corrupted['api_turn_v1']['run_owner_scope']
                    elif fault == 'wrong_scope':
                        corrupted['api_turn_v1']['run_owner_scope'] = '0' * 64
                    _payload(p, '{invalid json' if fault == 'payload' else corrupted)
                    if fault == 'payload':
                        assert _wire(p, invitation, path) == (503, {'error': {'code': 'run_state_unavailable'}})
                        for suffix, response in [('approval', {'choice': 'deny'}), ('clarify', {'answer': 'no'})]:
                            assert _wire(p, invitation, path + '/' + suffix, body=response) == (503, {'error': {'code': 'run_state_unavailable'}})
                        if receipt or peer:
                            status, result = _replay(p, invitation, original)
                            expected = 'run_state_unavailable' if receipt else 'room_document_outcome_unknown'
                            assert status == 503 and result['error']['code'] == expected, (status, result)
                    for suffix, data in [('stop', {}), ('resolve-unknown', body)]:
                        status, result = _wire(p, invitation, path + '/' + suffix, body=data)
                        expected = 'run_state_unavailable' if fault == 'payload' else 'run_not_found'
                        assert status == (503 if fault == 'payload' else 404) and result['error']['code'] == expected, (fault, status, result)
                    assert _row(p)['status'] == 'unknown'
            # Native operator identity remains an independent exact recovery path.
            valid['api_turn_v1']['settings']['room_dispatch'] = 'unreadable dispatch'
            _payload(p, valid)
            binding_key = 'gateway.api.binding.v1.' + original['target_session_id']
            with sqlite3.connect(p.peer / 'state.db') as db:
                binding = db.execute('SELECT value FROM state_meta WHERE key=?', (binding_key,)).fetchone()[0]
                epoch = db.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()[0]
            for fault, code in [('missing_binding', 'run_not_found'), ('profile', 'profile_mismatch'),
                                ('session', 'admission_conflict'), ('epoch', 'stale_epoch')]:
                with sqlite3.connect(p.peer / 'state.db') as db:
                    changed = json.loads(binding)
                    if fault == 'missing_binding':
                        db.execute('DELETE FROM state_meta WHERE key=?', (binding_key,))
                    elif fault == 'epoch':
                        db.execute('UPDATE runtime_epoch SET epoch=? WHERE singleton=1', (epoch + 1,))
                    else:
                        changed['profile_id' if fault == 'profile' else 'session_id'] = 'foreign'
                        db.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(changed), binding_key))
                try:
                    for suffix, data in [('stop', {}), ('resolve-unknown', body)]:
                        status, result = _wire(p, invitation, path + '/' + suffix, body=data)
                        assert status == (404 if fault == 'missing_binding' else 409) and result['error']['code'] == code, (fault, status, result)
                    assert _row(p)['status'] == 'unknown'
                finally:
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        db.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)', (binding_key, binding))
                        db.execute('UPDATE runtime_epoch SET epoch=? WHERE singleton=1', (epoch,))
            async def native():
                async with websocket(p.peer, pd) as pw:
                    result = await rpc(pw, 'prompt.resolve_unknown', session_id=original['target_session_id'], **body)
                    assert result['result']['status'] == 'terminal' and result['result']['outcome'] == 'interrupted'
            asyncio.run(native())
            assert _row(p)['status'] == 'terminal'
