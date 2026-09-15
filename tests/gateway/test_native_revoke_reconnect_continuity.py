"""Delayed reconnect cleanup cannot pin the real Files driver to a retired attempt.

The Home service is inert: no start/lifespan/listener/model execution. Real
registered RPC/HTTP handlers, grant stores, route CAS, driver and private bytes
are used. Only the wire and post-admission observation/execution are substituted.
Desktop's matching/conflict decision is source-traced, not reimplemented here.
"""
import asyncio
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from gateway import hosted_room_driver as state, hosted_room_links, hosted_rooms
from gateway.hosted_room_peer import decode_room_grant
from gateway.platforms import api_server_room_attachments as attachments_api
from gateway.platforms import api_server_runs
from tests.gateway.test_canonical_peer_files_target import files_target, inprocess_http  # noqa: F401
from tests.gateway.test_canonical_peer_target_setup import invite, invitation, target  # noqa: F401
from tui_gateway import hosted_room_peer_http, methods_groups
from tui_gateway.hosted_room_service import HostedRoomService


class SimulatedSourceExit(BaseException):
    """Abrupt source loss at the wire; do not run ordinary exception recovery."""


@pytest.mark.linux_only
@pytest.mark.asyncio
@pytest.mark.parametrize('replacement', [False, True, 'scope'], ids=[
    'matching-lost-response', 'delayed-conflict-cleanup', 'scope-cleanup'])
@pytest.mark.parametrize('source_exit', [False, True], ids=['ordinary-retry', 'source-recovery'])
async def test_real_driver_continues_after_delayed_reconnect(files_target, monkeypatch, replacement, source_exit):
    import tui_gateway.server as server

    target = files_target
    source_home = target.home / 'source'
    source_home.mkdir()
    service = HostedRoomService(server, db_path=source_home / 'state.db')
    assert service.runtime.status()['running'] is False
    # The normal resolver gets the already-created inert service, never autostart.
    monkeypatch.setattr(server, 'get_hosted_room_service', lambda: service)
    monkeypatch.setattr(methods_groups, 'get_hosted_room_service', lambda: service)
    installation = hosted_rooms.local_authority_gateway_id()
    invitation_scope = invitation() | {
        'home_install_id': installation, 'authority_gateway_id': installation,
        'ttl_seconds': 3600, 'status_ttl_seconds': 7200,
    }
    predecessor = await invite(target, invitation_scope | {'request_id': 'invite-predecessor'})
    issued = await invite(target, invitation_scope)
    service.create_room(room_id='room-one', name='Reconnect continuity', members=[{
        'member_id': 'member-one', 'profile': 'default', 'handle': 'one',
        'target': {'kind': 'peer', 'peer_id': installation, 'installation_id': installation,
                   'profile': 'default', 'capability_digest': issued['catalog']['catalog_digest']},
    }, {'member_id': 'home-observer', 'profile': 'default', 'handle': 'observer'}])
    responses = []
    calls = inprocess_http(target, monkeypatch, responses=responses)
    wire_open = hosted_room_peer_http._open_roomlink_url
    waiting, release = threading.Event(), threading.Event()
    uploads, manifests, recovery_states = [], [], []
    held = False

    def interrupted_wire(wire, **kwargs):
        nonlocal held
        method, url = wire.get_method(), wire.full_url
        if method == 'POST' and url.endswith('/attachments'):
            manifests.append(json.loads(wire.data)['hosted_room_dispatch'])
            if source_exit and len(manifests) > 1:
                recovering = state.get_task(service.db_path, identity)
                recovery_states.append((recovering['status'], recovering['execution_generation']))
        if method == 'PUT':
            uploads.append((url, wire.get_header('Authorization')))
            if len(uploads) == 2 and not held:
                held = True
                waiting.set()
                assert release.wait(10), 'test did not release the second upload'
                if source_exit:
                    raise SimulatedSourceExit()
        return wire_open(wire, **kwargs)

    monkeypatch.setattr(hosted_room_peer_http, '_open_roomlink_url', interrupted_wire)

    def registration(token, expected):
        return dict(room_id='room-one', member_id='member-one',
                    target_profile='default', target_url='http://127.0.0.1:8642',
                    grant=token['grant'], catalog=token['catalog'],
                    cancellation_scope_id='cancel-one', trace_id='trace-one',
                    expected_grant_sha256=expected)

    register = server._methods['groups.peer.register']
    initial = await asyncio.to_thread(register, 0, registration(predecessor, ''))
    assert initial.get('result', {}).get('registered') is True, initial
    predecessor_sha = hashlib.sha256(predecessor['grant'].encode()).hexdigest()
    first_params = registration(issued, predecessor_sha)
    result = await asyncio.to_thread(register, 1, first_params)
    assert result.get('result', {}).get('registered') is True, result
    initial_retirements = sum(path.endswith('/grants/revoke-exact') for _, path in calls)
    assert initial_retirements == 1
    original_sha = hashlib.sha256(issued['grant'].encode()).hexdigest()
    raw = [b'first unfinished document', b'second pending document']
    uploaded = [service.attachments.put(room_id='room-one', upload_id=f'upload-{i}',
        name=f'document-{i}.txt', kind='file', mime='text/plain', data=data)
        for i, data in enumerate(raw)]
    refs = [{key: item[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')}
            for item in uploaded]
    service.send(room_id='room-one', event_id='event-one', payload={
        'text': '@one Read both documents', 'thread_id': 'thread-one', 'attachments': refs,
    })
    queued = state.list_tasks(service.db_path, room_id='room-one', status='queued')
    assert len(queued) == 1 and queued[0]['execution_generation'] == 0
    identity, payload = queued[0]['identity'], queued[0]['payload']
    admitted = []

    async def inert_execution(adapter, launch, **kwargs):
        admitted.append(launch.admission)

    monkeypatch.setattr(api_server_runs, '_execute_run', inert_execution)
    # Do not start an executor or poll forever after a real durable admission.
    monkeypatch.setattr(service.runtime, '_wait_for_terminal', lambda *args, **kwargs: None)
    binding = service.bindings()[0]
    running = asyncio.create_task(asyncio.to_thread(service.runtime._run_room_once, binding))
    try:
        assert await asyncio.to_thread(waiting.wait, 10), service.runtime.status()
        assert manifests[0]['execution_generation'] == 1
        spool = attachments_api._default_spool()
        key = attachments_api._batch_key(
            hosted_room_peer_http.HostedMemberDispatch.from_mapping(manifests[0]))
        file_paths = [spool._file_path(key, item['attachment_id']) for item in uploaded]
        assert file_paths[0].read_bytes() == raw[0] and not file_paths[1].exists()
        assert not admitted
        with sqlite3.connect(spool.db_path) as conn:
            before_fences = conn.execute('SELECT * FROM roomlink_attachment_attempt_fences').fetchall()
            origin, ambiguous = conn.execute(
                'SELECT staging_origin_json, staging_origin_ambiguous FROM roomlink_attachment_batches').fetchone()
        assert json.loads(origin)['_token_sha256'] == original_sha and ambiguous == 0

        current = issued
        reclaimed = False
        if replacement:
            if replacement == 'scope':
                # Scope retirement removes the old reservation. A real later
                # invitation must rebind it, issued after BOTH durable cutoffs.
                old_claims = decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='status')
                cutoff = old_claims['issued_at'] + 0.5
                monkeypatch.setattr(hosted_rooms.time, 'time', lambda: cutoff)
                scope_denied = await target.connection.dispatch(dict(id=3, method='groups.peer.revoke',
                                                                     params={'grant': issued['grant']}))
                assert scope_denied.get('result') == {'revoked': True, 'exact': False}, scope_denied
                for name in ('state.db', 'shared-state.db'):
                    with sqlite3.connect(target.home / name) as conn:
                        assert conn.execute('SELECT revoked_before FROM hosted_room_revoked_grants').fetchone()[0] == cutoff
                monkeypatch.setattr(hosted_rooms.time, 'time', lambda: old_claims['issued_at'] + 1)
            # A different Desktop reconnect has its own native invitation receipt.
            current = await invite(target, invitation() | {
                'request_id': 'invite-replacement', 'home_install_id': installation,
                'authority_gateway_id': installation, 'ttl_seconds': 3600,
                'status_ttl_seconds': 7200,
            })
            assert current['grant'] != issued['grant']
            replaced = await asyncio.to_thread(register, 2, registration(current, original_sha))
            assert replaced.get('result', {}).get('registered') is True, replaced
            # Normal Home retirement is HTTP-only and leaves unfinished staging.
            assert sum(path.endswith('/grants/revoke-exact') for _, path in calls) == initial_retirements + 1
            if replacement != 'scope':
                assert file_paths[0].read_bytes() == raw[0]
            fingerprint = service.status_with_grant_fingerprints('room-one')['peer_routes'][0]['grant_sha256']
            assert fingerprint != original_sha
            # This is the real native edge eligible after Desktop observes conflict.
            if replacement != 'scope':
                denied = await target.connection.dispatch(dict(id=3, method='groups.peer.revoke_exact',
                                                               params={'grant': issued['grant']}))
                assert denied.get('result') == {'revoked': True, 'exact': True}, denied
            old_claims = decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='status')
            assert all(hosted_rooms.room_grant_is_revoked(target.home / name, claims=old_claims)
                       for name in ('state.db', 'shared-state.db'))
            reclaimed = not any(path.exists() for path in file_paths)
            if not source_exit:
                assert reclaimed
            with sqlite3.connect(spool.db_path) as conn:
                if reclaimed:
                    assert conn.execute('SELECT complete, staging_origin_json, staging_origin_ambiguous '
                                        'FROM roomlink_attachment_batches').fetchall() == [(0, origin, 0)]
                    assert conn.execute('SELECT stored FROM roomlink_attachment_files').fetchall() == [(0,), (0,)]
                assert conn.execute('SELECT * FROM roomlink_attachment_attempt_fences').fetchall() == before_fences
        else:
            # The first successful registration reply was lost to Desktop. The
            # matching route settles by retrying that exact registration, not revoke.
            recovered = await asyncio.to_thread(register, 2, first_params)
            assert recovered.get('result', {}).get('registered') is True, recovered
            assert service.status_with_grant_fingerprints('room-one')['peer_routes'][0]['grant_sha256'] == original_sha
            assert sum(path.endswith('/grants/revoke-exact') for _, path in calls) == initial_retirements
            assert file_paths[0].read_bytes() == raw[0]
        claims = decode_room_grant(target.adapter._room_grant_secret(), current['grant'], permission='attachment.stage')
        if replacement == 'scope':
            assert claims['issued_at'] > cutoff
        assert all(hosted_rooms.peer_room_grant_is_current(target.home / name, claims=claims)
                   for name in ('state.db', 'shared-state.db'))
        stored = hosted_room_links.load_room_link(service.db_path, room_id='room-one', member_id='member-one')
        assert stored.grant == current['grant']
    finally:
        release.set()
        if source_exit:
            with pytest.raises(SimulatedSourceExit):
                await asyncio.wait_for(running, 15)
        else:
            await asyncio.wait_for(running, 15)

    after = state.get_task(service.db_path, identity)
    if source_exit:
        assert after['status'] == 'running' and after['execution_generation'] == 1
        assert after['payload'] == payload and not admitted
        old_lease = service.runtime._leases['room-one']
        restarted = HostedRoomService(server, db_path=service.db_path)
        assert restarted.runtime.process_generation != service.runtime.process_generation
        monkeypatch.setattr(restarted.runtime, 'clock', lambda: old_lease.expires_at + 1)
        monkeypatch.setattr(restarted.runtime, '_wait_for_terminal', lambda *args, **kwargs: None)
        monkeypatch.setattr(server, 'get_hosted_room_service', lambda: restarted)
        monkeypatch.setattr(methods_groups, 'get_hosted_room_service', lambda: restarted)
        await asyncio.to_thread(restarted.runtime._run_room_once, binding)
        # Retain both automatic recovery and explicit Retry evidence. Retry must
        # observe the accepted attempt or recover it, not be trapped restaging a
        # deleted generation. No generation or task-state values are fabricated.
        retry_error = None
        if not admitted:
            try:
                await asyncio.to_thread(restarted.retry_room_task, 'room-one', task_id=identity.task_id)
            except Exception as exc:
                retry_error = {'type': type(exc).__name__, 'message': str(exc),
                               'status_code': getattr(exc, 'status_code', None)}
        if not admitted:
            now = old_lease.expires_at + restarted.runtime.indeterminate_defer_seconds + 2
            monkeypatch.setattr(restarted.runtime, 'clock', lambda: now)
            await asyncio.to_thread(restarted.runtime._run_room_once, binding)
        recovered = state.get_task(service.db_path, identity)
        successor = restarted.runtime._leases['room-one']
        assert successor.lease_generation > old_lease.lease_generation
        assert successor.process_generation == restarted.runtime.process_generation
        assert recovery_states and all(value == ('indeterminate', 1) for value in recovery_states)
        evidence = {
            'source_status': recovered['status'], 'generation': recovered['execution_generation'],
            'staging_reclaimed': reclaimed, 'admissions': len(admitted),
            'runtime_error': restarted.runtime.status()['last_error'], 'retry_error': retry_error,
            'manifest_generations': [item['execution_generation'] for item in manifests],
            'recovery_states': recovery_states,
            'lease_generations': [old_lease.lease_generation, successor.lease_generation],
            'http_statuses': [(method, path, status) for method, path, status, _ in responses],
        }
        print('RECOVERY_EVIDENCE=' + json.dumps(evidence, sort_keys=True))
        assert len(admitted) == 1, json.dumps(evidence, sort_keys=True)
        assert recovered['execution_generation'] == 1 and recovered['payload'] == payload
        assert all(item['execution_generation'] == 1 for item in manifests)
        if replacement:
            assert reclaimed
            with sqlite3.connect(spool.db_path) as conn:
                assert conn.execute('SELECT staging_origin_json, staging_origin_ambiguous '
                                    'FROM roomlink_attachment_batches').fetchall() == [(origin, 1)]
            assert all(hosted_rooms.room_grant_is_revoked(target.home / name, claims=old_claims)
                       for name in ('state.db', 'shared-state.db'))
        assert all(path.read_bytes() == data for path, data in zip(file_paths, raw))
        settings = admitted[0][2]['payload']['api_turn_v1']['settings']
        assert [Path(item['path']).read_bytes() for item in settings['room_input_media']['media']] == raw
        assert restarted.runtime._thread is None and not restarted.runtime._room_threads
        return
    if replacement:
        assert after['status'] == 'queued' and after['execution_generation'] == 1, after
        assert after['payload'] == payload and not admitted
        # The pending upload still carried the old token; it did not adopt B.
        assert uploads[1][1] == 'HermesRoom ' + issued['grant']
        assert any(method == 'PUT' and status in {401, 403} for method, _, status, _ in responses), responses
        # Advance only the driver's retry clock, not the receiver's grant clock.
        due = service.runtime._unavailable_route_retries[('room-one', 'member-one')]['next_attempt_at']
        monkeypatch.setattr(service.runtime, 'clock', lambda: due + 0.01)
        await asyncio.to_thread(service.runtime._run_room_once, binding)
    await asyncio.sleep(0)
    final = state.get_task(service.db_path, identity)
    expected_generation = 2 if replacement else 1
    assert final['status'] == 'running' and final['execution_generation'] == expected_generation, (final, service.runtime.status())
    assert [item['execution_generation'] for item in manifests] == ([1, 2] if replacement else [1])
    assert final['payload'] == payload and len(admitted) == 1
    authority, _, row = admitted[0]
    assert authority is target.authority
    settings = row['payload']['api_turn_v1']['settings']
    assert settings['room_dispatch']['execution_generation'] == expected_generation
    media = settings['room_input_media']['media']
    assert [Path(item['path']).read_bytes() for item in media] == raw
    assert service.runtime.status()['running'] is False
    assert service.runtime._thread is None and not service.runtime._room_threads
