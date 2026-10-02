"""Causal recovery, refusal classification and bounded output worker failures."""
import copy
import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_artifacts import terminal_artifact_manifest
from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
from gateway.platforms.api_server_peer_output import handle
from gateway.session_hosted_output import output_binding
from gateway.session_peer_output import PeerOutputStateUnavailable, receipt_fields
from hermes_state_runtime import settle_session_input
from tests.gateway.test_peer_output_contracts import Request, admitted
from tests.gateway.test_hosted_room_artifacts import _scope
from tui_gateway.hosted_room_peer_output import OutputPending, PeerOutputSource


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['prompt', 'missing', 'wrong_type', 'scope'])
async def test_corrupt_unknown_dispatch_never_projects_false_terminal(tmp_path, monkeypatch, fault):
    async with admitted(tmp_path, monkeypatch) as p:
        row = copy.deepcopy(p.row)
        row['status'] = 'unknown'
        dispatch = row['payload']['api_turn_v1']['settings']['room_dispatch']
        if fault == 'prompt':
            dispatch['prompt'] += ' changed'
        elif fault == 'missing':
            del dispatch['task_id']
        elif fault == 'wrong_type':
            row['payload']['api_turn_v1']['settings']['room_dispatch'] = 'corrupt'
        else:
            row['payload']['api_turn_v1']['run_owner_scope'] = 'changed'
        with pytest.raises(PeerOutputStateUnavailable):
            receipt_fields(row, {})
        assert receipt_fields({**p.row, 'status': 'unknown'}, {})['peer_output_unresolved'] == p.scope.as_mapping()


def grant_request(p, operation, body):
    grant = issue_room_grant(p.adapter._room_grant_secret(), grant_id='hardening',
        **{key: getattr(p.dispatch, key) for key in ('room_id', 'home_install_id', 'authority_gateway_id',
            'authority_epoch', 'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest')},
        permissions=('status', 'stop'))
    claims = decode_room_grant(p.adapter._room_grant_secret(), grant, permission='status')
    hosted_rooms.reserve_peer_room(p.authority.db.db_path, claims=claims, expires_at=claims['expires_at'])
    return Request(body, grant=grant, operation=operation)


@pytest.mark.asyncio
async def test_active_source_is_retryable_but_changed_commitment_stays_blocked(tmp_path, monkeypatch):
    async with admitted(tmp_path, monkeypatch) as p:
        binding = await output_binding(p.authority, p.ref, p.row)
        outbox = binding._outbox()
        artifact = outbox.put_bytes(scope=p.scope, data=b'retain me', source_name='note.txt')
        body = {'artifact_scope': p.scope.as_mapping(), 'manifest_digest': None}
        request = grant_request(p, 'discard', body)
        pending = await handle(p.adapter, request)
        assert pending.status == 503 and json.loads(pending.text)['error']['code'] == 'peer_output_not_ready'
        assert outbox.read(p.scope, artifact['artifact_id'])[1] == b'retain me'
        settle_session_input(p.authority.db, epoch=p.authority.epoch, admission_id=p.row['admission_id'],
            generation=p.row['generation'], outcome='interrupted')
        request.body = {**body, 'manifest_digest': 'changed'}
        assert (await handle(p.adapter, request)).status == 409
        assert outbox.list(p.scope) == [artifact]
        request.body = body
        assert (await handle(p.adapter, request)).status == 200
        assert outbox.list(p.scope) == []


@pytest.mark.asyncio
async def test_disposition_winning_during_read_is_retryable_without_returning_bytes(tmp_path, monkeypatch):
    from gateway.hosted_room_artifacts import RoomArtifactOutbox
    async with admitted(tmp_path, monkeypatch) as p:
        binding = await output_binding(p.authority, p.ref, p.row)
        outbox = binding._outbox()
        artifact = outbox.put_bytes(scope=p.scope, data=b'retain me', source_name='note.txt')
        manifest = terminal_artifact_manifest([artifact])
        settle_session_input(p.authority.db, epoch=p.authority.epoch, admission_id=p.row['admission_id'],
            generation=p.row['generation'], outcome='completed', result={'result': {
                'artifacts': manifest, 'artifact_scope': p.scope.as_mapping()}})
        request = grant_request(p, 'read', {'artifact_scope': p.scope.as_mapping(),
            'manifest_digest': manifest['manifest_digest'], 'artifact_id': artifact['artifact_id']})
        original = RoomArtifactOutbox.read
        def raced_read(store, scope, artifact_id):
            value = original(store, scope, artifact_id)
            # Another exact disposition commits after bytes were read, before recheck.
            from gateway.platforms.api_server_peer_output import _RECEIPT
            p.authority.db._execute_write(lambda conn: conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                (_RECEIPT + scope.key, json.dumps({'operation': 'discard'}))))
            return value
        monkeypatch.setattr(RoomArtifactOutbox, 'read', raced_read)
        response = await handle(p.adapter, request)
        assert response.status == 503 and 'data_base64' not in response.text


def source_fixture():
    scope = _scope()
    item = {'artifact_id': 'rart_' + 'a' * 32, 'kind': 'file', 'name': 'report.txt', 'mime': 'text/plain',
            'size': 1, 'sha256': '0' * 64}
    manifest = terminal_artifact_manifest([item])
    encoded = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
    record = {'operation': 'ack', 'state': 'pending', 'scope_json': encoded(scope.as_mapping()),
              'manifest_json': encoded(manifest)}
    service = SimpleNamespace(_peer_output_generation=0, _peer_output_io=OrderedDict(),
        _obligation=lambda *args: record, runtime=SimpleNamespace(wakeup=lambda: None))
    # Exercise the actual I/O worker/cache seam; authority checks have their own
    # real-owner tests. No transport method is called on these refusing paths.
    source = object.__new__(PeerOutputSource)
    source.service, source.scope, source.manifest = service, scope, manifest
    source.generation, source.endpoint = 0, 'http://127.0.0.1:1'
    source.route = SimpleNamespace(trace_id='trace', cancellation_scope_id='cancel', grant='grant')
    source.client = SimpleNamespace(output_request=lambda **kwargs: pytest.fail('unexpected network effect'))
    return source, record


def test_retired_obligation_refuses_all_prefetch_before_start():
    source, record = source_fixture()
    record['state'] = 'completed'
    with pytest.raises(ValueError, match='obligation changed'):
        source.read(source.scope, 'rart_' + 'a' * 32)
    assert not source.service._peer_output_io


def test_thread_start_failure_does_not_exhaust_bounded_io_slots(monkeypatch):
    import agent.memory_provider
    source, _ = source_fixture()
    def fail_start():
        raise RuntimeError('thread resources exhausted')
    monkeypatch.setattr(agent.memory_provider, 'spawn_context_thread',
                        lambda *args, **kwargs: SimpleNamespace(start=fail_start))
    for _ in range(6):
        with pytest.raises(OutputPending, match='could not start'):
            source.read(source.scope, 'rart_' + 'a' * 32)
        assert not source.service._peer_output_io


@pytest.mark.asyncio
async def test_missing_consent_retains_unreported_cleanup_but_proven_text_does_not(tmp_path, monkeypatch):
    from gateway.session_hosted_output_publication import CanonicalHostedOutput
    from tui_gateway.hosted_room_peer_output import consent_key
    async with admitted(tmp_path, monkeypatch) as p:
        room = {'room_id': p.scope.room_id, 'authority_gateway_id': p.scope.authority_gateway_id,
            'authority_epoch': p.scope.authority_epoch, 'members': [{'member_id': p.scope.member_id,
            'profile': p.scope.target_profile, 'target': {'kind': 'peer',
                'profile': p.scope.target_profile, 'installation_id': p.scope.target_install_id}}]}
        task = {'identity': SimpleNamespace(task_id=p.scope.task_id), 'execution_generation': p.scope.execution_generation,
            'status': 'cancelled', 'cancel_generation': 1, 'payload': {'target_member_id': p.scope.member_id,
                'target_profile': p.scope.target_profile}}
        service = SimpleNamespace(db_path=p.authority.db.db_path)
        assert CanonicalHostedOutput._unreported_output(service, room, task) == (p.scope, None)
        with p.authority.db._read_ctx() as conn:
            assert conn.execute('SELECT value FROM state_meta WHERE key=?', (consent_key(p.scope.as_mapping()),)).fetchone() is None
        p.authority.db._execute_write(lambda conn: conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
            (consent_key(p.scope.as_mapping()), json.dumps({'contract': None, 'dispatched': True}))))
        assert CanonicalHostedOutput._unreported_output(service, room, task) == (p.scope, None)
        p.authority.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
            (json.dumps({'contract': None, 'dispatched': True, 'provenance': 'canonical-dispatch-v1'}), consent_key(p.scope.as_mapping()))))
        assert CanonicalHostedOutput._unreported_output(service, room, task) is None


def test_observation_only_legacy_text_cannot_submit_a_run(monkeypatch):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
    from tests.tui_gateway.test_hosted_room_peer_http import _dispatch
    client = PeerRunsHTTPClient(base_url='http://127.0.0.1:1', api_key='')
    def no_request(*args, **kwargs):
        pytest.fail('observation-only legacy recovery attempted network admission')
    monkeypatch.setattr(client, '_request', no_request)
    with pytest.raises(PeerRunsHTTPError) as refused:
        client.recover_dispatch(dispatch=_dispatch(), grant='signed.room.grant', observation_only=True)
    assert refused.value.ambiguous and not refused.value.not_admitted
