"""Input-only RoomLink boundaries; no listener, worker, model or control loop.

Adapted from #98072 public 5425e0fa, 25a00343 and 6b33bc15. A capable
catalog is a fake target seam, not an assertion of canonical target activation.
"""
import hashlib
import json
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import hosted_rooms
from gateway.config import GatewayConfig
from gateway.hosted_room_grant_state import grant_state_db_paths, reserve_grant_state
from gateway.hosted_room_input_custody import initialize_input_custody
from gateway.hosted_room_peer import (
    HostedMemberDispatch, attachment_manifest_digest, catalog_mapping,
    decode_room_grant, issue_room_grant,
)
from gateway.hosted_room_execution_policy import execution_policy_mapping
from gateway.platforms import api_server_room_attachments as spool_api
from gateway.platforms import api_server_room_dispatch as dispatch_api
from gateway.platforms import api_server_room_grants as grants
from gateway.platforms import api_server_runs as runs
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from gateway.session_api_turn import prepare_api_execution
from gateway.session_ingress_media import release_admission_media, restore_native_media
from gateway.session_peer_input import check_peer_input, retain_peer_input
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, admit_session_input, begin_runtime_epoch


def _inputs(home, *, epoch=1, count=2, policy=None, catalog=None):
    data = [bytes([65 + i]) * 2048 for i in range(count)]
    manifest = [dict(attachment_id=f'att_{i:032x}', kind='file', name=f'{i}.txt',
        mime='text/plain', size=len(value), sha256=hashlib.sha256(value).hexdigest())
        for i, value in enumerate(data)]
    policy = policy or execution_policy_mapping(target_profile='default')
    catalog = catalog or catalog_mapping(installation_id='target', target_profile='default',
        persistent_process=True, text=True, attachments=True, execution_policy=policy)
    prompt = 'Keep the original user prompt.'
    dispatch = HostedMemberDispatch.from_mapping(dict(
        protocol_version=2, room_id='room', home_install_id='source', authority_gateway_id='source',
        authority_epoch=epoch, member_id='member', target_install_id='target', target_profile='default',
        task_id='task', execution_generation=1, cancellation_scope_id='cancel', source_event_seq=1,
        prompt=prompt, prompt_digest=hashlib.sha256(prompt.encode()).hexdigest(),
        capability_digest=catalog['catalog_digest'], execution_policy_digest=policy['policy_digest'],
        trace_id='trace', attachment_manifest_digest=attachment_manifest_digest(manifest)))
    secret = b'inert-fixture-secret' * 2
    token = issue_room_grant(secret, grant_id=f'fixture-{epoch}', room_id='room', home_install_id='source',
        authority_gateway_id='source', authority_epoch=epoch, member_id='member',
        target_install_id='target', target_profile='default', execution_policy_digest=policy['policy_digest'])
    claims = decode_room_grant(secret, token, permission='attachment.stage')
    spool = spool_api.RoomAttachmentSpool(home / 'shared-state.db', root=home / 'spool')
    spool.prepare(dispatch, manifest)
    for item, value in zip(manifest, data):
        spool.put(claims=claims, task_id='task', execution_generation=1,
                  attachment_id=item['attachment_id'], data=value)
    return spool, dispatch, claims, secret, token, policy, catalog, data


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(hosted_rooms, 'local_authority_gateway_id', lambda: 'target')
    db = SessionDB(tmp_path / 'state.db')
    initialize_input_custody(db)
    runner = SimpleNamespace(_draining=False, session_store=SessionStore(
        config=GatewayConfig(), sessions_dir=tmp_path / 'sessions'))
    owner = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='inert', db=db,
        epoch=begin_runtime_epoch(db, instance_id='inert'))
    runner.session_authority = owner
    owner._schedule = Mock(side_effect=AssertionError('executor is held'))
    adapter = SimpleNamespace(gateway_runner=runner, _ensure_session_db=lambda: db,
        _ensure_session_db_async=AsyncMock(return_value=db), _profile_scope=lambda _: nullcontext(),
        _model_routes={}, _model_name='inert', _background_tasks=set(),
        _parse_session_key_header=lambda _: (None, None), _resolve_route=lambda _: None,
        _request_route_conflict_error=lambda **_: None, _concurrency_limited_response=lambda: None,
        _run_idempotency_scope=lambda _: 'a' * 64, _activate_admitted_request=Mock(),
        _conversation_history_for_session=AsyncMock(return_value=[]),
        _durable_run_status=lambda *_: None)
    runner._adapter_for_source = lambda _: adapter
    runs._initialize_run_state(adapter, store_factory=lambda: RunIdempotencyStore(str(tmp_path / 'runs.db')))
    adapter._set_run_status = partial(runs._set_run_status, adapter)
    adapter._release_run_owner_if_forgotten = Mock()
    adapter._ensure_hosted_member_session = partial(dispatch_api._ensure_hosted_member_session, adapter)
    adapter._room_grant_claims = partial(grants._room_grant_claims, adapter)
    facade = SimpleNamespace(_openai_error=lambda message, **kw: {'error': {'message': message, **kw}},
        _api_request_profile=ContextVar('inert-profile', default='default'),
        _api_request_browser_control_principal=ContextVar('inert-principal', default=None),
        _api_request_browser_control_transport_family=ContextVar('inert-family', default=None),
        _request_turn_author=lambda _: None, _request_agent_overrides=lambda *_, **__: {})
    adapter._normalize_room_dispatch = partial(dispatch_api._normalize_room_dispatch, adapter, _api_server=facade)
    monkeypatch.setattr(runs, '_resolve_conversation_history', lambda *_, **__: ([], None, None, None))
    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)
    monkeypatch.setattr(runs.asyncio, 'to_thread', inline)
    launches = []
    token = object()
    monkeypatch.setattr(runs, '_execute_run', lambda _adapter, launch, **_: (launches.append(launch), token)[1])
    def no_task(value):
        assert value is token
        return Mock()
    monkeypatch.setattr(runs.asyncio, 'create_task', no_task)
    yield adapter, owner, facade, launches
    owner._schedule.assert_not_called()
    adapter._run_idempotency_store.close()
    db.close()


def _request(adapter, inputs):
    spool, dispatch, claims, secret, token, policy, catalog, data = inputs
    reserve_grant_state(grant_state_db_paths(), claims=claims, expires_at=claims['status_expires_at'])
    adapter._room_grant_token = lambda _: token
    adapter._room_grant_secret = lambda: secret
    return SimpleNamespace(headers={'Idempotency-Key': 'room:task:1'},
        json=AsyncMock(return_value={'input': dispatch.prompt, 'hosted_room_dispatch': dispatch.as_mapping()}))


@pytest.mark.asyncio
async def test_real_run_admission_retains_inputs_and_replays_without_spool(receiver, tmp_path, monkeypatch):
    adapter, owner, facade, launches = receiver
    inputs = _inputs(tmp_path)
    spool, dispatch, _, _, _, policy, catalog, data = inputs
    request = _request(adapter, inputs)
    monkeypatch.setattr(grants, '_local_room_catalog', lambda *_: (policy, catalog))
    monkeypatch.setattr(spool_api, '_default_spool', lambda: spool)
    response = await runs._handle_runs(adapter, request, _api_server=facade)
    assert response.status == 202, response.text
    assert len(launches) == 1
    _, ref, row = launches[0].admission
    with owner.db._read_ctx() as conn:
        persisted = json.loads(conn.execute('SELECT payload_json FROM session_admissions WHERE admission_id=?',
                                            (row['admission_id'],)).fetchone()[0])
    assert persisted['text'] == dispatch.prompt
    assert persisted['api_turn_v1']['settings']['room_input_media']['manifest'] == [
        {k: v for k, v in item.items() if k != 'path'} for item in spool.materialize(dispatch)]
    settings = persisted['api_turn_v1']['settings']
    assert [Path(p).read_bytes() for p in restore_native_media(settings['room_input_media']['media'])] == data
    assert str(spool.root) not in json.dumps(persisted)
    prepared = prepare_api_execution(owner, ref, persisted)
    assert prepared['content'].startswith(dispatch.prompt)
    assert all(item['name'] in prepared['content'] for item in settings['room_input_media']['manifest'])
    monkeypatch.setattr(spool, 'materialize', Mock(side_effect=AssertionError('accepted replay must not read spool')))
    replay = await runs._handle_runs(adapter, request, _api_server=facade)
    assert replay.status == 202 and json.loads(replay.text)['run_id'] == json.loads(response.text)['run_id']
    assert len(launches) == 1


@pytest.mark.parametrize('limit', [3072, 0, -1])
def test_batch_budget_precedes_all_capture(tmp_path, monkeypatch, limit):
    from gateway.platforms import base
    from gateway import session_peer_input as peer
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    spool, dispatch, *_ = _inputs(tmp_path)
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: limit)
    capture = Mock(wraps=peer.capture_native_media)
    monkeypatch.setattr(peer, 'capture_native_media', capture)
    if limit > 0:
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            retain_peer_input(spool, dispatch)
        capture.assert_not_called()
    else:
        assert len(retain_peer_input(spool, dispatch)['media']) == 2


@pytest.mark.parametrize('status', ['queued', 'unknown', 'terminal'])
def test_retained_peer_holders_protect_unrelated_native_cleanup(receiver, tmp_path, status):
    adapter, owner, _, _ = receiver
    spool, dispatch, *_ = _inputs(tmp_path)
    retained = retain_peer_input(spool, dispatch)
    owner.db.create_session('holder', source='test')
    peer = admit_session_input(owner.db, epoch=owner.epoch, principal_id='fixture', session_id='holder',
        request_id='peer', payload={'text': dispatch.prompt, 'api_turn_v1': {'history': [], 'settings': {
            'room_dispatch': dispatch.as_mapping(), 'room_input_media': retained}}})
    native = admit_session_input(owner.db, epoch=owner.epoch, principal_id='fixture', session_id='holder',
        request_id='native', payload={'text': 'native', 'attachments_v1': {'media': retained['media']}})
    def seed(conn):
        conn.execute('UPDATE session_admissions SET status=? WHERE admission_id=?', (status, peer['admission_id']))
        conn.execute("UPDATE session_admissions SET status='terminal',outcome='completed' WHERE admission_id=?",
                     (native['admission_id'],))
    owner.db._execute_write(seed)  # Fixture state only, not execution/Stop/unknown resolution.
    assert release_admission_media(owner.db, native['admission_id']) == 0
    assert len(check_peer_input(peer['payload']['api_turn_v1']['settings'])[1]) == 2


@pytest.mark.asyncio
async def test_default_canonical_target_stays_unavailable_before_session_binding(receiver, tmp_path, monkeypatch):
    adapter, _, facade, _ = receiver
    from gateway.hosted_room_peer import GatewayRoomCatalog
    from gateway.hosted_room_execution_policy import RoomExecutionPolicy
    policy, catalog = grants._local_room_catalog(adapter, 'default', 'target')
    assert not GatewayRoomCatalog.from_mapping(catalog).text
    assert RoomExecutionPolicy.from_mapping(policy)
    inputs = _inputs(tmp_path, policy=policy, catalog=catalog)
    request = _request(adapter, inputs)
    monkeypatch.setattr(grants, '_local_room_catalog', lambda *_: (policy, catalog))
    bind = AsyncMock(side_effect=AssertionError('unsupported target must not create a session'))
    adapter._ensure_hosted_member_session = bind
    _, error = await adapter._normalize_room_dispatch(request, await request.json())
    assert error.status == 403
    bind.assert_not_called()


@pytest.mark.asyncio
async def test_changed_grant_after_capture_cannot_reach_admission(receiver, tmp_path, monkeypatch):
    adapter, _, facade, launches = receiver
    inputs = _inputs(tmp_path)
    spool, dispatch, claims, *_ = inputs
    request = _request(adapter, inputs)
    monkeypatch.setattr(spool_api, '_default_spool', lambda: spool)
    adapter._room_grant_claims = Mock(side_effect=[claims, {**claims, 'authority_epoch': 2}])
    _, error = await dispatch_api.prepare_new_room_input(adapter, request,
        {'hosted_room_dispatch': dispatch.as_mapping()}, _openai_error=facade._openai_error)
    assert error.status == 409 and not launches


def test_old_epoch_discard_cannot_remove_new_epoch_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    old_spool, old_dispatch, old_claims, *_ = _inputs(tmp_path, epoch=1)
    spool, dispatch, *_ = _inputs(tmp_path, epoch=2)
    assert old_spool.discard_scope(old_claims) == 1
    assert len(spool.materialize(dispatch)) == 2
    with pytest.raises(spool_api.RoomAttachmentSpoolIncomplete):
        spool.materialize(old_dispatch)


@pytest.mark.parametrize("changed", ["claims", "profile-reservation"])
def test_spool_write_guard_rejects_changed_scope_before_mutation(receiver, tmp_path, monkeypatch, changed):
    adapter, _, _, _ = receiver
    inputs = _inputs(tmp_path)
    spool, dispatch, claims, *_ = inputs
    request = _request(adapter, inputs)
    from gateway.platforms import api_server
    monkeypatch.setattr(api_server, '_api_request_profile', ContextVar('guard-profile', default='default'))
    expected = {**claims, 'authority_epoch': 99} if changed == 'claims' else claims
    if changed == 'profile-reservation':
        adapter._ensure_session_db()._execute_write(lambda conn: conn.execute(
            "UPDATE hosted_room_peer_reservations SET authority_epoch=99 WHERE room_id='room'"))
    guard = spool_api._write_guard(adapter, request, expected, 'attachment.stage')
    manifest = [{k: v for k, v in item.items() if k != 'path'} for item in spool.materialize(dispatch)]
    with pytest.raises((ValueError, grants.RoomGrantReauthorizationRequired), match='room grant changed|no longer current'):
        spool.prepare(replace(dispatch, task_id='never-written'), manifest, authorize_write=guard)
    assert len(spool.materialize(dispatch)) == 2


@pytest.mark.parametrize('current', [False, True])
def test_staging_uses_route_preflight_before_any_upload(monkeypatch, current):
    from tui_gateway.hosted_room_peer_status import _RouteStatusPeerClient
    upload = Mock(return_value={'complete': True})
    guard = Mock(side_effect=None if current else RuntimeError('retired route'))
    client = _RouteStatusPeerClient(SimpleNamespace(stage_attachments=upload), grant='fixture',
        before_admission=guard, on_ready=Mock(), on_reauthorization=Mock(),
        on_unavailable=Mock(), on_refreshed=Mock())
    monkeypatch.setattr('gateway.hosted_room_peer.room_grant_needs_dispatch_refresh', lambda _: False)
    if current:
        assert client.stage_attachments(grant='fixture', dispatch={}, attachments=[]) == {'complete': True}
        assert guard.call_count == 2
        upload.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match='retired route'):
            client.stage_attachments(grant='fixture', dispatch={}, attachments=[])
        upload.assert_not_called()


@pytest.mark.asyncio
async def test_inspection_grant_cannot_discard_input_bytes(receiver, tmp_path, monkeypatch):
    adapter, _, facade, _ = receiver
    inputs = _inputs(tmp_path)
    spool, dispatch, claims, *_ = inputs
    request = _request(adapter, inputs)
    request.match_info = {'task_id': 'task', 'execution_generation': '1'}
    adapter._room_grant_claims = Mock(return_value={**claims, 'permissions': ['status']})
    discard = Mock(wraps=spool.discard_attempt)
    monkeypatch.setattr(spool, 'discard_attempt', discard)
    monkeypatch.setattr(spool_api, '_default_spool', lambda: spool)
    response = await spool_api._handle_room_attachment_discard(adapter, request,
        _openai_error=facade._openai_error, _api_request_profile=facade._api_request_profile)
    assert response.status == 401
    discard.assert_not_called()
    assert len(spool.materialize(dispatch)) == 2


@pytest.mark.asyncio
async def test_existing_input_endpoints_bind_real_guarded_spool_methods(receiver, tmp_path, monkeypatch):
    adapter, _, facade, _ = receiver
    inputs = _inputs(tmp_path)
    spool, dispatch, _, _, _, _, _, data = inputs
    request = _request(adapter, inputs)
    manifest = [{key: value for key, value in item.items() if key != 'path'} for item in spool.materialize(dispatch)]
    adapter._read_json_body = AsyncMock(return_value=({'hosted_room_dispatch': dispatch.as_mapping(),
                                                     'attachments': manifest}, None))
    monkeypatch.setattr(spool_api, '_default_spool', lambda: spool)
    response = await spool_api._handle_room_attachment_manifest(adapter, request,
        _openai_error=facade._openai_error, _api_request_profile=facade._api_request_profile)
    assert response.status == 200, response.text
    request.match_info = {'task_id': 'task', 'execution_generation': '1',
                          'attachment_id': manifest[0]['attachment_id']}
    async def chunks(_size):
        yield data[0]
    request.content = SimpleNamespace(iter_chunked=chunks)
    response = await spool_api._handle_room_attachment_upload(adapter, request,
        _openai_error=facade._openai_error, _api_request_profile=facade._api_request_profile)
    assert response.status == 200, response.text
    adapter._handle_room_member_invitation = Mock()
    adapter._handle_room_member_capabilities = Mock()
    adapter._handle_room_member_grant_refresh = Mock()
    adapter._handle_room_member_grant_revoke = Mock()
    routes = grants._http_routes(adapter)
    endpoints = [(method, path) for method, path, _ in routes]
    for method, path, handler in spool_api._http_routes(adapter):
        assert endpoints.count((method, path)) == 1
        assert callable(handler)
