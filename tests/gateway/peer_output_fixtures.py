"""In-process socket boundary and real Home/target product fixture; no workers."""
import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import asdict
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.error
from urllib.parse import urlsplit

from gateway import hosted_rooms, hosted_room_driver as tasks
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from gateway.session_hosted_service import CanonicalHostedRoomService
from gateway.runtime_ownership import process_ownership
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from gateway.run import _profile_runtime_scope


def inprocess_http(target, monkeypatch):
    from aiohttp.test_utils import make_mocked_request
    from tui_gateway import hosted_room_peer_http, hosted_room_peer_artifacts
    loop = asyncio.get_running_loop()
    calls, replies = [], []
    faults = SimpleNamespace(lost_ack=False, lost_discard=False, after_body=None, after_discard=None)
    async def send(wire):
        path, method = urlsplit(wire.full_url).path, wire.get_method()
        data = wire.data or b''
        if not isinstance(data, bytes):
            data = b''.join(data)
        class Content:
            async def iter_chunked(self, size):
                for offset in range(0, len(data), size):
                    yield data[offset:offset + size]
        req = make_mocked_request(method, path, headers=dict(wire.header_items()), payload=Content())
        match = await target.adapter._app.router.resolve(req)
        req._match_info = match
        async def body():
            value = json.loads(data) if data else {}
            if faults.after_body and path.endswith(('/artifacts/ack', '/artifacts/discard')):
                faults.after_body()
            return value
        req.json = body
        calls.append((method, path))
        with _profile_runtime_scope(target.home, hydrate_secrets=False):
            response = await match.handler(req)
        replies.append((method, path, response.status, response.body))
        if response.status >= 400:
            print("inert HTTP refusal", method, path, response.status, response.body)
            raise urllib.error.HTTPError(wire.full_url, response.status, response.reason,
                                         response.headers, io.BytesIO(response.body))
        if faults.lost_ack and path.endswith('/artifacts/ack'):
            faults.lost_ack = False
            raise TimeoutError('inert lost ACK response AFTER real target commit')
        if path.endswith('/artifacts/discard'):
            if faults.after_discard:
                faults.after_discard()
            if faults.lost_discard:
                faults.lost_discard = False
                raise TimeoutError('inert lost discard reply AFTER target retirement')
        return io.BytesIO(response.body)
    def open_wire(wire, **kwargs):
        if '/artifacts/' in wire.full_url:
            assert kwargs.get('reject_redirects') is True
        return asyncio.run_coroutine_threadsafe(send(wire), loop).result(15)
    monkeypatch.setattr(hosted_room_peer_http, '_open_roomlink_url', open_wire)
    monkeypatch.setattr(hosted_room_peer_artifacts, '_open_roomlink_url', open_wire)
    return SimpleNamespace(calls=calls, replies=replies, faults=faults)


@asynccontextmanager
async def peer_case(target, monkeypatch, *, defer_publication=True, resolve_queued=False, output_count=1, settle_target=True):
    from tests.gateway.test_peer_output_product import register_routes
    from tests.gateway.test_canonical_peer_target_setup import invite, invitation
    from gateway.session_peer_output import initialize_peer_output
    from gateway.hosted_room_peer import GatewayRoomCatalog, decode_room_grant
    from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    from tui_gateway.hosted_room_driver import HostedRoomRuntime, _find_terminal_receipt
    from gateway.hosted_room_input_custody import initialize_input_custody
    from gateway.platforms import api_server_runs
    from tools import hosted_room_artifact
    from gateway.session_api_turn import api_execution
    from gateway.session_results import execution_result
    from gateway.session_hosted_output import current_output_binding

    # Two physical installs share one interpreter only at this inert HTTP boundary.
    # Resolve each real persisted installation ID from its held Home scope.
    from hermes_constants import get_hermes_home
    monkeypatch.setattr("hermes_cli.install_identity.get_default_hermes_root", get_hermes_home)

    def forbidden(*args, **kwargs):
        raise AssertionError('listener/runtime/model startup outside the bounded fixture')
    monkeypatch.setattr(HostedRoomRuntime, 'start', forbidden)
    monkeypatch.setattr(target.authority, '_schedule', lambda ref: None)
    target.runner._cached_agent_for = lambda _: None
    register_routes(target)
    initialize_peer_output(target.adapter)
    # Context sizing normally fetches external model metadata. This fixture
    # tests real Output/Files/ACK, not model-catalog HTTP; sockets stay forbidden.
    monkeypatch.setattr('model_tools._resolve_active_context_length', lambda: 32768)
    wire = inprocess_http(target, monkeypatch)
    home = target.home.parent / 'source'
    home.mkdir()
    (home / 'config.yaml').write_text('approvals:\n  mode: manual\n')
    process_ownership.reserve([home])
    with _profile_runtime_scope(home, hydrate_secrets=False):
        db = SessionDB(home / 'state.db')
        initialize_input_custody(db)
        runner = SimpleNamespace(_draining=False, config=GatewayConfig(multiplex_profiles=True), adapters={},
            session_store=SessionStore(config=GatewayConfig(), sessions_dir=home / 'sessions'))
        authority = SessionAuthority(runner, profile_id=str(home), instance_id='home-owner', db=db,
            epoch=begin_runtime_epoch(db, instance_id='home-owner'))
        runner.session_authority = authority
        registry = runner.session_authorities = SessionAuthorities(home)
        registry.add(home, authority)
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        authority.hosted_room_service = service
        home_id = hosted_rooms.local_authority_gateway_id()
        service.authorize_room('alice', 'room-one', create=True)
    issued = await invite(target, invitation() | dict(member_id='writer', home_install_id=home_id, authority_gateway_id=home_id))
    claims = decode_room_grant(target.adapter._room_grant_secret(), issued['grant'], permission='artifact.ack')
    catalog = GatewayRoomCatalog.from_mapping(issued['catalog'])
    client = PeerRunsHTTPClient(base_url='http://127.0.0.1:8642', api_key='', receipt_db_path=db.db_path)
    passive = (await asyncio.to_thread(client.probe, grant=issued['grant']))['catalog']
    from gateway.hosted_room_peer import _catalog_digest
    assert passive['attachments'] is False
    assert _catalog_digest(dict(passive, attachments=True)) == issued['catalog']['catalog_digest']
    route = PeerMemberRoute(home_install_id=home_id, member_id='writer', target_install_id=catalog.installation_id,
        target_profile='default', capability_digest=catalog.catalog_digest, cancellation_scope_id='cancel-one',
        trace_id='trace-one', grant=issued['grant'], execution_policy_digest=catalog.execution_policy.policy_digest,
        attachments=True)
    with _profile_runtime_scope(home, hydrate_secrets=False):
        service.create_room(room_id='room-one', name='Peer Files', members=[
            dict(member_id='writer', profile='default', handle='writer', target=dict(kind='peer',
                 peer_id=catalog.installation_id, installation_id=catalog.installation_id, profile='default', capability_digest=catalog.catalog_digest)),
            dict(member_id='reader', profile='default', handle='reader')])
        service.register_peer_route(room_id='room-one', member_id='writer', route=route, client=client,
                                   target_url=client.base_url, catalog=catalog)
        raw = [b'independent source document', base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII=')]
        uploads = [service.attachments.put(room_id='room-one', upload_id='input-' + str(n), name=name,
            kind=kind, mime=mime, data=data) for n, (name, kind, mime, data) in enumerate([
                ('input.txt', 'file', 'text/plain', raw[0]), ('pixel.png', 'image', 'image/png', raw[1])])]
        manifest = [{k: item[k] for k in ('attachment_id', 'name', 'kind', 'mime', 'size')} for item in uploads]
        service.send(room_id='room-one', event_id='request-one', payload=dict(thread_id='thread-one',
            text='@writer Return a file from this document and PNG.', attachments=manifest))
        task = tasks.list_tasks(db.db_path, room_id='room-one', status='queued')[0]
        assert task['payload']['recipient_member_ids'] == ['writer', 'reader']
        binding = service.bindings()[0]
        if resolve_queued:
            assert task['status'] == 'queued'
            await asyncio.to_thread(service._resolve_member_transport, binding, task)
        lease = tasks.acquire_lease(db.db_path, room_id='room-one', gateway_id=binding.gateway_id,
            authority_epoch=binding.authority_epoch, process_generation='manual-fixture', ttl_seconds=600, clock=time.time)
        attempt = tasks.start_task(db.db_path, task['identity'], lease, expected_cancel_generation=0, clock=time.time)
        task = tasks.get_task(db.db_path, task["identity"])
        rpc = await asyncio.to_thread(service._resolve_member_transport, binding, task)
    launched, executions, outputs, contexts = [], [], [], []
    async def inert(adapter, launch, **kwargs):
        launched.append(launch.admission)
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    output = target.home / 'cache' / 'answer.txt'
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(b'exact produced bytes from peer document and PNG')
    async def handle(event):
        executions.append(event.message_id)
        from contextvars import copy_context
        contexts.append(copy_context())
        from model_tools import get_tool_definitions
        assert any(t["function"]["name"] == "share_group_file"
                   for t in get_tool_definitions(enabled_toolsets=["bot_room"]))
        prepared = api_execution.get()
        assert prepared is not None and current_output_binding() is not None
        admitted_media = launched[0][2]['payload']['api_turn_v1']['settings']['room_input_media']['media']
        assert [Path(x['path']).read_bytes() for x in admitted_media] == raw
        assert admitted_media[0]['path'] in str(prepared['content'])
        assert 'data:image/png;base64,' in str(prepared['content'])
        for index in range(output_count):
            produced = output if index == 0 else output.with_name(f'answer-{index}.txt')
            if index:
                produced.write_bytes(f'exact peer output {index}'.encode())
            result = json.loads(await asyncio.to_thread(hosted_room_artifact.share_group_file, str(produced)))
            outputs.append(result)
            assert result.get('ok') is True, result
        execution_result.get().update(result=dict(final_response='Shared the file.', messages=[], completed=True), usage={})
        return 'Shared the file.'
    target.runner._handle_message = handle
    try:
        coords = dict(profile='default', source='bot_room')
        with _profile_runtime_scope(home, hydrate_secrets=False):
            session_id = (await asyncio.to_thread(rpc.create, **coords, title='Group: room-one'))['session_id']
            accepted = await asyncio.to_thread(rpc.submit, **coords, session_id=session_id, prompt=task['payload']['prompt'],
                task=task['identity'], execution_generation=attempt.execution_generation,
                on_terminal=forbidden, attachments=task['payload']['attachments'])
        await asyncio.sleep(0)
        assert len(launched) == 1
        owner, ref, row = launched[0]
        stored = None
        if settle_target:
            with _profile_runtime_scope(target.home, hydrate_secrets=False):
                await asyncio.wait_for(owner._drain(ref), timeout=10)
                assert current_output_binding() is None
                if contexts:
                    assert json.loads(contexts[0].run(hosted_room_artifact.share_group_file, str(output)))["ok"] is False
            with _profile_runtime_scope(home, hydrate_secrets=False):
                history = await asyncio.to_thread(rpc.history, **coords, session_id=session_id)
                terminal = _find_terminal_receipt(history, task['identity'], attempt.execution_generation)
                assert terminal is not None and terminal.status == 'settled', history
                stored = tasks.settle_task(db.db_path, attempt, **asdict(terminal), clock=time.time)
        case = SimpleNamespace(home=home, db=db, authority=authority, service=service, wire=wire, issued=issued,
            claims=claims, client=client, rpc=rpc, task=task, attempt=attempt, binding=binding, accepted=accepted,
            launched=launched, executions=executions, outputs=outputs, output=output, raw=raw, stored=stored,
            target=target, row=row)
        yield case
    finally:
        db.close()
        process_ownership.release(home)
