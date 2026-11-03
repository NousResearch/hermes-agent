"""Already-reserved compute remains able to persist while new work is refused."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_reserved_worker_persists_while_runtime_drains(tmp_path):
    from gateway.session_worker import worker_request
    from gateway.session_worker_reservation import reserve_admission_worker
    from tests.gateway.test_worker_admission_reservation import hello
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input, RuntimeStoreError
    db = SessionDB(tmp_path / 'state.db')
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,  # noqa: ASYNC220 -- Popen returns immediately; the test drives the child through its handle
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        db.create_session('owned', 'cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path),
            _require_admission_open=lambda: None, logical_owner=lambda sid: sid, sessions={})
        row = admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
        claim_session_input(db, epoch=epoch, session_id='owned')
        scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=process,
                                         principal_id='human', hello=hello(process))
        actor = SimpleNamespace(subject='human', profile_id=str(tmp_path), capabilities={'worker:adopt'})
        connection = SimpleNamespace(authority=authority, actor=actor)
        identity = {k: v for k, v in scope.items() if k != 'epoch'}
        await worker_request(connection, None, identity, operation='adopt')
        def draining():
            raise RuntimeStoreError('runtime_draining')
        authority._require_admission_open = draining
        await worker_request(connection, None, scope | {
            'sequence': 1, 'operation': 'session.prompt', 'payload': {'system_prompt': 'settled while draining'}}, operation='persist')
        assert db.get_session('owned')['system_prompt'] == 'settled while draining'
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            await worker_request(connection, None, identity | {'kind': 'compute'}, operation='register')
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('purpose', ['worker-adoption', 'interactive'])
@pytest.mark.parametrize('state', ['ready', 'draining'])
async def test_draining_ws_redeems_only_worker_ticket(tmp_path, monkeypatch, purpose, state):
    from gateway.run_api import GatewayRuntimeAPI
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import web_server as web
    from hermes_cli import web_server_chat
    tickets = TicketStore('owner', [str(tmp_path)])
    ticket = tickets.mint(profile_id=str(tmp_path), subject='human', purpose=purpose)
    authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / 'state.db'), profile_id=str(tmp_path), runner=None)
    runner = SimpleNamespace(_draining=True, session_runtime_descriptor={'state': state},
                             session_authority=authority, session_ticket_store=tickets)
    seen, sent = [], []
    async def fallback(*args):
        pytest.fail('draining native calls must not enter legacy fallback')
    async def handle(ws, **kwargs):
        await ws.accept(subprotocol='hermes-gateway-v1')
        seen.append((await ws.receive_json(), kwargs['auth_identity']['capabilities']))
    monkeypatch.setattr('tui_gateway.ws.handle_ws', handle)
    monkeypatch.setattr(web, '_DASHBOARD_EMBEDDED_CHAT_ENABLED', True)
    monkeypatch.setattr(web_server_chat, '_ws_request_is_allowed', lambda ws: True)
    incoming = iter([{'type': 'websocket.connect'}, {'type': 'websocket.receive', 'text': json.dumps({'method': 'worker.persist'})}])
    async def receive(): return next(incoming)
    async def send(value): sent.append(value)
    scope = {'type': 'websocket', 'path': '/api/ws', 'scheme': 'ws', 'query_string': b'',
        'client': ('127.0.0.1', 1234), 'server': ('127.0.0.1', 9999),
        'headers': [(b'sec-websocket-protocol', ('hermes-gateway-v1, hermes-gateway-ticket.' + ticket).encode())],
        'subprotocols': ['hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket]}
    await GatewayRuntimeAPI(fallback, runner, web.app)(scope, receive, send)
    if purpose == 'worker-adoption':
        assert seen == [({'method': 'worker.persist'}, frozenset({'worker:adopt'}))]
    else:
        assert not seen
        assert sent[-1]['type'] == 'websocket.close'


@pytest.mark.asyncio
async def test_listener_and_tickets_live_until_settlement(tmp_path, monkeypatch):
    from gateway.run_runtime import drain_gateway_runtime, settle_gateway_runtime
    from gateway.runtime_bootstrap import TicketStore
    tickets = TicketStore('owner', [str(tmp_path)])
    ticket = tickets.mint(profile_id=str(tmp_path), subject='human', purpose='worker-adoption')
    closed = []
    async def stop_api(handle): closed.append(handle)
    async def stop_service(runner): pass
    monkeypatch.setattr('gateway.run_api.stop_gateway_api', stop_api)
    monkeypatch.setattr('gateway.session_hosted_service.stop_hosted_service', stop_service)
    runner = SimpleNamespace(_draining=False, adapters={}, session_runtime_descriptor={'state': 'ready'},
        session_ticket_store=tickets, session_api='listener')
    await drain_gateway_runtime(runner)
    assert not closed
    assert tickets.redeem(ticket, profile_id=str(tmp_path), purpose='worker-adoption')['subject'] == 'human'
    await settle_gateway_runtime(runner)
    assert closed == ['listener']
