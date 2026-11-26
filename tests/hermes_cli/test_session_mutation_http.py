"""Real HTTP and WS clients share the owner revision and retry receipt."""
import asyncio
import json
import socket
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, WebSocket
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_http_and_ws_clients_compete_for_one_revision(tmp_path, monkeypatch):
    from hermes_cli.web_routers.sessions import manage_router
    from hermes_cli import web_server as web
    monkeypatch.setattr(web, '_SESSION_TOKEN', 'fixture-token')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.state.auth_required = False
    app.include_router(manage_router)
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, 'session-fixture', 'test')
        epoch = store.runtime.begin_epoch(instance_id='owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path), instance_id='owner', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['session-fixture'] = LiveSession(None, 'route')
        app.state.session_authority = authority

        @app.websocket('/ws')
        async def ws_route(ws: WebSocket):
            if ws.query_params.get('token') != 'fixture-token':
                await ws.close(code=4401)
                return
            await ws.accept()
            connection = AuthorityConnection(authority, object(), {'user_id': 'dashboard-token'})
            try:
                request = await ws.receive_json()
                await ws.send_json(await connection.dispatch(request))
            finally:
                await connection.close()

        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, lifespan='off', log_level='error'))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}') as client:
                for _ in range(200):
                    if server.started:
                        break
                    await asyncio.sleep(.01)
                assert server.started
                body = {'request_id': 'edit', 'expected_revision': 0, 'title': 'Title', 'pinned': True}
                first = await client.patch('/api/sessions/session-fixture?token=fixture-token', json=body, headers={'Authorization': 'Bearer fixture-token'})
                assert first.status_code == 200, first.text
                assert first.json()['revision'] == 1
                assert first.json()['ok'] is True
                import websockets
                async with websockets.connect(f'ws://127.0.0.1:{port}/ws?token=fixture-token') as ws:
                    await ws.send(json.dumps({'id': 1, 'method': 'session.mutate', 'params': {
                        'session_id': 'session-fixture', 'request_id': 'competitor', 'expected_revision': 0,
                        'operation': 'sidebar', 'payload': {'title': 'Stale'}}}))
                    competing = json.loads(await ws.recv())
                    assert competing['error']['message'] == 'revision_conflict'
                retry = await client.patch('/api/sessions/session-fixture?token=fixture-token', json=body, headers={'Authorization': 'Bearer fixture-token'})
                assert retry.json() == first.json()
                denied = await client.patch('/api/sessions/session-fixture', json=body)
                assert denied.status_code == 401
                foreign = await client.patch('/api/sessions/session-fixture?token=fixture-token', json=body | {'profile': 'other'}, headers={'Authorization': 'Bearer fixture-token'})
                assert foreign.status_code in (400, 403, 404)
                assert session_row(store, 'session-fixture')['runtime_revision'] == 1
                assert session_row(store, 'session-fixture')['title'] == 'Title'
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 10)
            listener.close()


@pytest.mark.asyncio
async def test_http_import_and_delete_keep_atomic_receipts(tmp_path, monkeypatch):
    from hermes_cli.web_routers.sessions import manage_router
    from hermes_cli import web_server as web
    monkeypatch.setattr(web, '_SESSION_TOKEN', 'fixture-token')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.state.auth_required = False
    app.include_router(manage_router)
    with open_persistence(profile_scope(home=tmp_path)) as store:
        epoch = store.runtime.begin_epoch(instance_id='owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path), instance_id='owner', epoch=epoch, persistence=storage_handle(tmp_path))
        app.state.session_authority = authority
        headers = {'Authorization': 'Bearer fixture-token'}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost', headers=headers) as client:
            body = {'request_id': 'import', 'expected_revision': 0,
                    'sessions': [{'id': 'imported-session', 'source': 'cli', 'messages': [{'role': 'user', 'content': 'keep'}]}]}
            raw_holder(store)._execute_write(lambda c: c.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.%' BEGIN SELECT RAISE(ABORT, 'receipt failed'); END"))
            failed = await client.post('/api/sessions/import', json=body)
            assert failed.status_code == 503, failed.text
            assert session_row(store, 'imported-session') is None
            raw_holder(store)._execute_write(lambda c: c.execute('DROP TRIGGER fail_receipt'))
            imported = await client.post('/api/sessions/import', json=body)
            assert imported.status_code == 200, imported.text
            assert (await client.post('/api/sessions/import', json=body)).json() == imported.json()
            authority.sessions['imported-session'] = LiveSession(None, 'route')
            query = {'request_id': 'delete', 'expected_revision': imported.json()['revision'], 'expected_generation': 0}
            deleted = await client.delete('/api/sessions/imported-session', params=query)
            assert deleted.status_code == 200, deleted.text
            assert deleted.json()['deleted_ids'] == ['imported-session']
            assert (await client.delete('/api/sessions/imported-session', params=query)).json() == deleted.json()
            assert session_row(store, 'imported-session') is None


@pytest.mark.asyncio
async def test_delete_reports_already_absent_only_for_rows_that_are_gone(tmp_path, monkeypatch):
    """M3: the authority also answers ``not_found`` for an EXISTING cold row it will not restore
    for this caller (an unbound API-server row is mounted for its owner only). That is not an
    idempotent success — Desktop would drop the row and it reappears on reload. Only a genuinely
    absent id is ``already_absent``."""
    from gateway.session_contract import SessionActor
    from hermes_cli.web_routers.sessions import manage_router
    from hermes_cli import web_server as web
    import hermes_cli.web_server_sessions as web_sessions
    monkeypatch.setattr(web, '_SESSION_TOKEN', 'fixture-token')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.state.auth_required = False
    app.include_router(manage_router)
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, 'cold-api', 'api_server', chat_id='cold-api', session_key='agent:main:api:cold')
        epoch = store.runtime.begin_epoch(instance_id='owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path),
                                     instance_id='owner', epoch=epoch, persistence=storage_handle(tmp_path))
        login = SessionActor('basic:alice', str(tmp_path), frozenset({'session:read', 'session:control'}), 'http')
        monkeypatch.setattr(web_sessions, '_session_mutation_context', lambda request, profile: (authority, login))
        query = {'request_id': 'delete', 'expected_revision': 0, 'expected_generation': 0}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost',
                                     headers={'Authorization': 'Bearer fixture-token'}) as client:
            unrestorable = await client.delete('/api/sessions/cold-api', params=query)
            gone = await client.delete('/api/sessions/never-existed', params=query)
        assert unrestorable.status_code == 404 and unrestorable.json()['detail'] == 'not_found', unrestorable.text
        assert session_row(store, 'cold-api') is not None
        assert gone.status_code == 200 and gone.json() == {'ok': True, 'already_absent': True}, gone.text
