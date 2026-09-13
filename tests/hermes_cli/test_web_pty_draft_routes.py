"""Actual /api/pty + /api/pub routes with a real echo PTY process."""
import asyncio
from contextlib import suppress
import json
import os
import queue
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli.pty_session import PtySessionRegistry

PREFIX = '\x00hermes-draft-v1:'


@pytest.mark.linux_only
@pytest.mark.parametrize('keepalive', [False, True])
def test_routes_bind_private_controller_to_actual_pty_and_consume_all_control(monkeypatch, keepalive):
    import hermes_cli.web_server as server
    import hermes_cli.web_server_chat as chat
    from hermes_cli.web_routers.chat_ws import router
    import json

    captured = queue.Queue()
    registry = PtySessionRegistry(ttl=60, max_sessions=4, buffer_cap=4096, read_timeout=.02)
    monkeypatch.setattr(chat, 'PTY_REGISTRY', registry)
    monkeypatch.setattr(server, '_DASHBOARD_EMBEDDED_CHAT_ENABLED', True)
    monkeypatch.setattr(server.app.state, 'bound_host', '127.0.0.1', raising=False)
    monkeypatch.setattr(server.app.state, 'bound_port', 9999, raising=False)
    monkeypatch.setattr(server.app.state, 'auth_required', False, raising=False)
    code = "import os,tty; tty.setraw(0); os.write(1,b'ready');\nwhile True:\n data=os.read(0,65536); os.write(1,b'INPUT:'+data.hex().encode()+b'\\n')"

    def resolve(**kwargs):
        captured.put(kwargs['sidecar_url'])
        return [sys.executable, '-u', '-c', code], None, dict(os.environ)

    monkeypatch.setattr(chat, '_resolve_chat_argv', resolve)
    app = FastAPI()
    app.include_router(router)
    auth = {'token': server._SESSION_TOKEN}
    params = {**auth, 'channel': 'original', **({'attach': 'opaque'} if keepalive else {})}
    pty_url = '/api/pty?' + urlencode(params)

    def state(node, binding):
        node.send_json({**binding, 'type': 'draft.state', 'session_id': 'native-sid',
                        'draft_id': 'native-draft', 'available': True})

    with TestClient(app, headers={'host': '127.0.0.1'}) as client:
        try:
            with client.websocket_connect(pty_url) as browser:
                assert browser.receive_bytes() == b'ready'
                url = captured.get(timeout=5)
                query = parse_qs(urlsplit(url).query)
                assert query.get('controller'), 'PTY launch must contain a server-issued private controller'
                pub_url = '/api/pub?' + urlencode({key: value[0] for key, value in query.items()}) + '&controller_generation=1'
                with client.websocket_connect(pub_url) as node:
                    binding = node.receive_json()
                    assert binding['type'] == 'draft.refresh'
                    state(node, binding)
                    identity = browser.receive_json()['identity']
                    assert query['controller'][0] not in json.dumps(identity)
                    request = {'type': 'draft.attach', 'request_id': 'one', 'expected': identity, 'path': '/staged/file.txt'}
                    browser.send_text(PREFIX + '{broken')
                    browser.send_bytes((PREFIX + '{"type":"shell.exec"}').encode())
                    browser.send_text(PREFIX + json.dumps(request))
                    assert node.receive_json() == request
                    node.send_json({'type': 'draft.result', 'request_id': 'one', 'identity': identity, 'status': 'attached'})
                    assert browser.receive_json()['status'] == 'attached'
                    browser.send_text('proof')
                    assert browser.receive_bytes() == b'INPUT:70726f6f66\n'
                    if keepalive:
                        # New browser channel must retain the original live child controller.
                        with client.websocket_connect('/api/pty?' + urlencode({**params, 'channel': 'replacement'})) as second:
                            refresh = node.receive_json()
                            assert refresh['type'] == 'draft.refresh'
                            state(node, refresh)
                            while True:
                                frame = second.receive()
                                if 'text' in frame:
                                    new_identity = json.loads(frame['text'])['identity']
                                    break
                            assert new_identity['pty_instance'] == identity['pty_instance']
                            assert new_identity['connection_generation'] > identity['connection_generation']
                            second.send_text(PREFIX + json.dumps({**request, 'request_id': 'two', 'expected': new_identity}))
                            assert node.receive_json()['request_id'] == 'two'
                # Credential cannot authorize a controller for a made-up PTY.
                with client.websocket_connect('/api/pub?' + urlencode({**auth, 'channel': 'original', 'controller': 'guess', 'controller_generation': 3})) as rogue:
                    with pytest.raises(WebSocketDisconnect):
                        rogue.receive_json()
        finally:
            client.portal.call(registry.close_all)


def test_controller_that_dials_during_spawn_gets_retryable_close(monkeypatch):
    import hermes_cli.web_server as server
    from hermes_cli.pty_draft_control import PtyDraftControl
    from hermes_cli.web_routers.chat_ws import router

    monkeypatch.setattr(server, '_DASHBOARD_EMBEDDED_CHAT_ENABLED', True)
    control = PtyDraftControl('launching')
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        with client.websocket_connect('/api/pub?' + urlencode({
            'token': server._SESSION_TOKEN, 'channel': 'launching',
            'controller': control.credential, 'controller_generation': 1,
        })) as node:
            with pytest.raises(WebSocketDisconnect) as closed:
                node.receive_json()
            assert closed.value.code == 1013


@pytest.mark.asyncio
async def test_cancelled_pub_claim_releases_controller_ownership(monkeypatch):
    import hermes_cli.web_server as server
    from hermes_cli.pty_draft_control import PtyDraftControl
    from hermes_cli.pty_session import PtySession
    from hermes_cli.web_routers.chat_ws import router

    monkeypatch.setattr(server, '_DASHBOARD_EMBEDDED_CHAT_ENABLED', True)
    control = PtyDraftControl('claim-cancelled')
    session = PtySession('key', SimpleNamespace(close=lambda: None),
                         buffer_cap=1024, read_timeout=.01, draft=control)
    app = FastAPI()
    app.include_router(router)
    incoming = asyncio.Queue()
    await incoming.put({'type': 'websocket.connect'})
    refreshing, release = asyncio.Event(), asyncio.Event()

    async def send(message):
        if message.get('text') and json.loads(message['text'])['type'] == 'draft.refresh':
            refreshing.set()
            await release.wait()

    scope = {'type': 'websocket', 'path': '/api/pub', 'root_path': '',
             'scheme': 'ws', 'headers': [(b'host', b'127.0.0.1')],
             'client': ('127.0.0.1', 5000), 'server': ('127.0.0.1', 80),
             'query_string': urlencode({'token': server._SESSION_TOKEN, 'channel': control.channel,
                                        'controller': control.credential, 'controller_generation': 1}).encode()}
    task = asyncio.create_task(app(scope, incoming.get, send))
    try:
        await asyncio.wait_for(refreshing.wait(), 3)
        assert control.controller is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert control.controller is None
        assert not control.pending and control.state is None
    finally:
        release.set()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await session.close()
