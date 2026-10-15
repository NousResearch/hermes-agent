"""The browser terminal uses backend authorization and preserves sandbox policy."""
from __future__ import annotations

import pytest
from starlette.websockets import WebSocketDisconnect

from tests.hermes_cli.test_browser_frontend import browser_app as browser_app, _login  # noqa: PLC0414 — pytest fixture re-export


def _terminal_url(client):
    _login(client)
    ticket = client.post('/api/auth/ws-ticket', headers={'Origin': 'https://browser.example.test'}).json()['ticket']
    return f'wss://browser.example.test/api/browser/terminal?ticket={ticket}'


def test_terminal_rejects_missing_credentials_and_cross_origin(browser_app):
    for headers in ({'Origin': 'https://browser.example.test'}, {'Origin': 'https://evil.example.test'}):
        with pytest.raises(WebSocketDisconnect) as exc:
            with browser_app.websocket_connect('wss://browser.example.test/api/browser/terminal', headers=headers):
                pass
        assert exc.value.code == 1008


@pytest.mark.platforms('posix')
def test_terminal_streams_a_real_shell_and_closes_on_disconnect(browser_app, tmp_path, monkeypatch):
    monkeypatch.setenv('SHELL', '/bin/sh')
    from hermes_cli.web_routers import browser_terminal
    from hermes_cli import config_effective
    monkeypatch.setattr(config_effective, 'load_user_config_effective', lambda: {'terminal': {'backend': 'local', 'cwd': str(tmp_path)}})
    url = _terminal_url(browser_app)
    with browser_app.websocket_connect(url, headers={'Origin': 'https://browser.example.test'}) as ws:
        ready = ws.receive_json()
        assert ready['type'] == 'ready'
        assert ready['cwd'] == str(tmp_path)
        # The command contains octal escapes, not the expected output marker,
        # so terminal echo alone cannot make this assertion pass.
        ws.send_json({'type': 'write', 'data': "printf '\\102\\122\\117\\127\\123\\105\\122\\137\\117\\113\\n'\r"})
        output = ''
        for _ in range(20):
            message = ws.receive_json()
            assert message['type'] == 'data'
            output += message['data']
            if 'BROWSER_OK' in output:
                break
        assert 'BROWSER_OK' in output
        ws.send_json({'type': 'resize', 'cols': 100, 'rows': 30})
    assert browser_terminal._active_terminals == 0


@pytest.mark.platforms('posix')
def test_terminal_never_falls_back_from_sandbox_to_host_shell(browser_app, monkeypatch):
    from hermes_cli import config_effective
    monkeypatch.setattr(config_effective, 'load_user_config_effective', lambda: {'terminal': {'backend': 'docker'}})
    url = _terminal_url(browser_app)
    with browser_app.websocket_connect(url, headers={'Origin': 'https://browser.example.test'}) as ws:
        message = ws.receive_json()
        assert message['type'] == 'error'
        assert 'local terminal backend' in message['error']


def test_terminal_reserves_capacity_before_awaiting_accept(monkeypatch):
    import asyncio
    from hermes_cli.web_routers import browser_terminal as module, status
    from hermes_cli import web_server_chat
    monkeypatch.setattr(module, '_active_terminals', 0)
    monkeypatch.setattr(status, '_frontend_identity', lambda: 'desktop')
    monkeypatch.setattr(web_server_chat, '_ws_request_is_allowed', lambda ws: True)
    monkeypatch.setattr(web_server_chat, '_ws_auth_ok', lambda ws: True)

    class WaitingSocket:
        headers = {'origin': 'https://browser.example.test', 'host': 'browser.example.test'}
        closed = None
        async def accept(self):
            await asyncio.Event().wait()
        async def close(self, code=None):
            self.closed = code

    async def probe():
        sockets = [WaitingSocket() for _ in range(9)]
        tasks = [asyncio.create_task(module.browser_terminal(ws)) for ws in sockets]
        await asyncio.sleep(0)  # each handler advances to its first await
        assert module._active_terminals == 8
        assert sockets[-1].closed == 1013
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert module._active_terminals == 0

    asyncio.run(probe())


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('startup_fails', [False, True])
def test_terminal_owns_in_flight_spawn_during_repeated_cancellation(monkeypatch, tmp_path, startup_fails):
    """Raw asyncio cancellation must not orphan a shell or release its slot early."""
    import asyncio
    import threading
    from hermes_cli.pty_bridge import PtyBridge
    from hermes_cli.web_routers import browser_terminal as module, status
    from hermes_cli import web_server_chat

    monkeypatch.setattr(module, '_active_terminals', 0)
    monkeypatch.setattr(status, '_frontend_identity', lambda: 'desktop')
    monkeypatch.setattr(web_server_chat, '_ws_request_is_allowed', lambda ws: True)
    monkeypatch.setattr(web_server_chat, '_ws_auth_ok', lambda ws: True)
    monkeypatch.setattr(module, '_terminal_launch', lambda *args: (['/bin/sh'], str(tmp_path), {}))
    original_spawn = PtyBridge.spawn
    original_close = PtyBridge.close
    release_spawn = threading.Event()
    release_close = threading.Event()
    acquired = []

    class Socket:
        headers = {'origin': 'https://browser.example.test', 'host': 'browser.example.test'}
        query_params = {}
        closed = False
        async def accept(self):
            pass
        async def send_json(self, message):
            pytest.fail('A cancelled startup must not send a ready message')
        async def close(self, code=None):
            self.closed = True

    async def probe():
        loop = asyncio.get_running_loop()
        spawned = asyncio.Event()
        closing = asyncio.Event()

        def delayed_spawn(*args, **kwargs):
            if not startup_fails:
                acquired.append(original_spawn(*args, **kwargs))
            loop.call_soon_threadsafe(spawned.set)
            assert release_spawn.wait(10), 'Test did not release startup'
            if startup_fails:
                raise OSError('Simulated startup failure')
            return acquired[0]

        def delayed_close(bridge):
            loop.call_soon_threadsafe(closing.set)
            try:
                assert release_close.wait(10), 'Test did not release teardown'
            finally:
                original_close(bridge)

        monkeypatch.setattr(PtyBridge, 'spawn', delayed_spawn)
        monkeypatch.setattr(PtyBridge, 'close', delayed_close)
        ws = Socket()
        task = asyncio.create_task(module.browser_terminal(ws))
        try:
            await asyncio.wait_for(spawned.wait(), 10)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()  # another cancellation while startup is still in flight
            await asyncio.sleep(0)
            assert not task.done()
            assert module._active_terminals == 1
            release_spawn.set()
            if not startup_fails:
                await asyncio.wait_for(closing.wait(), 10)
                task.cancel()  # teardown itself must also retain ownership
                await asyncio.sleep(0)
                assert not task.done()
                assert module._active_terminals == 1
            release_close.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            assert ws.closed
            assert module._active_terminals == 0
            if acquired:
                assert acquired[0]._closed
                assert not acquired[0]._proc.isalive()
        finally:
            release_spawn.set()
            release_close.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(probe())
