"""Overlapping app lifespans must keep requests and shells under a live owner."""
import asyncio
from contextlib import AsyncExitStack, ExitStack
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli.web_host_terminal_sessions import get_host_terminals, host_terminal_lifespan


@pytest.mark.asyncio
@pytest.mark.parametrize("older_first", [True, False])
async def test_overlapping_lifespans_close_only_their_owned_registry(older_first, monkeypatch):
    app = FastAPI()
    async with AsyncExitStack() as older, AsyncExitStack() as newer:
        await older.enter_async_context(host_terminal_lifespan(app))
        first = get_host_terminals(app)
        await newer.enter_async_context(host_terminal_lifespan(app))
        second = get_host_terminals(app)
        assert second is not first

        closing, remaining = (older, newer) if older_first else (newer, older)
        retired, live = (first, second) if older_first else (second, first)
        await closing.aclose()
        assert retired._closed and not live._closed
        assert get_host_terminals(app) is live

        # Restoring a registry is not enough: its lifespan must still reap it.
        reaped = asyncio.Event()
        original_reap = live.reap_idle

        async def reap_idle():
            await original_reap()
            reaped.set()

        monkeypatch.setattr(live, "reap_idle", reap_idle)
        await asyncio.wait_for(reaped.wait(), 3)
        await remaining.aclose()

    assert first._closed and second._closed
    assert not hasattr(app.state, "host_terminals")
    with pytest.raises(RuntimeError, match="not running"):
        get_host_terminals(app)
    assert not hasattr(app.state, "host_terminals")

    # A later startup gets a new owner, never a closed predecessor.
    async with host_terminal_lifespan(app):
        restarted = get_host_terminals(app)
        assert restarted is not first and restarted is not second
        assert not restarted._closed
    assert restarted._closed


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("older_first", [True, False])
def test_real_shells_remain_owned_after_overlapping_lifespan_exit(tmp_path, monkeypatch, older_first):
    import psutil

    from hermes_cli import web_host_terminal, web_server
    from hermes_cli.web_routers.webapp import router

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "localhost", raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_host_terminal, "shell_spec", lambda: (["/bin/sh", "-i"], "sh"))
    app = FastAPI(lifespan=host_terminal_lifespan)
    app.include_router(router)
    url = f"ws://localhost/api/host-terminal?token={web_server._SESSION_TOKEN}"
    bridges = []
    processes = []

    def open_shell(client):
        with client.websocket_connect(url) as ws:
            metadata = json.loads(ws.receive_text().removeprefix(web_host_terminal.META_PREFIX))
            session = get_host_terminals(app)._sessions[metadata["terminalId"]]
            bridges.append(session.bridge)
            processes.append(psutil.Process(session.bridge.pid))
            ws.send_bytes(b"printf 'LIFE%s=%s\\n' SPAN $$\n")
            output = b""
            marker = f"LIFESPAN={session.bridge.pid}".encode()
            while marker not in output:
                output += ws.receive_bytes()
            assert session.bridge.is_alive()
        return session.bridge

    try:
        with ExitStack() as older, ExitStack() as newer:
            first_client = older.enter_context(TestClient(app, base_url="http://localhost"))
            first = open_shell(first_client)
            second_client = newer.enter_context(TestClient(app, base_url="http://localhost"))
            second = open_shell(second_client)
            closing, live_client = (older, second_client) if older_first else (newer, first_client)
            retired, live = (first, second) if older_first else (second, first)
            closing.close()
            assert not retired.is_alive() and live.is_alive()
            # This request used to create a third, unowned registry when the
            # newer lifespan exited first. Its child escaped the older teardown.
            open_shell(live_client)
        assert all(not bridge.is_alive() for bridge in bridges)
        assert all(not process.is_running() for process in processes)

        # An accepted request outside a lifespan must fail, not spawn a shell.
        # TestClient without a context deliberately does not run startup.
        with TestClient(app).websocket_connect(url) as ws:
            assert "terminalId" not in ws.receive_text()
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_bytes()
            assert error.value.code == 1011
        assert not hasattr(app.state, "host_terminals")
    finally:
        # Keep the red regression safe even when a child escapes ownership.
        registry = getattr(app.state, "host_terminals", None)
        if registry is not None:
            for session in registry._sessions.values():
                session.bridge.close()
        for bridge in bridges:
            bridge.close()
