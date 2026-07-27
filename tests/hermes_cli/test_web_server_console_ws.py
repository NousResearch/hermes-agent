"""Dashboard Hermes Console websocket tests."""

from __future__ import annotations

import time
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server


@pytest.fixture
def console_client(monkeypatch, _isolate_hermes_home):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    previous_bound_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

    client = TestClient(web_server.app)
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
        if previous_auth_required is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous_auth_required
        if previous_bound_host is None:
            if hasattr(web_server.app.state, "bound_host"):
                delattr(web_server.app.state, "bound_host")
        else:
            web_server.app.state.bound_host = previous_bound_host


def _url(token: str | None = None, **params: str) -> str:
    query = {"token": web_server._SESSION_TOKEN, **params}
    if token is not None:
        query["token"] = token
    return f"/api/console?{urlencode(query)}"


def _recv_until(conn, frame_type: str, *, status: str | None = None) -> dict:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        frame = conn.receive_json()
        if frame.get("type") != frame_type:
            continue
        if status is not None and frame.get("status") != status:
            continue
        return frame
    raise AssertionError(f"Timed out waiting for {frame_type} frame")


def test_console_ws_rejects_missing_or_bad_token(console_client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with console_client.websocket_connect("/api/console"):
            pass
    assert exc.value.code == 4401

    with pytest.raises(WebSocketDisconnect) as exc:
        with console_client.websocket_connect(_url(token="wrong")):
            pass
    assert exc.value.code == 4401


def test_console_ws_runs_read_only_command(console_client):
    with console_client.websocket_connect(_url()) as conn:
        ready = conn.receive_json()
        assert ready["type"] == "ready"
        assert ready["prompt"] == "hermes> "

        conn.send_json({"type": "input", "line": "help"})

        output = _recv_until(conn, "output")
        assert "Hermes Console" in output["data"]
        complete = _recv_until(conn, "complete", status="ok")
        assert complete["prompt"] == "hermes> "


def test_console_ws_confirmed_command_executes_after_confirmation(console_client):
    from hermes_cli.config import load_config

    with console_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "ready"
        conn.send_json({"type": "input", "line": "config set display.interface cli"})

        confirmation = _recv_until(conn, "confirm_required")
        assert confirmation["command"] == "config set display.interface cli"
        assert confirmation["message"]

        conn.send_json({"type": "confirm", "command": confirmation["command"]})
        _recv_until(conn, "complete", status="ok")

    assert load_config()["display"]["interface"] == "cli"


def test_console_ws_cancel_returns_to_prompt(console_client, monkeypatch):
    from hermes_cli.console_engine import ConsoleResult, HermesConsoleEngine

    def slow_execute(self, line: str, *, confirmed: bool = False):
        time.sleep(0.5)
        return ConsoleResult("ok", output="late", command=line)

    monkeypatch.setattr(HermesConsoleEngine, "execute", slow_execute)

    with console_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "ready"
        conn.send_json({"type": "input", "line": "status"})
        conn.send_json({"type": "cancel"})

        complete = _recv_until(conn, "complete", status="cancelled")
        assert complete["prompt"] == "hermes> "
