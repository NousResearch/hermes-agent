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


def test_console_ws_cancel_returns_to_prompt(console_client, monkeypatch):
    from hermes_cli.console_engine import ConsoleResult, HermesConsoleEngine

    def slow_execute(self, line: str, *, confirmed: bool = False):
        time.sleep(0.2)
        return ConsoleResult("ok", output="late", command=line)

    monkeypatch.setattr(HermesConsoleEngine, "execute", slow_execute)

    with console_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "ready"
        conn.send_json({"type": "input", "line": "status"})
        conn.send_json({"type": "cancel"})

        complete = _recv_until(conn, "complete", status="cancelled")
        assert complete["prompt"] == "hermes> "


@pytest.fixture
def blocking_provider():
    """Loopback OpenAI-compatible server whose chat completion blocks until the peer closes
    the socket (like a llama.cpp generation) or the test releases it."""
    import json
    import select
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    state = {"started": threading.Event(), "peer_closed": threading.Event(), "release": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if not self.path.endswith("/chat/completions"):  # capability probes
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            state["started"].set()
            deadline = time.monotonic() + 30
            while not state["release"].is_set() and time.monotonic() < deadline:
                readable, _, _ = select.select([self.connection], [], [], 0.05)
                if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                    state["peer_closed"].set()
                    return
            body = json.dumps({"id": "x", "object": "chat.completion", "model": "test-model", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["base_url"] = f"http://127.0.0.1:{srv.server_port}/v1"
    try:
        yield state
    finally:
        state["release"].set()
        srv.shutdown()


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_console_cancel_stops_forked_agent_request_before_reporting(console_client, monkeypatch, blocking_provider, stop):
    """#106179: cancelling (or timing out) a console command whose worker forked an AIAgent must interrupt
    that agent — closing its in-flight provider request — and wait for the worker to exit BEFORE the
    prompt reports cancelled/timeout. asyncio can only drop the waiter; the thread keeps decoding otherwise."""
    import threading

    from agent import curator
    from hermes_cli.web_routers import chat_ws

    monkeypatch.setattr(
        curator, "_resolve_review_provider",
        lambda: ({"api_key": "test-key", "base_url": blocking_provider["base_url"]}, "test-model", "openai-compat", {}),
    )
    worker_exited = threading.Event()
    real_execute = chat_ws._execute_console_line

    def observed_execute(*args, **kwargs):
        try:
            return real_execute(*args, **kwargs)
        finally:
            worker_exited.set()

    monkeypatch.setattr(chat_ws, "_execute_console_line", observed_execute)
    if stop == "timeout":
        monkeypatch.setattr(chat_ws, "_CONSOLE_COMMAND_TIMEOUT_SECONDS", 2.0)
    line = "curator run --consolidate --dry-run"

    with console_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "ready"
        conn.send_json({"type": "input", "line": line})
        _recv_until(conn, "complete", status="confirm_required")
        assert worker_exited.wait(10)  # the confirm probe's worker, not the one under test
        worker_exited.clear()
        conn.send_json({"type": "confirm", "command": line})
        assert blocking_provider["started"].wait(60), "forked agent never reached the provider"
        if stop == "cancel":
            conn.send_json({"type": "cancel"})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            frame = conn.receive_json()
            if frame.get("type") == "complete" and frame.get("status") in {"cancelled", "timeout"}:
                break
        else:
            raise AssertionError("no cancelled/timeout frame")
        observed = (frame["status"], blocking_provider["peer_closed"].is_set(), worker_exited.is_set())
        blocking_provider["release"].set()  # a leaked worker (the bug) must not wedge socket teardown
        worker_exited.wait(30)
    assert observed == ("cancelled" if stop == "cancel" else "timeout", True, True), (
        "(status, provider request closed, worker exited) at the terminal frame")


def test_interrupt_scope_cancels_agents_that_start_after_the_cancel():
    """A turn that begins after the host cancelled must be interrupted on entry, else a cancel racing
    agent construction leaves a live request behind."""
    from agent.interrupt_scope import InterruptScope, bind_interrupt_scope, track_in_interrupt_scope

    class Agent:
        def __init__(self):
            self.stops = []

        def hard_interrupt(self, message=None, *, tool_reason=None):
            self.stops.append(message)

    scope = InterruptScope()
    early, late_agent, unscoped = Agent(), Agent(), Agent()
    with bind_interrupt_scope(scope):
        with track_in_interrupt_scope(early):
            scope.cancel("Console command cancelled")
            with track_in_interrupt_scope(late_agent):
                pass
    with track_in_interrupt_scope(unscoped):  # no scope bound: nothing to register with
        scope.cancel("Console command cancelled")
    assert early.stops == ["Console command cancelled"]
    assert late_agent.stops == ["Console command cancelled"]
    assert unscoped.stops == []
