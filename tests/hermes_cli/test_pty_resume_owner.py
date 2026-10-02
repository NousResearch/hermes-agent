"""A dashboard resume must reattach its live PTY, not compete for its session lease."""

import json
import sys
from contextlib import asynccontextmanager
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server
from hermes_cli import web_server_chat as chat
from hermes_cli.pty_session import PtySessionRegistry
from hermes_cli.web_routers.chat_ws import router


pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def live_pty_client(tmp_path, monkeypatch):
    """Real WS/registry/PTY/file I/O; replace only the expensive TUI launcher."""
    registry = PtySessionRegistry(ttl=60, max_sessions=8, buffer_cap=4096, read_timeout=0.02)
    monkeypatch.setattr(chat, "PTY_REGISTRY", registry)
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    # Keep the real gate: a token is required even on this in-process test server.
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    homes = {}
    for profile in ("a", "b"):
        home = tmp_path / profile
        home.mkdir()
        homes[profile] = home
    monkeypatch.setenv("HERMES_HOME", str(homes["a"]))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    script = tmp_path / "terminal.py"
    script.write_text(
        "import json, os, sys, tty\n"
        "from pathlib import Path\n"
        "tty.setraw(0)\n"
        "sid = os.environ.get('HERMES_TUI_RESUME', 'session-a')\n"
        "Path(os.environ['HERMES_TUI_ACTIVE_SESSION_FILE']).write_text(json.dumps({'session_id': sid}))\n"
        "os.write(1, b'READY\\n')\n"
        "while True:\n"
        "    data = os.read(0, 4096)\n"
        "    if not data: break\n"
        "    if data.startswith(b'SWITCH:'):\n"
        "        sid = data.decode().strip().split(':', 1)[1]\n"
        "        Path(os.environ['HERMES_TUI_ACTIVE_SESSION_FILE']).write_text(json.dumps({'session_id': sid}))\n"
        "    os.write(1, data)\n",
        encoding="utf-8",
    )

    async def resolve(**kwargs):
        env = {
            "HERMES_HOME": str(homes[kwargs.get("profile") or "a"]),
            "HERMES_RUNTIME_DIR": str(tmp_path / "runtime"),
            "HERMES_TUI_ACTIVE_SESSION_FILE": kwargs["active_session_file"],
        }
        if kwargs.get("resume"):
            env["HERMES_TUI_RESUME"] = "session-a" if kwargs["resume"] == "parent-a" else kwargs["resume"]
        return [sys.executable, "-S", str(script)], str(tmp_path), env

    monkeypatch.setattr(chat, "_resolve_chat_argv_async", resolve)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await registry.close_all()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client, registry


def _url(*, token=None, **params):
    return "ws://127.0.0.1/api/pty?" + urlencode({"token": web_server._SESSION_TOKEN if token is None else token, **params})


def _receive_until(ws, marker):
    output = b""
    while marker not in output:
        message = ws.receive()
        if "bytes" in message:
            output += message["bytes"]
        else:
            assert "text" in message, message
            assert json.loads(message["text"])["type"] == "pty.attached"
    return output


@pytest.mark.parametrize("initial_resume", [None, "session-a", "parent-a"])
@pytest.mark.parametrize("profile", ["", "a"])
def test_switching_sessions_preserves_live_owner(live_pty_client, initial_resume, profile):
    client, registry = live_pty_client
    initial = {"resume": initial_resume} if initial_resume else {}
    with client.websocket_connect(_url(attach="tab", channel="first", profile=profile, **initial)) as ws:
        _receive_until(ws, b"READY\n")
        owner = next(iter(registry._sessions.values()))
        pid = owner.bridge.pid
        ws.send_bytes(b"before-switch\n")
        _receive_until(ws, b"before-switch\n")
    with client.websocket_connect(_url(attach="tab", channel="second", profile=profile, resume="session-b")) as ws:
        _receive_until(ws, b"READY\n")
        ws.send_bytes(b"other-session\n")
        _receive_until(ws, b"other-session\n")
    # React generates a new channel when the selected resume target changes.
    with client.websocket_connect(_url(attach="tab", channel="return", profile=profile, resume="session-a")) as ws:
        replay = _receive_until(ws, b"READY\n")
        assert len(registry._sessions) == 2, "resume spawned a second owner for session-a"
        assert owner.attached and owner.bridge.pid == pid and owner.bridge.is_alive()
        ws.send_bytes(b"after-switch\n")
        replay += _receive_until(ws, b"after-switch\n")
        assert b"before-switch" in replay
        assert b"other-session" not in replay
    assert not owner.attached, "alias detach must release the original registry entry"
    assert owner.last_detached_at is not None
    client.portal.call(registry.reap_idle, owner.last_detached_at + 61)
    assert not registry._sessions
    assert not owner.bridge.is_alive()


@pytest.mark.parametrize(
    "other_profile,other_tab,owner_survives",
    [("b", "tab", False), ("a", "rotated-tab", True)],
    ids=["other-profile-same-tab", "same-profile-rotated-tab"],
)
def test_resume_owner_is_scoped_and_still_requires_auth(live_pty_client, other_profile, other_tab, owner_survives):
    client, registry = live_pty_client
    with client.websocket_connect(_url(attach="tab", channel="first", profile="a")) as ws:
        _receive_until(ws, b"READY\n")
        owner = next(iter(registry._sessions.values()))

    # Knowing a tab token and session id is not authorization to reattach it.
    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect(_url(token="wrong", attach="tab", channel="denied", profile="a", resume="session-a")):
            pytest.fail("unauthenticated attach accepted")
    assert denied.value.code == 4401
    assert not owner.attached and owner.bridge.is_alive()
    assert len(registry._sessions) == 1

    # Another profile (even with the same sid), or a rotated tab identity,
    # must get its own terminal, never the fresh owner's alias. A profile switch
    # on the same tab additionally supersedes the previous profile's terminal
    # (upstream close_other_sessions); a rotated tab leaves the owner alive.
    with client.websocket_connect(_url(attach=other_tab, channel="other", profile=other_profile, resume="session-a")) as ws:
        _receive_until(ws, b"READY\n")
        assert len(registry._sessions) == (2 if owner_survives else 1)
        assert not owner.attached
        assert owner.bridge.is_alive() is owner_survives
        ws.send_bytes(b"other-scope\n")
        _receive_until(ws, b"other-scope\n")
    with client.websocket_connect(_url(attach="tab", channel="return", profile="a", resume="session-a")) as ws:
        replay = _receive_until(ws, b"READY\n")
        if owner_survives:
            assert owner.attached and len(registry._sessions) == 2
        else:
            # The superseded owner is gone, and returning to profile a supersedes
            # profile b's terminal in turn: one fresh terminal remains.
            assert not owner.attached and owner.key not in registry._sessions
            assert len(registry._sessions) == 1
        ws.send_bytes(b"original-scope\n")
        replay += _receive_until(ws, b"original-scope\n")
        assert b"other-scope" not in replay


def test_explicit_resume_does_not_reuse_a_terminal_that_changed_conversation(live_pty_client):
    client, registry = live_pty_client
    with client.websocket_connect(_url(attach="tab", channel="first", resume="session-a")) as ws:
        _receive_until(ws, b"READY\n")
        moved_owner = next(iter(registry._sessions.values()))
        ws.send_bytes(b"SWITCH:session-b\n")
        _receive_until(ws, b"SWITCH:session-b\n")

    # The owner is alive, but its original exact registry key no longer names
    # the conversation it displays. Keep B alive while resuming the requested A.
    with client.websocket_connect(_url(attach="tab", channel="return-a", resume="session-a")) as ws:
        replay = _receive_until(ws, b"READY\n")
        assert not moved_owner.attached, "explicit A incorrectly reattached the terminal now displaying B"
        assert moved_owner.bridge.is_alive() and len(registry._sessions) == 2
        ws.send_bytes(b"input-for-a\n")
        replay += _receive_until(ws, b"input-for-a\n")
        assert b"SWITCH:session-b" not in replay

    with client.websocket_connect(_url(attach="tab", channel="return-b", resume="session-b")) as ws:
        replay = _receive_until(ws, b"READY\n")
        assert moved_owner.attached and len(registry._sessions) == 2
        ws.send_bytes(b"input-for-b\n")
        replay += _receive_until(ws, b"input-for-b\n")
        assert b"SWITCH:session-b" in replay and b"input-for-a" not in replay
    client.portal.call(registry.reap_idle, moved_owner.last_detached_at + 61)
    assert not registry._sessions and not moved_owner.bridge.is_alive()
