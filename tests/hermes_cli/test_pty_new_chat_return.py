"""New-chat terminal rotation must not rotate the browser's resume scope.

Real authenticated WS routing, PTY subprocesses, registry and breadcrumb I/O;
only the expensive Ink/agent launcher is replaced by a raw echo child.
"""
import importlib.util
from pathlib import Path

import pytest
from starlette.websockets import WebSocketDisconnect

spec = importlib.util.spec_from_file_location("owner_fixture", Path(__file__).with_name("test_pty_resume_owner.py"))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
live_pty_client = fixture_module.live_pty_client
_url = fixture_module._url
_receive_until = fixture_module._receive_until
pytestmark = pytest.mark.platforms("posix")


@pytest.mark.parametrize("picked", [False, True], ids=["default-workspace", "selected-workspace"])
@pytest.mark.parametrize("profile", ["", "a"])
@pytest.mark.parametrize("initial_resume", [None, "session-a", "parent-a"])
def test_return_after_new_chat(live_pty_client, tmp_path, monkeypatch, picked, profile, initial_resume):
    client, registry = live_pty_client
    resolve = fixture_module.chat._resolve_chat_argv_async
    fresh_ids = iter(["session-a", "session-b"] if initial_resume is None else ["session-b"])
    resolutions = []

    async def distinct_fresh_id(**kwargs):
        resolutions.append(kwargs)
        argv, cwd, env = await resolve(**kwargs)
        if not kwargs.get("resume"):
            env["HERMES_TUI_RESUME"] = next(fresh_ids)
        return argv, cwd, env

    monkeypatch.setattr(fixture_module.chat, "_resolve_chat_argv_async", distinct_fresh_id)
    workspace = tmp_path / "Project With Spaces"
    workspace.mkdir()
    initial = {"resume": initial_resume} if initial_resume else {}
    # No tab parameter on A also verifies upgrading an already-open legacy PTY.
    with client.websocket_connect(_url(attach="original-tab-token", channel="original", profile=profile, **initial)) as ws:
        _receive_until(ws, b"READY\n")
        owner = next(iter(registry._sessions.values()))
        pid = owner.bridge.pid
        ws.send_bytes(b"only-a\n")
        _receive_until(ws, b"only-a\n")
    params = {"fresh": "1", "profile": profile}
    if picked:
        params["cwd"] = str(workspace)
    identity = {"attach": "new-chat-token", "tab": "original-tab-token", "profile": profile}
    with client.websocket_connect(_url(**{**identity, **params}, channel="new")) as ws:
        _receive_until(ws, b"READY\n")
        assert not owner.attached and len(registry._sessions) == 2
        other = next(s for s in registry._sessions.values() if s is not owner)
        ws.send_bytes(b"only-b\n")
        _receive_until(ws, b"only-b\n")
    assert resolutions[-1].get("workspace_cwd") == (str(workspace) if picked else None)
    with client.websocket_connect(_url(**identity, channel="return", resume="session-a", cwd=str(workspace))) as ws:
        replay = _receive_until(ws, b"READY\n")
        assert owner.bridge.is_alive() and owner.bridge.pid == pid
        assert owner.attached and len(registry._sessions) == 2
        assert "workspace_cwd" not in resolutions[-1]
        ws.send_bytes(b"after-return\n")
        replay += _receive_until(ws, b"after-return\n")
        assert b"only-a" in replay and b"only-b" not in replay
    assert not owner.attached and owner.last_detached_at is not None
    with client.websocket_connect(_url(**identity, channel="return-b", resume="session-b")) as ws:
        replay = _receive_until(ws, b"READY\n")
        assert other.attached and len(registry._sessions) == 2
        ws.send_bytes(b"b-again\n")
        replay += _receive_until(ws, b"b-again\n")
        assert b"only-b" in replay and b"only-a" not in replay
    client.portal.call(registry.reap_idle, other.last_detached_at + 61)
    assert not registry._sessions
    assert not owner.bridge.is_alive() and not other.bridge.is_alive()


@pytest.mark.parametrize("other_profile,other_tab", [("b", "tab"), ("a", "other-tab")])
def test_rotated_protocol_keeps_auth_profile_and_tab_boundaries(live_pty_client, other_profile, other_tab):
    client, registry = live_pty_client
    with client.websocket_connect(_url(attach="terminal-a", tab="tab", profile="a", channel="original")) as ws:
        _receive_until(ws, b"READY\n")
        owner = next(iter(registry._sessions.values()))
    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect(_url(token="wrong", attach="terminal-b", tab="tab", profile="a", channel="denied", resume="session-a")):
            pytest.fail("unauthenticated attach accepted")
    assert denied.value.code == 4401
    assert len(registry._sessions) == 1 and not owner.attached
    with client.websocket_connect(_url(attach="terminal-b", tab=other_tab, profile=other_profile, channel="other", resume="session-a")) as ws:
        _receive_until(ws, b"READY\n")
        assert not owner.attached and len(registry._sessions) == 2
    with client.websocket_connect(_url(attach="terminal-b", tab="tab", profile="a", channel="return", resume="session-a")) as ws:
        _receive_until(ws, b"READY\n")
        assert owner.attached and len(registry._sessions) == 2
