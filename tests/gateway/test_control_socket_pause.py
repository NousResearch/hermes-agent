"""pause-for-update control-socket verb (#92091 step 2, campaign #91277).

The updater asks a running gateway to drain and exit cleanly (releasing its
venv file handles) instead of tree-killing it mid-turn. Fallback contract:
older gateways without the verb answer nothing, and callers keep the legacy
marker/force-kill path.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.control_socket import (
    GatewayControlServer,
    pause_gateway_for_update,
    query_gateway_control,
)


def _make_server(tmp_path, handler):
    server = GatewayControlServer(
        home=tmp_path, verb_handlers={"pause-for-update": handler}
    )
    return server


def test_pause_verb_dispatches_and_returns_ack(tmp_path):
    calls = []

    def handler():
        calls.append(1)
        return {"pausing": True, "already_stopping": False, "pid": 111,
                "drain_timeout": 30.0}

    server = _make_server(tmp_path, handler)
    raw = json.dumps({"verb": "pause-for-update", "id": 7}).encode()
    response = json.loads(server.handle_request_line(raw).decode())
    assert response["ok"] is True
    assert response["result"]["pausing"] is True
    assert response["result"]["drain_timeout"] == 30.0
    assert response["id"] == 7
    assert calls == [1]


def test_unknown_verb_still_lists_pause(tmp_path):
    server = _make_server(tmp_path, lambda: {})
    raw = json.dumps({"verb": "nope"}).encode()
    response = json.loads(server.handle_request_line(raw).decode())
    assert response["ok"] is False
    assert "pause-for-update" in response["supported_verbs"]


@pytest.mark.platforms("posix")  # unix socket transport
def test_pause_client_roundtrip_over_real_socket(tmp_path):
    """Full client→socket→handler→ACK path over a REAL unix socket."""

    async def scenario():
        acks = []

        def handler():
            acks.append(1)
            return {"pausing": True, "already_stopping": False,
                    "pid": 4242, "drain_timeout": 12.5}

        server = _make_server(tmp_path, handler)
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: pause_gateway_for_update(tmp_path, timeout=5.0)
            )
        finally:
            await server.stop()
        return result, acks

    result, acks = asyncio.run(scenario())
    assert acks == [1]
    assert result is not None
    assert result["pausing"] is True and result["drain_timeout"] == 12.5


@pytest.mark.platforms("posix")  # unix socket transport
def test_pause_client_none_when_gateway_lacks_verb(tmp_path):
    """Back-compat: a step-1 gateway (identify/status only) answers ok:false
    for the unknown verb → the client returns None → caller keeps the legacy
    kill path."""

    async def scenario():
        server = GatewayControlServer(home=tmp_path)  # no pause handler
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: pause_gateway_for_update(tmp_path, timeout=5.0)
            )
        finally:
            await server.stop()

    assert asyncio.run(scenario()) is None


def test_pause_client_none_when_no_socket(tmp_path):
    assert pause_gateway_for_update(tmp_path, timeout=0.5) is None

def test_windows_update_socket_pause_owns_shutdown_before_marker(monkeypatch, tmp_path):
    """A current gateway's socket ACK must suppress the legacy marker watcher race."""
    from gateway import control_socket
    from hermes_cli import update_cmd_windows

    events = []
    timeouts = []
    proc = SimpleNamespace(profile="default", path=tmp_path)

    def pause(_home, *, timeout):
        events.append("socket")
        timeouts.append(timeout)
        return {
            "pausing": True, "already_stopping": False, "pid": 42, "drain_timeout": 2025.0,
        }

    monkeypatch.setattr(control_socket, "pause_gateway_for_update", pause)
    monkeypatch.setattr(
        update_cmd_windows,
        "_write_update_planned_stop_marker",
        lambda *_args: events.append("marker") or True,
    )

    profiles, pids, acks = update_cmd_windows._request_socket_pauses(
        [42], {42: proc}, set()
    )

    assert events == ["socket"]
    assert timeouts == [6.0]
    assert profiles == {"default": 42}
    assert pids == [42]
    assert acks[0]["drain_timeout"] == 2025.0


def test_windows_update_marker_is_only_socket_fallback(monkeypatch, tmp_path):
    """Old/no-socket gateways retain the planned-stop fallback, after the socket probe."""
    from gateway import control_socket
    from hermes_cli import update_cmd_windows

    events = []
    timeouts = []
    proc = SimpleNamespace(profile="default", path=tmp_path)

    def no_pause(_home, *, timeout):
        events.append("socket")
        timeouts.append(timeout)
        return None

    monkeypatch.setattr(control_socket, "pause_gateway_for_update", no_pause)
    monkeypatch.setattr(
        update_cmd_windows,
        "_write_update_planned_stop_marker",
        lambda *_args: events.append("marker") or True,
    )

    _profiles, _pids, acks = update_cmd_windows._request_socket_pauses(
        [42], {42: proc}, set()
    )

    assert events == ["socket", "marker"]
    assert timeouts == [6.0]
    assert acks == []


def test_pause_handler_declares_complete_restart_wait_budget(monkeypatch):
    """The ACK covers after-turn + stop, not only restart_drain_timeout (#129947)."""
    import atexit

    from gateway import control_socket
    from gateway import run as gateway_run
    from gateway import run_plugin_rewire
    from gateway import run_profile_reconcile
    from hermes_cli import gateway as gateway_cli

    captured = {}

    class FakeControlServer:
        def __init__(self, *args, verb_handlers=None, **kwargs):
            captured.update(verb_handlers or {})

        async def start(self):
            return True

        def cleanup_files(self):
            pass

    monkeypatch.setattr(control_socket, "GatewayControlServer", FakeControlServer)
    monkeypatch.setattr(atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 180.0)
    monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 2025.0)

    for name in (
        "migrate_profile_identity_verb",
        "purge_profile_identity_verb",
        "unserve_profile_verb",
        "serve_profile_verb",
    ):
        monkeypatch.setattr(
            run_profile_reconcile, name, lambda _runner: (lambda *_args, **_kwargs: {})
        )
    monkeypatch.setattr(
        run_plugin_rewire,
        "reload_plugins_verb",
        lambda _runner, _loop: (lambda *_args, **_kwargs: {}),
    )

    restart_calls = []
    runner = SimpleNamespace(
        request_restart=lambda **kwargs: restart_calls.append(kwargs) or True
    )

    async def scenario():
        await gateway_run._start_gateway_start_control_socket(runner)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, captured["pause-for-update"])

    ack = asyncio.run(scenario())

    assert ack["pausing"] is True
    assert ack["drain_timeout"] == 2025.0
    assert restart_calls == [{"detached": False, "via_service": True}]
