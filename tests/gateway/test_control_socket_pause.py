"""pause-for-update control-socket verb (#92091 step 2, campaign #91277).

The updater asks a running gateway to drain and exit cleanly (releasing its
venv file handles) instead of tree-killing it mid-turn. Fallback contract:
older gateways without the verb answer nothing, and callers keep the legacy
marker/force-kill path.

The ACK also carries ``restart_in_flight`` — the restart-task state snapshotted
BEFORE this request was dispatched (#135878): the updater's ``already_stopping``
splits into a refusal because a restart is already draining (wait for it) vs a
request that never landed (stop now).
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


class _RestartRecordingRunner:
    """The subset of the runner the pause handler drives: request_restart flips
    ``_restart_task_started`` synchronously (as the real one does before returning)
    and refuses once a restart already started."""

    def __init__(self):
        self._restart_task_started = False
        self.request_restart_calls = 0

    def request_restart(self, *, detached: bool = False, via_service: bool = False) -> bool:
        self.request_restart_calls += 1
        if self._restart_task_started:
            return False
        self._restart_task_started = True
        return True


def _pause_ack_over_real_socket(runner, tmp_path, monkeypatch):
    """The pause verb's ACK for ``runner`` over a REAL unix socket + executor thread.

    The production factory derives the socket home from ``HERMES_HOME`` (not a parameter), so the
    test points the env at ``tmp_path`` — every file this path touches stays in the tmp home.
    """

    async def scenario():
        from gateway.run import _start_gateway_start_control_socket
        server = await _start_gateway_start_control_socket(runner)
        assert server is not None
        assert Path(server._home) == tmp_path, "the socket must bind the isolated tmp home"
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: pause_gateway_for_update(tmp_path, timeout=5.0))
        finally:
            await server.stop()
            import atexit
            atexit.unregister(server.cleanup_files)

    return asyncio.run(scenario())


@pytest.mark.platforms("posix")  # unix socket transport
def test_fresh_drain_ack_reports_no_restart_in_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ack = _pause_ack_over_real_socket(_RestartRecordingRunner(), tmp_path, monkeypatch)
    assert ack is not None
    assert ack["pausing"] is True and ack["already_stopping"] is False
    # The snapshot precedes this request's own dispatch, so a fresh accepted drain
    # must not read as a restart already being in flight.
    assert ack["restart_in_flight"] is False


@pytest.mark.platforms("posix")  # unix socket transport
def test_refused_during_foreign_restart_reports_in_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = _RestartRecordingRunner()
    assert runner.request_restart() is True  # an earlier drain (SIGUSR1, /restart, ...) took it
    ack = _pause_ack_over_real_socket(runner, tmp_path, monkeypatch)
    assert ack is not None
    assert ack["pausing"] is False and ack["already_stopping"] is True
    # This refusal means "a restart is already draining" — the updater's wait-for-it branch (#135878).
    assert ack["restart_in_flight"] is True


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
    server = _make_server(tmp_path, dict)
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
