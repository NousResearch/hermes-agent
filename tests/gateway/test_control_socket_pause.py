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


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket transport")
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


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket transport")
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
