"""Tests for the loopback Desktop dashboard candidate handoff."""

from __future__ import annotations

import json
import socket
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


class _Transport:
    def __init__(self, host: str):
        self.host = host

    def get_extra_info(self, name: str):
        return (self.host, 0) if name == "peername" else None


@pytest.mark.asyncio
async def test_dashboard_candidates_returns_live_serve_ports(tmp_path, monkeypatch):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        port = listener.getsockname()[1]
        (tmp_path / "spawn-ledger.json").write_text(
            json.dumps([
                {"purpose": "serve", "port": port},
                {"purpose": "mcp-helper", "port": port},
                {"purpose": "serve", "port": port + 1},
            ]),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        request = SimpleNamespace(
            remote="127.0.0.1",
            transport=_Transport("127.0.0.1"),
            method="GET",
            path_qs="/api/desktop/dashboard-candidates",
            headers={},
        )

        response = await adapter._handle_dashboard_candidates(request)

        assert response.status == 200
        assert json.loads(response.text) == {"candidates": [port]}
    finally:
        listener.close()


@pytest.mark.asyncio
async def test_dashboard_candidates_requires_auth_off_loopback(tmp_path, monkeypatch):
    (tmp_path / "spawn-ledger.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "k" * 32}))
    request = SimpleNamespace(
        remote="10.0.0.8",
        transport=_Transport("10.0.0.8"),
        method="GET",
        path_qs="/api/desktop/dashboard-candidates",
        headers={},
    )

    response = await adapter._handle_dashboard_candidates(request)

    assert response.status == 401
