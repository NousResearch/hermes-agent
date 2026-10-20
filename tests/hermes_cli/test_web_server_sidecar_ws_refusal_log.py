"""Every pre-accept sidecar WS refusal must log its reason (#138057).

``/api/ws``, ``/api/pub`` and ``/api/events`` share
``_close_unless_sidecar_allowed``, which closes the upgrade *before*
``accept()`` — the browser only reports a generic 1006, so the structured
log line is the only diagnostic a user has. The console/pty gate
(``_ws_gate``) already logs every reason; these tests pin the same
contract for the sidecar guard.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from hermes_cli import web_server
from hermes_cli.web_routers import chat_ws


class _RecordingWS(SimpleNamespace):
    """WebSocket stand-in good enough for the pre-accept guard."""

    def __init__(self, *, query=None, client_host="127.0.0.1", path="/api/events", headers=None):
        query = query or {}

        class _QP:
            def get(self, key, default=""):
                return query.get(key, default)

        super().__init__(
            query_params=_QP(),
            headers=headers or {},
            client=SimpleNamespace(host=client_host) if client_host else None,
            url=SimpleNamespace(path=path),
        )
        self.closed = []

    async def close(self, code=1000, reason=None):
        self.closed.append(code)


@pytest.fixture
def loopback_app(monkeypatch):
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.auth_required = False
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    yield
    web_server.app.state.bound_host = prev_host
    web_server.app.state.auth_required = prev_required


def _run_guard(ws: _RecordingWS) -> bool:
    return asyncio.run(chat_ws._close_unless_sidecar_allowed(ws))


def test_missing_credential_logs_reason_and_closes_4401(loopback_app, caplog):
    ws = _RecordingWS(query={}, headers={"host": "127.0.0.1:8080"})
    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        assert _run_guard(ws) is False
    assert ws.closed == [4401]
    assert "auth rejected reason=no_credential" in caplog.text
    assert "/api/events" in caplog.text


def test_host_mismatch_logs_reason_and_closes_4403(loopback_app, caplog):
    ws = _RecordingWS(
        query={"token": web_server._SESSION_TOKEN},
        headers={"host": "evil.example:9999", "origin": "https://evil.example:9999"},
    )
    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        assert _run_guard(ws) is False
    assert ws.closed == [4403]
    assert "host_mismatch" in caplog.text


def test_non_loopback_peer_logs_reason_and_closes_4403(loopback_app, caplog):
    ws = _RecordingWS(
        query={"token": web_server._SESSION_TOKEN},
        client_host="192.168.1.42",
        headers={"host": "127.0.0.1:8080"},
    )
    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        assert _run_guard(ws) is False
    assert ws.closed == [4403]
    assert "peer_not_loopback" in caplog.text


def test_disabled_chat_logs_and_closes_4403(loopback_app, caplog, monkeypatch):
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", False)
    ws = _RecordingWS(query={"token": web_server._SESSION_TOKEN})
    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        assert _run_guard(ws) is False
    assert ws.closed == [4403]
    assert "embedded chat disabled" in caplog.text


def test_allowed_upgrade_logs_no_refusal(loopback_app, caplog):
    ws = _RecordingWS(
        query={"token": web_server._SESSION_TOKEN},
        headers={"host": "127.0.0.1:8080"},
    )
    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        assert _run_guard(ws) is True
    assert ws.closed == []
    assert "refused" not in caplog.text
    assert "auth rejected" not in caplog.text
