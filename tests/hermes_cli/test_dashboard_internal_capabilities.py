"""Route-level regression coverage for dashboard internal WS capabilities."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server
from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests


@pytest.fixture
def gated_ws_client(monkeypatch):
    previous = {
        "auth_required": getattr(web_server.app.state, "auth_required", None),
        "bound_host": getattr(web_server.app.state, "bound_host", None),
        "bound_port": getattr(web_server.app.state, "bound_port", None),
    }
    _reset_for_tests()
    monkeypatch.setattr(web_server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    web_server.app.state.auth_required = True
    web_server.app.state.bound_host = "testserver"
    web_server.app.state.bound_port = 80

    with TestClient(web_server.app, base_url="http://testserver") as client:
        yield client

    _reset_for_tests()
    for key, value in previous.items():
        setattr(web_server.app.state, key, value)


def _assert_rejected(client: TestClient, path: str, *, code: int = 4401) -> None:
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(path):
            pass
    assert exc_info.value.code == code


def test_sidecar_capability_cannot_open_broader_dashboard_routes(gated_ws_client):
    sidecar_url = web_server._build_sidecar_url("lane-a", profile="worker")
    assert sidecar_url is not None
    capability = parse_qs(urlsplit(sidecar_url).query)["internal"][0]

    _assert_rejected(gated_ws_client, f"/api/ws?internal={capability}")
    _assert_rejected(
        gated_ws_client,
        f"/api/pty?internal={capability}&channel=lane-a&profile=worker",
    )
    _assert_rejected(
        gated_ws_client,
        f"/api/events?internal={capability}&channel=lane-a",
    )
    _assert_rejected(gated_ws_client, f"/api/console?internal={capability}")


def test_gateway_capability_cannot_open_other_dashboard_routes(gated_ws_client):
    gateway_url = web_server._build_gateway_ws_url()
    assert gateway_url is not None
    capability = parse_qs(urlsplit(gateway_url).query)["internal"][0]

    for path in (
        f"/api/pub?internal={capability}&channel=lane-a&profile=worker",
        f"/api/pty?internal={capability}&channel=lane-a&profile=worker",
        f"/api/events?internal={capability}&channel=lane-a",
        f"/api/console?internal={capability}",
    ):
        _assert_rejected(gated_ws_client, path)


@pytest.mark.parametrize(
    ("channel", "profile"),
    [("lane-b", "worker"), ("lane-a", "other")],
)
def test_sidecar_capability_rejects_binding_tampering(
    gated_ws_client, channel: str, profile: str
):
    sidecar_url = web_server._build_sidecar_url("lane-a", profile="worker")
    assert sidecar_url is not None
    capability = parse_qs(urlsplit(sidecar_url).query)["internal"][0]

    _assert_rejected(
        gated_ws_client,
        f"/api/pub?internal={capability}&channel={channel}&profile={profile}",
    )


def test_matching_sidecar_capability_can_reconnect(gated_ws_client):
    sidecar_url = web_server._build_sidecar_url("lane-a", profile="worker")
    assert sidecar_url is not None
    path = urlsplit(sidecar_url).path + "?" + urlsplit(sidecar_url).query

    for _ in range(2):
        with gated_ws_client.websocket_connect(path) as ws:
            ws.close()
