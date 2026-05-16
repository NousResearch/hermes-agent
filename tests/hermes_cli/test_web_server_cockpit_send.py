"""BIF-516 web endpoint tests for production Cockpit send intents."""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli import web_server

    monkeypatch.setenv("HERMES_COCKPIT_SEND_ENABLED", "1")
    monkeypatch.setenv("HERMES_COCKPIT_SEND_REAL_GATEWAY_ENABLED", "1")
    monkeypatch.setenv("HERMES_COCKPIT_SEND_KILL_SWITCH", "0")
    web_server._COCKPIT_SEND_SERVICE = None
    web_server._COCKPIT_LANE_RESOLVER = None
    yield TestClient(web_server.app)
    web_server._COCKPIT_SEND_SERVICE = None
    web_server._COCKPIT_LANE_RESOLVER = None


@pytest.fixture()
def auth_headers():
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

    return {_SESSION_HEADER_NAME: _SESSION_TOKEN}


def _assert_no_send_leaks(body):
    encoded = json.dumps(body, sort_keys=True)
    for forbidden in ("111111111111111111", "discord-message-123", "cockpit-key-web", "dashboard-user-raw"):
        assert forbidden not in encoded
    for forbidden_key in ("canonical_target", "chat_id", "thread_id"):
        assert forbidden_key not in encoded


def test_cockpit_send_intent_requires_auth(client):
    response = client.post("/api/cockpit/send-intents", json={})

    assert response.status_code == 401


def test_cockpit_capabilities_advertise_real_risky_send_when_configured(client, auth_headers, monkeypatch):
    from hermes_cli import web_server

    monkeypatch.setattr(
        web_server,
        "_cockpit_channel_directory_loader",
        lambda: {"platforms": {"discord": [{"id": "111111111111111111", "name": "hermes", "guild": "Nous", "type": "channel"}]}},
    )

    response = client.get("/api/cockpit/capabilities", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["read_only"] is False
    assert body["external_send_enabled"] is True
    assert body["control_enabled"] is False
    assert body["routing_enabled"] is True
    assert body["voice_enabled"] is False
    assert body["attachments_enabled"] is False
    assert body["risky_send"]["enabled"] is True
    assert body["risky_send"]["kill_switch_active"] is False
    assert body["risky_send"]["delivery_mode"] == "gateway_direct"
    assert body["risky_send"]["allowed_lanes"] == [
        {"lane_alias": "%discord/hermes", "lane_label": "Discord #hermes", "platform": "discord"}
    ]
    assert "/api/cockpit/send-intents" in body["endpoints"]
    _assert_no_send_leaks(body)


def test_cockpit_send_intent_resolves_lane_dispatches_and_returns_display_safe_audit(client, auth_headers, monkeypatch):
    from hermes_cli import web_server

    class Adapter:
        calls = []

        def dispatch(self, *, lane, message_text, idempotency_key):
            self.calls.append((lane.canonical_target["chat_id"], message_text, idempotency_key))
            return {"success": True, "message_id": "discord-message-123"}

    adapter = Adapter()
    monkeypatch.setattr(
        web_server,
        "_cockpit_channel_directory_loader",
        lambda: {"platforms": {"discord": [{"id": "111111111111111111", "name": "hermes", "guild": "Nous", "type": "channel"}]}},
    )
    monkeypatch.setattr(web_server, "_build_cockpit_gateway_adapter", lambda: adapter)

    response = client.post(
        "/api/cockpit/send-intents",
        headers=auth_headers,
        json={
            "lane_alias": "%discord/hermes",
            "idempotency_key": "cockpit-key-web",
            "message_text": "Cockpit test for 111111111111111111",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "dispatched"
    assert body["record"]["content_preview_redacted"] == "Cockpit test for [id]"
    assert body["audit"][-1]["event_type"] == "cockpit_send_dispatched"
    assert adapter.calls == [("111111111111111111", "Cockpit test for 111111111111111111", "cockpit-key-web")]
    _assert_no_send_leaks(body)


def test_cockpit_send_intent_fails_closed_on_ambiguous_lane(client, auth_headers, monkeypatch):
    from hermes_cli import web_server

    monkeypatch.setattr(
        web_server,
        "_cockpit_channel_directory_loader",
        lambda: {"platforms": {"discord": [
            {"id": "111111111111111111", "name": "hermes", "guild": "Nous", "type": "channel"},
            {"id": "222222222222222222", "name": "hermes", "guild": "Other", "type": "channel"},
        ]}},
    )

    response = client.post(
        "/api/cockpit/send-intents",
        headers=auth_headers,
        json={"lane_alias": "#hermes", "idempotency_key": "cockpit-key-web", "message_text": "hello"},
    )

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False
    assert body["status"] == "ambiguous_lane"
    assert body["error_code"] == "ambiguous_lane_alias"
    _assert_no_send_leaks(body)
