"""The Agent Card is correct and served at the well-known URL."""

from __future__ import annotations

from starlette.testclient import TestClient

from plugins.platforms.a2a.card import build_agent_card
from plugins.platforms.a2a.entry import _default_service_url, build_app


def test_card_has_required_fields():
    card = build_agent_card("http://localhost:9100/")
    assert card.name == "Hermes Agent"
    assert card.url == "http://localhost:9100/"
    assert card.version
    assert card.capabilities.streaming is True
    assert card.skills and card.skills[0].id == "general-agent"


def test_card_serializes_with_camelcase_aliases():
    card = build_agent_card("http://localhost:9100/")
    dumped = card.model_dump(by_alias=True, exclude_none=True)
    # A2A wire format is camelCase.
    assert "defaultInputModes" in dumped
    assert "protocolVersion" in dumped
    assert dumped["preferredTransport"] == "JSONRPC"


def test_card_served_at_well_known_url():
    app = build_app("127.0.0.1", 9100)
    with TestClient(app) as client:
        resp = client.get("/.well-known/agent-card.json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Hermes Agent"
    assert body["preferredTransport"] == "JSONRPC"
    assert any(skill["id"] == "general-agent" for skill in body["skills"])


def test_default_service_url_brackets_ipv6_hosts():
    assert _default_service_url("::1", 9100) == "http://[::1]:9100/"
    assert _default_service_url("127.0.0.1", 9100) == "http://127.0.0.1:9100/"
