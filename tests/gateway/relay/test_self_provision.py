"""Unit tests for managed-boot relay self-provisioning.

Covers gateway.relay.self_provision_if_managed() + the relay_endpoint() /
relay_route_keys() config readers. The connector HTTP POST is monkeypatched
(the cross-repo E2E exercises the real /relay/provision); these prove the
TRIGGER logic, in-process env wiring, and fail-soft boot behaviour.
"""

from __future__ import annotations

import pytest

import gateway.relay as relay


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "GATEWAY_RELAY_URL",
        "GATEWAY_RELAY_ID",
        "GATEWAY_RELAY_SECRET",
        "GATEWAY_RELAY_DELIVERY_KEY",
        "GATEWAY_RELAY_ENDPOINT",
        "GATEWAY_RELAY_ROUTE_KEYS",
        "GATEWAY_RELAY_PLATFORM",
        "GATEWAY_RELAY_BOT_ID",
    ):
        monkeypatch.delenv(k, raising=False)
    # Never read config.yaml off disk in these tests.
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {}, raising=False)


def _stub_post(captured: dict):
    """A fake _post_provision that records its kwargs and returns creds."""

    def _fake(**kwargs):
        captured.update(kwargs)
        return {
            "secret": "a" * 64,
            "deliveryKey": "b" * 64,
            "tenant": "org-tenant-x",
            "gatewayId": kwargs["gateway_id"],
            "routeKeys": kwargs["route_keys"],
        }

    return _fake


def _arm(monkeypatch, *, managed=True, url="wss://connector.example/relay", token="nas-token"):
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: managed)
    monkeypatch.setattr(relay, "relay_url", lambda: url)
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: token)


# ─────────────────────────── config readers ───────────────────────────

def test_relay_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_ENDPOINT", "https://gw.example.com/inbound/")
    assert relay.relay_endpoint() == "https://gw.example.com/inbound"


def test_relay_endpoint_absent_is_none():
    assert relay.relay_endpoint() is None


def test_relay_route_keys_csv(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_ROUTE_KEYS", "guild-1, guild-2 ,, guild-3")
    assert relay.relay_route_keys() == ["guild-1", "guild-2", "guild-3"]


def test_relay_route_keys_empty():
    assert relay.relay_route_keys() == []


def test_provision_url_maps_ws_to_http():
    assert relay._provision_url("wss://c.example/relay") == "https://c.example/relay/provision"
    assert relay._provision_url("ws://c.example/relay") == "http://c.example/relay/provision"
    assert relay._provision_url("https://c.example") == "https://c.example/relay/provision"


# ─────────────────────────── trigger logic ───────────────────────────

def test_skips_when_not_managed(monkeypatch):
    _arm(monkeypatch, managed=False)
    called = {"n": 0}
    monkeypatch.setattr(relay, "_post_provision", lambda **k: called.__setitem__("n", called["n"] + 1) or {})
    assert relay.self_provision_if_managed() is False
    assert called["n"] == 0


def test_skips_when_relay_not_configured(monkeypatch):
    _arm(monkeypatch, url=None)
    called = {"n": 0}
    monkeypatch.setattr(relay, "_post_provision", lambda **k: called.__setitem__("n", called["n"] + 1) or {})
    assert relay.self_provision_if_managed() is False
    assert called["n"] == 0


def test_skips_when_secret_already_pinned(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_ID", "gw-pinned")
    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "deadbeef")
    called = {"n": 0}
    monkeypatch.setattr(relay, "_post_provision", lambda **k: called.__setitem__("n", called["n"] + 1) or {})
    assert relay.self_provision_if_managed() is False
    assert called["n"] == 0
    # The pinned secret is untouched.
    assert relay.relay_connection_auth() == ("gw-pinned", "deadbeef")


# ─────────────────────────── happy path ───────────────────────────

def test_provisions_and_sets_env_in_process(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_ENDPOINT", "https://gw.example.com/inbound")
    monkeypatch.setenv("GATEWAY_RELAY_ROUTE_KEYS", "guild-1,guild-2")
    captured: dict = {}
    monkeypatch.setattr(relay, "_post_provision", _stub_post(captured))

    assert relay.self_provision_if_managed() is True
    # The connector POST carried the gateway-asserted endpoint + route keys.
    assert captured["provision_url"] == "https://connector.example/relay/provision"
    assert captured["access_token"] == "nas-token"
    assert captured["gateway_endpoint"] == "https://gw.example.com/inbound"
    assert captured["route_keys"] == ["guild-1", "guild-2"]
    # Creds landed in os.environ (in-process), so register_relay_adapter() reads them.
    gid, secret = relay.relay_connection_auth()
    assert gid and secret == "a" * 64
    key, _host, _port = relay.relay_inbound_config()
    assert key == "b" * 64


def test_outbound_only_when_no_endpoint(monkeypatch):
    _arm(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(relay, "_post_provision", _stub_post(captured))

    assert relay.self_provision_if_managed() is True
    assert captured["gateway_endpoint"] is None
    assert captured["route_keys"] == []
    assert relay.relay_connection_auth()[1] == "a" * 64


# ─────────────────────────── fail-soft ───────────────────────────

def test_token_failure_is_non_fatal(monkeypatch):
    _arm(monkeypatch)

    def _boom():
        raise RuntimeError("no token")

    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", _boom)
    # Must not raise; returns False; no creds set.
    assert relay.self_provision_if_managed() is False
    assert relay.relay_connection_auth() == (None, None)


def test_connector_failure_is_non_fatal(monkeypatch):
    _arm(monkeypatch)

    def _boom(**kwargs):
        raise RuntimeError("connector returned HTTP 503")

    monkeypatch.setattr(relay, "_post_provision", _boom)
    assert relay.self_provision_if_managed() is False
    assert relay.relay_connection_auth() == (None, None)
