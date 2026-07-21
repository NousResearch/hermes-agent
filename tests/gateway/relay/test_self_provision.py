"""Unit tests for boot-time relay self-provisioning.

Covers gateway.relay.self_provision_relay() + the relay_endpoint() /
relay_route_keys() config readers. The connector HTTP POST is monkeypatched
(the cross-repo E2E exercises the real /relay/provision); these prove the
TRIGGER logic, in-process env wiring, and fail-soft boot behaviour.

The trigger is deliberately NOT is_managed() (that means NixOS/package-manager-
managed, which is False on a NAS-hosted Fly agent). The real gate is
"relay_url set + no pinned secret + a resolvable NAS token".
"""

from __future__ import annotations

import os

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
        "GATEWAY_RELAY_INSTANCE_ID",
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


def _arm(monkeypatch, *, url="wss://connector.example/relay", token="nas-token"):
    """Arm the real trigger: a relay URL + a resolvable NAS token.

    Note there is intentionally no `managed` knob — self-provision no longer
    consults is_managed(). A test that wants the "no NAS identity" branch
    monkeypatches resolve_nous_access_token to raise instead.
    """
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


def test_relay_instance_id_from_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_INSTANCE_ID", "  inst-abc  ")
    assert relay.relay_instance_id() == "inst-abc"


def test_relay_instance_id_absent_is_none():
    assert relay.relay_instance_id() is None


def test_relay_instance_id_from_config(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"gateway": {"relay_instance_id": "inst-from-config"}},
        raising=False,
    )
    assert relay.relay_instance_id() == "inst-from-config"


def test_provision_url_maps_ws_to_http():
    assert relay._provision_url("wss://c.example/relay") == "https://c.example/relay/provision"
    assert relay._provision_url("ws://c.example/relay") == "http://c.example/relay/provision"
    assert relay._provision_url("https://c.example") == "https://c.example/relay/provision"


# ─────────────────────────── trigger logic ───────────────────────────

def test_provisions_on_nas_host_that_is_NOT_is_managed(monkeypatch):
    """Regression: a NAS-hosted Fly agent sets neither HERMES_MANAGED nor a
    .managed marker, so is_managed() is False. Self-provision must STILL fire —
    the old is_managed() gate silently no-oped exactly this case in staging.
    """
    # Force is_managed() False to model a real hosted agent; it must be irrelevant.
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    _arm(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(relay, "_post_provision", _stub_post(captured))

    assert relay.self_provision_relay() is True
    assert relay.relay_connection_auth()[1] == "a" * 64


def test_skips_when_relay_not_configured(monkeypatch):
    _arm(monkeypatch, url=None)
    called = {"n": 0}
    monkeypatch.setattr(relay, "_post_provision", lambda **k: called.__setitem__("n", called["n"] + 1) or {})
    assert relay.self_provision_relay() is False
    assert called["n"] == 0


def test_skips_when_secret_already_pinned(monkeypatch):
    """A self-hosted, enrolled gateway has a pinned secret -> never self-provisions."""
    _arm(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_ID", "gw-pinned")
    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "deadbeef")
    called = {"n": 0}
    monkeypatch.setattr(relay, "_post_provision", lambda **k: called.__setitem__("n", called["n"] + 1) or {})
    assert relay.self_provision_relay() is False
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

    assert relay.self_provision_relay() is True
    # The connector POST carried the gateway-asserted endpoint + route keys.
    assert captured["provision_url"] == "https://connector.example/relay/provision"
    assert captured["access_token"] == "nas-token"
    assert captured["gateway_endpoint"] == "https://gw.example.com/inbound"
    assert captured["route_keys"] == ["guild-1", "guild-2"]
    # Creds landed in os.environ (in-process), so register_relay_adapter() reads them.
    gid, secret = relay.relay_connection_auth()
    assert gid and secret == "a" * 64
    # The delivery key is persisted in-process too (issued by the connector,
    # kept for forward-compat; inbound rides the WS so it isn't consumed).
    assert os.environ["GATEWAY_RELAY_DELIVERY_KEY"] == "b" * 64


def test_outbound_only_when_no_endpoint(monkeypatch):
    _arm(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(relay, "_post_provision", _stub_post(captured))

    assert relay.self_provision_relay() is True
    assert captured["gateway_endpoint"] is None
    assert captured["route_keys"] == []
    assert relay.relay_connection_auth()[1] == "a" * 64


# ─────────────────── instance-id forwarding (Phase 6 Unit α) ───────────────────

def test_forwards_instance_id_to_provision(monkeypatch):
    """A managed agent stamped with GATEWAY_RELAY_INSTANCE_ID forwards it to the
    connector so it can bind gatewayId -> instanceId (per-instance routing)."""
    _arm(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_INSTANCE_ID", "inst-abc")
    captured: dict = {}
    monkeypatch.setattr(relay, "_post_provision", _stub_post(captured))

    assert relay.self_provision_relay() is True
    assert captured["instance_id"] == "inst-abc"


def test_instance_id_absent_forwards_none(monkeypatch):
    """No stamp (self-hosted / pre-Phase-6) -> instance_id None; the connector
    stores null and per-instance routing simply has no binding yet."""
    _arm(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(relay, "_post_provision", _stub_post(captured))

    assert relay.self_provision_relay() is True
    assert captured["instance_id"] is None


def test_post_provision_body_includes_instanceId_only_when_set(monkeypatch):
    """The real _post_provision adds `instanceId` to the JSON body ONLY when a
    value is supplied — omitting it lets the connector store null (back-compat),
    rather than binding an empty string."""
    import json

    sent: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"secret": "a" * 64, "deliveryKey": "b" * 64, "tenant": "t", "gatewayId": "gw-1"}).encode()

    def _fake_urlopen(req, timeout=None):  # noqa: ANN001
        sent["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    # With an instance id -> present in the body.
    relay._post_provision(
        provision_url="https://c.example/relay/provision",
        access_token="tok",
        gateway_id="gw-1",
        platform="discord",
        bot_id="app",
        gateway_endpoint=None,
        route_keys=[],
        instance_id="inst-abc",
    )
    assert sent["body"]["instanceId"] == "inst-abc"

    # Without one -> the key is absent entirely (not "" ).
    relay._post_provision(
        provision_url="https://c.example/relay/provision",
        access_token="tok",
        gateway_id="gw-1",
        platform="discord",
        bot_id="app",
        gateway_endpoint=None,
        route_keys=[],
    )
    assert "instanceId" not in sent["body"]


# ─────────────────────────── fail-soft ───────────────────────────

def test_no_nas_token_is_non_fatal(monkeypatch):
    """A self-hosted box with a relay URL but no resolvable NAS identity skips
    quietly (this is the branch that replaces the old is_managed() gate for the
    non-NAS case)."""
    monkeypatch.setattr(relay, "relay_url", lambda: "wss://connector.example/relay")

    def _boom():
        raise RuntimeError("no token")

    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", _boom)
    # Must not raise; returns False; no creds set.
    assert relay.self_provision_relay() is False
    assert relay.relay_connection_auth() == (None, None)


def test_connector_failure_is_non_fatal(monkeypatch):
    _arm(monkeypatch)

    def _boom(**kwargs):
        raise RuntimeError("connector returned HTTP 503")

    monkeypatch.setattr(relay, "_post_provision", _boom)
    assert relay.self_provision_relay() is False
    assert relay.relay_connection_auth() == (None, None)
