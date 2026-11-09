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

import json
import logging
import os

import pytest

from agent import secret_scope
from gateway import relay
from gateway.run_startup import GatewayStartupMixin


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
        "GATEWAY_RELAY_WAKE_URL",
    ):
        monkeypatch.delenv(k, raising=False)
    # Never read config.yaml off disk in these tests.
    monkeypatch.setattr("gateway.run._load_gateway_config", dict, raising=False)


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


def test_provision_url_maps_ws_to_http():
    assert relay._provision_url("wss://c.example/relay") == "https://c.example/relay/provision"
    assert relay._provision_url("ws://c.example/relay") == "http://c.example/relay/provision"
    assert relay._provision_url("https://c.example") == "https://c.example/relay/provision"


# ─────────────────────────── trigger logic ───────────────────────────


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

    def _fake_urlopen(req, timeout=None):
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


# ─────────────────── wake-url forwarding (Phase 5 Unit C) ───────────────────


def test_forwards_wake_url_to_provision(monkeypatch):
    """A suspendable agent stamped with GATEWAY_RELAY_WAKE_URL forwards it to the
    connector so the connector can poke it awake when the first buffered event
    lands on a flipped destination (Unit C wake primitive)."""
    _arm(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_WAKE_URL", "https://wake.example/poke")
    captured: dict = {}
    monkeypatch.setattr(relay, "_post_provision", _stub_post(captured))

    assert relay.self_provision_relay() is True
    assert captured["wake_url"] == "https://wake.example/poke"




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


# ─────────────────────────── displayName (Phase 1 parity, gg#171) ───────────────────────────


def test_relay_display_name_suppresses_stock_brand(monkeypatch):
    """The default 'Hermes Agent' brand is identical on every install — forwarding
    it would shadow the connector's linked-owner fallback (which actually
    disambiguates) with a uniform label. Only customized names are forwarded."""
    monkeypatch.delenv("GATEWAY_RELAY_DISPLAY_NAME", raising=False)

    class _Skin:
        def get_branding(self, key, fallback=""):
            return "Hermes Agent" if key == "agent_name" else fallback

    monkeypatch.setattr("hermes_cli.skin_engine.get_active_skin", lambda: _Skin())
    assert relay.relay_display_name() is None


# ───────────────── boot refresh routing on a multiplexed gateway ─────────────────


def _boot_with_expired_login(tmp_path, monkeypatch, *, env_file: str, stored_portal: str) -> list[str]:
    """Launch home whose Nous access token expired while the box slept, so the startup
    self-provision is the first caller to refresh it. Returns the Portal URLs the refresh used."""
    launch_home = tmp_path / ".hermes"
    launch_home.mkdir()
    (launch_home / ".env").write_text(env_file, encoding="utf-8")
    (launch_home / "auth.json").write_text(json.dumps({
        "active_provider": "nous",
        "providers": {"nous": {
            "portal_base_url": stored_portal,
            "access_token": "expired-access-token", "refresh_token": "refresh-token",
            "expires_at": "2000-01-01T00:00:00+00:00",
        }},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    for name in ("HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL", "NOUS_INFERENCE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(relay, "relay_url", lambda: "wss://connector.example/relay")
    monkeypatch.setattr(relay, "register_relay_adapter", lambda: False)
    portals: list[str] = []

    def _refresh(**kwargs):
        portals.append(kwargs["portal_base_url"])
        raise RuntimeError("stop before any network call")

    monkeypatch.setattr("hermes_cli.auth._refresh_nous_or_quarantine", _refresh)
    return portals


def test_startup_refresh_uses_launch_profile_portal_when_multiplexed(tmp_path, monkeypatch, caplog):
    """A Portal override in the launch profile's .env survives the startup refresh."""
    portals = _boot_with_expired_login(
        tmp_path, monkeypatch, env_file="HERMES_PORTAL_BASE_URL=https://portal.example\n",
        stored_portal="https://portal.example")
    secret_scope.set_multiplex_active(True)

    with caplog.at_level(logging.WARNING):
        GatewayStartupMixin._start_register_plugins_relay_hooks()

    assert portals == ["https://portal.example"]
    assert not [r for r in caplog.records if "no profile secret scope" in r.getMessage()]


def test_startup_refresh_without_override_stays_on_default_portal(tmp_path, monkeypatch):
    """With no override anywhere, the startup refresh keeps the default Portal."""
    from hermes_cli.auth_nous import DEFAULT_NOUS_PORTAL_URL

    portals = _boot_with_expired_login(
        tmp_path, monkeypatch, env_file="", stored_portal=DEFAULT_NOUS_PORTAL_URL)
    secret_scope.set_multiplex_active(True)

    GatewayStartupMixin._start_register_plugins_relay_hooks()

    assert portals == [DEFAULT_NOUS_PORTAL_URL]


def test_startup_refresh_single_profile_reads_process_env_override(tmp_path, monkeypatch):
    """Single-profile hosts (multiplexing off) still take the override from the process env."""
    portals = _boot_with_expired_login(
        tmp_path, monkeypatch, env_file="", stored_portal="https://portal.example")
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "https://portal.example")

    GatewayStartupMixin._start_register_plugins_relay_hooks()

    assert portals == ["https://portal.example"]


def test_startup_idp_secret_ref_keeps_launch_env_value_when_multiplexed(tmp_path, monkeypatch):
    """Only the Nous token step runs in the launch profile's scope: a ``${VAR}`` IdP secret injected
    into the launch env (systemd, ``op run``) still reaches the IdP, as it did unscoped."""
    import io
    from unittest.mock import Mock

    launch_home = tmp_path / ".hermes"
    launch_home.mkdir()
    (launch_home / ".env").write_text("", encoding="utf-8")
    (launch_home / "config.yaml").write_text(
        "gateway:\n  idp:\n    token_url: https://idp.example/token\n"
        "    client_id: relay-client\n    client_secret: ${RELAY_IDP_SECRET}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    # The autouse fixture stubs config reads; this test needs the real ``${VAR}`` expansion.
    from hermes_cli.config_effective import load_user_config_effective
    monkeypatch.setattr("gateway.run._load_gateway_config",
                        lambda config_path=None: load_user_config_effective(launch_home / "config.yaml"))
    monkeypatch.setenv("RELAY_IDP_SECRET", "injected-launch-secret")
    for name in ("GATEWAY_RELAY_IDP_TOKEN_URL", "GATEWAY_RELAY_IDP_CLIENT_ID",
                 "GATEWAY_RELAY_IDP_CLIENT_SECRET", "GATEWAY_RELAY_IDP_SCOPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(relay, "relay_url", lambda: "wss://connector.example/relay")
    monkeypatch.setattr(relay, "register_relay_adapter", lambda: False)
    monkeypatch.setattr(relay, "_post_provision", Mock(return_value={}))
    bodies: list[str] = []

    def _idp(req, timeout=None):
        bodies.append(req.data.decode())
        return io.BytesIO(b'{"access_token": "idp-workload-token"}')

    monkeypatch.setattr("urllib.request.urlopen", _idp)
    secret_scope.set_multiplex_active(True)

    GatewayStartupMixin._start_register_plugins_relay_hooks()

    assert len(bodies) == 1
    assert "client_secret=injected-launch-secret" in bodies[0]

