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
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

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
        "GATEWAY_RELAY_WAKE_URL",
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

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", _fake_urlopen)

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

def test_relay_wake_url_from_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_WAKE_URL", "  https://wake.example/poke  ")
    assert relay.relay_wake_url() == "https://wake.example/poke"


def test_relay_wake_url_absent_is_none():
    assert relay.relay_wake_url() is None


def test_relay_wake_url_from_config(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"gateway": {"relay_wake_url": "https://wake.from-config/poke"}},
        raising=False,
    )
    assert relay.relay_wake_url() == "https://wake.from-config/poke"


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


def test_wake_url_absent_forwards_none(monkeypatch):
    """No stamp (self-hosted / non-suspendable) -> wake_url None; the connector
    stores null and simply never pokes (it can't wake what it can't reach)."""
    _arm(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(relay, "_post_provision", _stub_post(captured))

    assert relay.self_provision_relay() is True
    assert captured["wake_url"] is None


def test_post_provision_body_includes_wakeUrl_only_when_set(monkeypatch):
    """The real _post_provision adds `wakeUrl` to the JSON body ONLY when a value
    is supplied — omitting it lets the connector store null (back-compat)."""
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

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", _fake_urlopen)

    # With a wake url -> present in the body.
    relay._post_provision(
        provision_url="https://c.example/relay/provision",
        access_token="tok",
        gateway_id="gw-1",
        platform="discord",
        bot_id="app",
        gateway_endpoint=None,
        route_keys=[],
        wake_url="https://wake.example/poke",
    )
    assert sent["body"]["wakeUrl"] == "https://wake.example/poke"

    # Without one -> the key is absent entirely (not "").
    relay._post_provision(
        provision_url="https://c.example/relay/provision",
        access_token="tok",
        gateway_id="gw-1",
        platform="discord",
        bot_id="app",
        gateway_endpoint=None,
        route_keys=[],
    )
    assert "wakeUrl" not in sent["body"]


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


# ───────────────── redirect credential stripping (real servers) ─────────────────

class _RedirectingProvisionHandler(BaseHTTPRequestHandler):
    """Answers POST /relay/provision with a 302 to a configurable target.

    A second, independent server plays the redirect target and records the
    headers it received — used to prove the gateway's Bearer identity token
    never reaches an unintended origin. 302 (not 307/308) because stdlib's
    HTTPRedirectHandler only follows a POST redirect for 301/302/303 — 307/308
    are reserved for method-preserving redirects and raise immediately for
    POST, so they can't carry the header anywhere in the first place.
    """

    redirect_to = ""  # full URL, set per test
    received_headers: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path.rstrip("/") == "/relay/provision":
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
        else:
            self._respond()

    def do_GET(self):
        # A 302 to a POST converts the redirected request to a bodyless GET
        # (stdlib drops the body on any POST redirect); the target server
        # only ever receives that GET, never the original POST.
        self._respond()

    def _respond(self):
        type(self).received_headers = dict(self.headers)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"secret": "leaked"}).encode())

    def log_message(self, format, *args):
        pass


def test_post_provision_strips_bearer_on_cross_host_redirect(monkeypatch):
    """The connector's own Bearer identity token must not follow a redirect
    to a different origin (compromised/misconfigured proxy in front of the
    connector) — mirrors the cross-host credential-redirect fix already
    applied to the model-catalog fetch paths. The redirect is still followed
    (a legitimate reachability concern), just without the credential."""
    _RedirectingProvisionHandler.received_headers = {}
    server = HTTPServer(("127.0.0.1", 0), _RedirectingProvisionHandler)
    target_server = HTTPServer(("127.0.0.1", 0), _RedirectingProvisionHandler)
    port = server.server_address[1]
    target_port = target_server.server_address[1]
    _RedirectingProvisionHandler.redirect_to = f"http://127.0.0.1:{target_port}/collect"
    Thread(target=server.serve_forever, daemon=True).start()
    Thread(target=target_server.serve_forever, daemon=True).start()

    try:
        result = relay._post_provision(
            provision_url=f"http://127.0.0.1:{port}/relay/provision",
            access_token="super-secret-bearer",
            gateway_id="gw-1",
            platform="discord",
            bot_id="app",
            gateway_endpoint=None,
            route_keys=[],
        )
    finally:
        server.shutdown()
        target_server.shutdown()

    # The redirect target answered normally (it isn't blocked), proving the
    # request really was followed — but without the Bearer token attached.
    assert result["secret"] == "leaked"
    headers = {k.lower(): v for k, v in _RedirectingProvisionHandler.received_headers.items()}
    assert "authorization" not in headers
