"""Multiplexed dynamic-route admission must read the launch registry, not the routed profile's.

Live incident: POST /p/norn/webhooks/orca-provisioner-alert answered 503. The subscription lives
only in the launch (default) multiplexer registry — per the documented route-profile design a
``/p/<profile>/`` dynamic route is one registry entry with a ``profile`` field, never a copy in the
served profile's home. Request handling (unscoped) loaded the route and authenticated the edge,
but durable admission runs inside ``webhook_ingress.producer_scope`` → ``_async_profile_runtime_scope``
(routed profile home). ``session_envelope._snapshot_native`` → ``webhook_delivery.route_digest``
reloaded dynamic routes through ``get_hermes_home()`` — now the routed profile's home, which holds
no registry — cleared the route, and raised ``RuntimeStoreError('not_found')``: a retryable 503.
"""
import hashlib
import hmac
import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _WebhookDeliveryIdentity
from gateway.run import GatewayRunner
from gateway.session import SessionStore
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import initialize_session_authority
from hermes_state import SessionDB

_ROUTE = "orca-provisioner-alert"
_SECRET = "edge-route-secret-abc123"


def _github_signature(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _write_subscription(home: Path) -> None:
    (home / "webhook_subscriptions.json").write_text(json.dumps({
        _ROUTE: {"secret": _SECRET, "profile": "norn", "prompt": "alert: {text}",
                 "deliver": "log"}}), encoding="utf-8")


def _launch_and_norn(tmp_path, monkeypatch):
    launch = tmp_path / "root"
    norn = launch / "profiles" / "norn"
    norn.mkdir(parents=True)
    (norn / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    return launch, norn


def test_reload_dynamic_routes_ignores_routed_profile_scope(tmp_path, monkeypatch):
    """The subscriptions registry is the launch home's: a per-turn routed-profile scope (which
    admission runs under) must not redirect the reload at an empty profile registry."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from gateway.platforms.webhook_delivery import route_digest

    launch, norn = _launch_and_norn(tmp_path, monkeypatch)
    _write_subscription(launch)
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {}}))
    adapter._reload_dynamic_routes()
    assert _ROUTE in adapter._routes, "launch registry must load the dynamic route unscoped"
    chat_id = _WebhookDeliveryIdentity.from_parts("norn", _ROUTE, "d1").session_chat_id
    expected = route_digest(adapter, chat_id)

    token = set_hermes_home_override(norn)
    try:
        adapter._reload_dynamic_routes()  # what route_digest does at admission time
        assert _ROUTE in adapter._routes
        assert route_digest(adapter, chat_id) == expected
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_prefixed_dynamic_route_admission_commits_under_routed_profile_scope(
        tmp_path, monkeypatch):
    """Full durable admission of a /p/<profile>/ dynamic route: 202 and a committed row in the
    routed profile's ledger — not the live 503 from route_digest losing the route mid-admission."""
    launch, norn = _launch_and_norn(tmp_path, monkeypatch)
    _write_subscription(launch)

    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {}}))
    runner = GatewayRunner()
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = (("default", launch), ("norn", norn))
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve",
                        lambda multiplex: [("default", launch), ("norn", norn)])
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir",
                        lambda name: norn if name == "norn" else launch)

    registry = SessionAuthorities(launch, multiplexed=True)
    runner.session_authorities = registry

    async def _build_authority(home, profile_id, name):
        store = SessionStore(sessions_dir=home / "sessions", config=runner.config)
        store._db = SessionDB(db_path=home / "state.db")
        runner.session_store = store
        runner._session_db = store._db
        authority = await initialize_session_authority(runner, profile_id=str(home),
                                                       instance_id="mux-route-test")
        registry.add(home, authority, name=name)
        authority._schedule = lambda ref: None  # hold execution; admission is what is under test
        return authority

    launch_authority = await _build_authority(launch, "default", None)
    norn_authority = await _build_authority(norn, "norn", "norn")
    runner.session_authority = launch_authority  # producer_scope requires the launch authority

    adapter.gateway_runner = runner
    runner._profile_adapters = {"norn": {Platform.WEBHOOK: adapter}}
    runner._native_transport_homes = {"norn": norn}
    runner._wire_adapter_handlers(adapter)

    app = web.Application()
    app.router.add_post("/p/{profile}/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        body = json.dumps({"text": "disk full"}).encode()
        headers = {"X-GitHub-Delivery": "edge-delivery-1",
                   "X-Hub-Signature-256": _github_signature(body, _SECRET)}
        response = await client.post(f"/p/norn/webhooks/{_ROUTE}", data=body, headers=headers)
        assert response.status == 202, await response.text()
        assert (await response.json())["status"] == "accepted"

    rows = norn_authority.db._read_all(
        "SELECT payload_json FROM session_admissions WHERE status='queued'")
    assert len(rows) == 1
    envelope = json.loads(rows[0]["payload_json"])["native_text_v1"]
    assert envelope["source"]["profile"] == "norn"
    assert envelope["webhook_delivery"]["deliver"] == "log"
    chat_id = _WebhookDeliveryIdentity.from_parts("norn", _ROUTE, "edge-delivery-1").session_chat_id
    assert envelope["source"]["chat_id"] == chat_id
    expected_digest = hashlib.sha256(json.dumps(
        json.loads((launch / "webhook_subscriptions.json").read_text())[_ROUTE],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert envelope["webhook_route"] == expected_digest
    assert launch_authority.db._read_all("SELECT admission_id FROM session_admissions") == []
