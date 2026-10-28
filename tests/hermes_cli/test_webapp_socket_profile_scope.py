"""The Webapp's gateway socket answers for the profile its ticket names, on a multiplexed gateway.

A browser tab's ``/api/ws`` reaches the gateway through the dashboard mount (``chat_ws.gateway_ws``),
not the native ticket path (``gateway/run_api.py``) that wraps the socket in ``owner_scope``. The
authority answers ``-32601`` for ``billing.state``, so the frame falls back to the legacy handler,
which reads the Nous Portal override through the socket's scope: unbound, a multiplexed gateway
refused the read (``UnscopedSecretError`` -> "no profile secret scope on a multiplexed call") and the
token went to the default Portal. ``dispatch_legacy`` binds a non-launch ticket's profile; a
launch-profile socket relies on ``gateway_ws``'s own scope.

Real ``gateway_ws``, ``handle_ws``, session authorities, legacy fallback and billing handler; only
the upgrade gate (which stamps the ticket's identity and home on the socket) and HTTP are substituted.
"""

import asyncio
import base64
import io
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect


def _jwt(label):
    def part(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    return ".".join((part({"alg": "none"}), part({
        "sub": label, "scope": "inference:invoke", "exp": int(time.time()) + 3600,
    }), "fixture"))


def _home(home, label):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model:\n  provider: nous\n  default: fixture-model\n")
    (home / ".env").write_text(f"HERMES_PORTAL_BASE_URL=https://{label}-portal.invalid\n")
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {"nous": {
        "access_token": _jwt(label), "refresh_token": f"{label}-refresh",
        "expires_at": "2999-01-01T00:00:00+00:00", "scope": "inference:invoke", "client_id": "fixture-client",
        "portal_base_url": "https://portal.nousresearch.com",
        "inference_base_url": "https://inference-api.nousresearch.com/v1",
    }}}))


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """A multiplexed runner serving the launch home and profile ``b``, each with its own authority."""
    from agent import secret_scope
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from hermes_cli import auth, nous_account, nous_billing
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    from tui_gateway import launch_profile_policy, server

    launch, b = tmp_path / "launch", tmp_path / "launch" / "profiles" / "b"
    _home(launch, "launch")
    _home(b, "b")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL", "NOUS_INFERENCE_BASE_URL",
                "HERMES_DEV_BILLING_FIXTURE", "HERMES_DEV_SUBSCRIPTION_FIXTURE", "HERMES_DEV_CREDITS_FIXTURE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr(server, "_hermes_home", str(launch))
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(auth, "_RESOLVE_TOKEN_CACHE", {})
    monkeypatch.setattr(nous_billing, "_token_cache", {})
    nous_account.reset_nous_portal_account_info_cache()
    previous = secret_scope.is_multiplex_active()
    launch_profile_policy.activate_multi_profile_hosting()

    by_home, dbs = {}, []
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=True))
    runner.session_authorities = SimpleNamespace(for_home=lambda home: by_home.get(Path(home).resolve()))
    for home in (launch, b):
        db = SessionDB(home / "state.db")
        dbs.append(db)
        store = SessionStore(config=GatewayConfig(), sessions_dir=home / "sessions")
        store._db = db
        runner.session_store = store
        by_home[home.resolve()] = SessionAuthority(runner, profile_id=str(home), instance_id="first", db=db,
                                                   epoch=begin_runtime_epoch(db, instance_id="first"))
    runner.session_authority = by_home[launch.resolve()]

    portal_hits = []

    def http_edge(request):
        portal_hits.append(request.url.host)
        return httpx.Response(200, json={"paid_service_access": True})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: real_client(*a, **{**k, "transport": httpx.MockTransport(http_edge)}))

    def urlopen(request, *args, **kwargs):
        portal_hits.append(httpx.URL(request.full_url).host)
        response = io.BytesIO(json.dumps({"ok": True, "balanceUsd": "0", "role": "MEMBER"}).encode())
        response.status = 200
        return response
    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    from hermes_cli import mcp_startup
    from hermes_cli.web_routers import chat_ws

    async def allowed(*_a, **_k):
        return True
    monkeypatch.setattr(chat_ws, "_close_unless_sidecar_allowed", allowed)
    monkeypatch.setattr(mcp_startup, "start_deferred_mcp_discovery_now", lambda: None)
    try:
        yield SimpleNamespace(launch=launch, b=b, authority=runner.session_authority, portal_hits=portal_hits)
    finally:
        secret_scope.set_multiplex_active(previous)
        for db in dbs:
            db.close()


class _Socket:
    """An accepted ``/api/ws`` connection carrying what the upgrade gate stamps on a Webapp ticket."""

    def __init__(self, authority, frames, ticket_home=None):
        self.app = SimpleNamespace(state=SimpleNamespace(session_authority=authority))
        self.scope, self.client, self.closed, self.sent = {}, None, None, []
        self._hermes_auth_identity = {"user_id": "owner", "provider": "session-token"}
        self._hermes_ticket_profile_home = None if ticket_home is None else str(ticket_home)
        self._frames = [json.dumps(frame) for frame in frames]
        self._answered = asyncio.Event()

    async def accept(self, subprotocol=None):
        pass

    async def close(self, code=None):
        self.closed = code

    async def send_text(self, text):
        frame = json.loads(text)
        self.sent.append(frame)
        if frame.get("id") is not None:
            self._answered.set()

    async def receive_text(self):
        if self._frames:
            return self._frames.pop(0)
        await asyncio.wait_for(self._answered.wait(), 20)
        raise WebSocketDisconnect(1000)

    def reply(self, req_id):
        return next(frame for frame in self.sent if frame.get("id") == req_id)


def _billing_state_over_socket(gateway, ticket_home):
    from hermes_cli.web_routers import chat_ws
    ws = _Socket(gateway.authority, [{"jsonrpc": "2.0", "id": "bill", "method": "billing.state", "params": {}}],
                 ticket_home)
    gateway.portal_hits.clear()
    asyncio.run(chat_ws.gateway_ws(ws))
    assert ws.closed is None, ws.closed
    return ws.reply("bill"), list(gateway.portal_hits)


def test_billing_state_on_a_webapp_socket_reaches_its_tickets_profile_portal(gateway, caplog):
    with caplog.at_level(logging.WARNING):
        seen = [_billing_state_over_socket(gateway, home) for home in (gateway.b, gateway.launch, gateway.b)]

    for (reply, hits), label in zip(seen, ("b", "launch", "b")):
        assert reply["result"]["logged_in"] is True, reply
        assert hits and set(hits) == {f"{label}-portal.invalid"}, (label, hits)
    assert "no profile secret scope" not in caplog.text


def test_a_profile_ticket_with_no_authority_bound_is_refused_not_served_as_the_launch_profile(gateway):
    from hermes_cli.web_routers import chat_ws
    ws = _Socket(None, [{"jsonrpc": "2.0", "id": "bill", "method": "billing.state", "params": {}}], gateway.b)

    asyncio.run(chat_ws.gateway_ws(ws))

    assert ws.closed == 1013
    assert ws.sent == [] and gateway.portal_hits == []
