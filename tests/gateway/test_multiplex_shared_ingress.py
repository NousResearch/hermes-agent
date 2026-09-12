"""Inbound-port platforms of a SECONDARY profile are served on the default profile's shared listener
at ``/p/<profile>/<path>`` (gateway/platforms/shared_ingress.py).

Invariants: the forwarded request is verified by the NAMED profile's adapter with that profile's
secret and runs under that profile's runtime scope; the un-prefixed path is untouched; a profile
without an adapter for the path is a 404 rather than the default's adapter; a shared-listener
adapter binds no port of its own.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from gateway.config import GatewayConfig, Platform, PlatformConfig  # noqa: E402


def _line_sig(body: bytes, secret: str) -> str:
    return base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


class _Runner:
    """Just enough GatewayRunner for shared_ingress: the served-profile adapter map."""

    def __init__(self, adapters: dict[str, dict[Platform, Any]]):
        self.config = GatewayConfig(multiplex_profiles=True)
        self._profile_adapters = adapters
        self.adapters: dict = {}


def _line_adapter(secret: str, profile: str):
    from plugins.platforms.line.adapter import LineAdapter
    adapter = LineAdapter(PlatformConfig(enabled=True, extra={
        "channel_access_token": f"tok-{profile}", "channel_secret": secret, "port": 1}))
    adapter._shared_listener_profile = profile
    adapter.set_owner_profile(profile)
    return adapter


async def _publish_line(adapter, runner) -> list[tuple[str, Path]]:
    """Wire the LINE webhook app the way ``connect()`` does, without the LINE API or a bind, and
    record the HERMES_HOME the handler ran under."""
    from gateway.platforms.shared_ingress import bind_listener
    from hermes_constants import get_hermes_home
    seen: list[tuple[str, Path]] = []
    adapter.gateway_runner = runner

    async def dispatch(event):
        seen.append((event.get("type"), Path(get_hermes_home())))

    adapter._dispatch_event = dispatch
    app = web.Application(client_max_size=1024)
    app.router.add_post(adapter.webhook_path, adapter._handle_webhook)
    bound = await bind_listener(adapter, app, "127.0.0.1", 1, adapter.webhook_path)
    assert bound is None  # shared-listener mode never binds
    return seen


@pytest.fixture
def mux_home(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    for name in ("coder", "ops"):
        (root / "profiles" / name).mkdir(parents=True)
        (root / "profiles" / name / ".env").write_text(f"LINE_CHANNEL_SECRET=secret-{name}\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", root), ("coder", root / "profiles" / "coder"), ("ops", root / "profiles" / "ops")])
    return root


async def _shared_listener(runner) -> TestClient:
    """The default profile's listener as the webhook adapter builds it: bare routes + /p/ forwarding."""
    from gateway.platforms.webhook import WebhookAdapter
    listener = WebhookAdapter(PlatformConfig(enabled=True, extra={"port": 1, "routes": {}}))
    listener.gateway_runner = runner
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", listener._handle_webhook)
    app.router.add_post("/p/{profile}/webhooks/{route_name}", listener._handle_webhook)
    app.router.add_route("*", "/p/{profile}/{tail:.*}", listener._handle_profile_ingress)
    return TestClient(TestServer(app))


@pytest.mark.asyncio
async def test_prefixed_line_webhook_is_verified_by_the_named_profiles_secret_under_its_scope(mux_home):
    coder, ops = _line_adapter("secret-coder", "coder"), _line_adapter("secret-ops", "ops")
    runner = _Runner({"coder": {Platform("line"): coder}, "ops": {Platform("line"): ops}})
    coder_seen, ops_seen = await _publish_line(coder, runner), await _publish_line(ops, runner)
    body = b'{"events":[{"type":"probe"}]}'
    async with await _shared_listener(runner) as client:
        ok = await client.post("/p/coder/line/webhook", data=body,
                               headers={"X-Line-Signature": _line_sig(body, "secret-coder")})
        assert ok.status == 200
        # ops' secret is rejected at coder's URL — the adapter is per profile, so is the secret.
        wrong = await client.post("/p/coder/line/webhook", data=body,
                                  headers={"X-Line-Signature": _line_sig(body, "secret-ops")})
        assert wrong.status == 401
        # The un-prefixed path keeps serving only the default profile's own routes.
        bare = await client.post("/line/webhook", data=body,
                                 headers={"X-Line-Signature": _line_sig(body, "secret-coder")})
        assert bare.status == 404
        # An unserved profile, and a served profile without that platform, are 404 — never another adapter.
        assert (await client.post("/p/nope/line/webhook", data=body)).status == 404
        runner._profile_adapters["ops"] = {}
        assert (await client.post("/p/ops/line/webhook", data=body,
                                  headers={"X-Line-Signature": _line_sig(body, "secret-ops")})).status == 404
    assert [t for t, _ in coder_seen] == ["probe"] and ops_seen == []
    assert coder_seen[0][1] == mux_home / "profiles" / "coder"


@pytest.mark.asyncio
async def test_shared_listener_adapter_records_its_public_ingress_url(mux_home, monkeypatch):
    """Runtime status carries the /p/<profile>/ URL so `gateway status` / the dashboard can show it."""
    writes: list[dict] = []
    monkeypatch.setattr("gateway.status.write_runtime_status", lambda **kw: writes.append(kw))
    coder = _line_adapter("secret-coder", "coder")
    coder._runtime_status_platform_key = "coder:line"
    runner = _Runner({"coder": {Platform("line"): coder}})
    runner.adapters = {Platform.API_SERVER: type("L", (), {"_host": "0.0.0.0", "_port": 8642})()}
    await _publish_line(coder, runner)
    assert coder._shared_ingress_url == "http://127.0.0.1:8642/p/coder/line/webhook"
    assert {"platform": "coder:line", "ingress_url": coder._shared_ingress_url} == {
        k: v for k, v in writes[-1].items() if k in ("platform", "ingress_url")}
    assert coder._media_url("tok", "a.png").startswith("http://127.0.0.1:8642/p/coder/line/media/")
