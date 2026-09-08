"""Per-profile toolset isolation on the multiplexed api_server (#91583 defect 2).

Two defects covered:

1. With multiplexing ON, ``/p/<x>/v1/toolsets`` must reflect profile *x*'s own
   ``platform_toolsets.api_server`` — for both the gateway-owning default
   profile and a secondary profile — never the listener owner's config.

2. With multiplexing OFF, a ``/p/<other>/`` prefix used to be silently
   ignored, serving the gateway owner's config under another profile's URL.
   It must now be rejected (404); only a self-referential prefix (naming the
   profile this gateway actually serves) falls through.

E2E-style: real profile homes under a temp HERMES root, real config.yaml
files read through the canonical loaders, and real aiohttp request routing
(TestClient) through the profile-prefix middleware. No mocked config reads.
"""
from __future__ import annotations

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from gateway.config import GatewayConfig, PlatformConfig  # noqa: E402
from gateway.platforms.api_server import (  # noqa: E402
    APIServerAdapter,
    _PROFILE_REJECTED,
)

OWNER_KEY = "owner-key-1234567890abcdef"
LOKAJ_KEY = "lokaj-key-1234567890abcdef"


@pytest.fixture()
def hermes_root(tmp_path, monkeypatch):
    """Two real profile homes: default (owner) and 'lokaj' (secondary)."""
    root = tmp_path / "hermes"
    lokaj = root / "profiles" / "lokaj"
    lokaj.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "platform_toolsets:\n  api_server: [web, file]\n",
        encoding="utf-8",
    )
    (lokaj / "config.yaml").write_text(
        "platform_toolsets:\n  api_server: [web, file, computer_use]\n",
        encoding="utf-8",
    )
    (lokaj / ".env").write_text(f"API_SERVER_KEY={LOKAJ_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    # get_default_hermes_root memoizes per (native_home, env_home) pair, so
    # the env change alone re-keys it; no cache reset needed.
    return root


def _make_adapter(multiplex: bool) -> APIServerAdapter:
    cfg = PlatformConfig(enabled=True, extra={"key": OWNER_KEY})
    adapter = APIServerAdapter(cfg)

    class _Runner:
        config = GatewayConfig(multiplex_profiles=multiplex)

    adapter.gateway_runner = _Runner()
    return adapter


def _make_app(adapter: APIServerAdapter) -> web.Application:
    """Mirror connect()'s wiring: middleware + native and /p/ mirror routes."""
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    return app


async def _enabled_toolsets(cli: TestClient, path: str, key: str):
    resp = await cli.get(path, headers={"Authorization": f"Bearer {key}"})
    if resp.status != 200:
        return resp.status, None
    body = await resp.json()
    return resp.status, {d["name"] for d in body["data"] if d["enabled"]}


class TestMultiplexOnToolsetIsolation:
    @pytest.mark.asyncio
    async def test_each_profile_sees_its_own_toolsets(self, hermes_root):
        adapter = _make_adapter(multiplex=True)
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            # Owner (default) profile: bare route and /p/default mirror.
            for path in ("/v1/toolsets", "/p/default/v1/toolsets"):
                status, enabled = await _enabled_toolsets(cli, path, OWNER_KEY)
                assert status == 200, path
                assert "computer_use" not in enabled, path
                assert {"web", "file"} <= enabled, path

            # Secondary profile: its own config, its own key.
            status, enabled = await _enabled_toolsets(
                cli, "/p/lokaj/v1/toolsets", LOKAJ_KEY
            )
            assert status == 200
            # The exact repro from #91583 defect 2: computer_use enabled in
            # lokaj's config must be reported enabled under /p/lokaj/…
            assert "computer_use" in enabled

    @pytest.mark.asyncio
    async def test_owner_key_does_not_open_secondary_profile(self, hermes_root):
        """Cross-profile auth stays closed: owner key must not read lokaj."""
        adapter = _make_adapter(multiplex=True)
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            status, _ = await _enabled_toolsets(
                cli, "/p/lokaj/v1/toolsets", OWNER_KEY
            )
            assert status == 401

    @pytest.mark.asyncio
    async def test_unknown_profile_is_404(self, hermes_root):
        adapter = _make_adapter(multiplex=True)
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            resp = await cli.get(
                "/p/ghost/v1/toolsets",
                headers={"Authorization": f"Bearer {OWNER_KEY}"},
            )
            assert resp.status == 404


class TestMultiplexOffPrefixFailsClosed:
    """Single-profile gateways must not serve another profile's URL."""

    def test_foreign_prefix_rejected(self, hermes_root):
        adapter = _make_adapter(multiplex=False)

        class _Req:
            match_info = {"profile": "lokaj"}

        assert adapter._resolve_request_profile(_Req()) is _PROFILE_REJECTED

    def test_self_referential_prefix_falls_through(self, hermes_root):
        """/p/default/ on the default-profile gateway keeps working."""
        adapter = _make_adapter(multiplex=False)

        class _Req:
            match_info = {"profile": "default"}

        assert adapter._resolve_request_profile(_Req()) is None

    def test_own_named_profile_prefix_falls_through(self, hermes_root, monkeypatch):
        """A gateway launched FOR profile lokaj accepts /p/lokaj/…"""
        monkeypatch.setenv(
            "HERMES_HOME", str(hermes_root / "profiles" / "lokaj")
        )
        adapter = _make_adapter(multiplex=False)

        class _Req:
            match_info = {"profile": "lokaj"}

        assert adapter._resolve_request_profile(_Req()) is None

    @pytest.mark.asyncio
    async def test_foreign_prefix_is_404_end_to_end(self, hermes_root):
        adapter = _make_adapter(multiplex=False)
        async with TestClient(TestServer(_make_app(adapter))) as cli:
            resp = await cli.get(
                "/p/lokaj/v1/toolsets",
                headers={"Authorization": f"Bearer {OWNER_KEY}"},
            )
            assert resp.status == 404
            # Bare route unaffected.
            status, enabled = await _enabled_toolsets(
                cli, "/v1/toolsets", OWNER_KEY
            )
            assert status == 200
            assert "computer_use" not in enabled


class TestWebhookMultiplexOffPrefixFailsClosed:
    """Same bug class in the webhook adapter's prefix resolver."""

    def _adapter(self, multiplex: bool):
        from gateway.platforms.webhook import WebhookAdapter, _PROFILE_REJECTED

        class _Runner:
            config = GatewayConfig(multiplex_profiles=multiplex)

        adapter = WebhookAdapter.__new__(WebhookAdapter)
        adapter.gateway_runner = _Runner()
        return adapter, _PROFILE_REJECTED

    def test_foreign_prefix_rejected(self, hermes_root):
        adapter, rejected = self._adapter(multiplex=False)

        class _Req:
            match_info = {"profile": "lokaj"}

        assert adapter._resolve_request_profile(_Req()) is rejected

    def test_self_referential_prefix_falls_through(self, hermes_root):
        adapter, _rejected = self._adapter(multiplex=False)

        class _Req:
            match_info = {"profile": "default"}

        assert adapter._resolve_request_profile(_Req()) is None
