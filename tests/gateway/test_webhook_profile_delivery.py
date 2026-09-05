"""Invariant: webhook cross-platform delivery must honor the route's profile binding.

A webhook route bound to ``profile: X`` (config key or /p/X/ URL prefix) with
``deliver: <platform>`` must deliver its response through profile X's adapter —
the bot credential X actually connected with — never the first-connected adapter
for that platform. Under multiplex, ``_find_adapter`` falls back to insertion order
over ``_profile_adapters``; for two Discord profiles that is alphabetical, so the
final response posts from the wrong bot (live 2026-09-04: the ai-coach-events route
bound to ``trainer`` posted as the accountant bot 1542626633670991913 instead of
the trainer bot 1542632896924221460).

Covered here:
- agent-mode route: the route profile reaches the delivery resolver via
  ``_delivery_info`` (through ``_dispatch_agent_run``) and selects profile X's adapter;
- deliver_only routes share the same egress and must honor the binding too;
- the /p/<profile>/ URL prefix flows end-to-end (HTTP boundary);
- un-profiled routes keep using the primary adapter (no regression);
- fail-closed: a bound profile whose platform adapter never connected returns a
  not-connected error — it must NOT fall back to the primary/first-connected adapter,
  which would leak the response through another profile's bot.
"""
import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(routes=None):
    return PlatformConfig(enabled=True, extra={
        "host": "0.0.0.0", "port": 0, "routes": routes or {},
        "rate_limit": 30, "max_body_bytes": 1_048_576,
    })


def _make_adapter(routes=None):
    return WebhookAdapter(_make_config(routes=routes))


def _discord(bot_id):
    """Mock Discord adapter identified by the bot id it would post as."""
    bot = AsyncMock()
    bot.platform = Platform.DISCORD
    bot.bot_user_id = bot_id
    bot.send = AsyncMock(return_value=SendResult(success=True))
    return bot


def _profile_runner(primary_discord, profile_discords=None):
    """Bare GatewayRunner exposing the REAL ``_authorization_adapter`` mixin.

    Mirrors the construction in tests/gateway/test_multiplex_interactive_auth.py:
    ``object.__new__`` plus exactly the attributes the mixin reads (``config``,
    ``adapters``, ``_profile_adapters``, ``_primary_profile_name``).
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._primary_profile_name = "default"
    runner.adapters = {} if primary_discord is None else {Platform.DISCORD: primary_discord}
    runner._profile_adapters = profile_discords or {}
    return runner


def _adapter_with_delivery(adapter, delivery, chat_id="webhook:ai-coach-events:d-1"):
    """Register a delivery entry the way ``_dispatch_agent_run`` would."""
    adapter._delivery_info[chat_id] = dict(delivery)
    adapter._delivery_info_created[chat_id] = time.time()
    return chat_id


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application(client_max_size=adapter._max_body_bytes)
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    app.router.add_post("/p/{profile}/webhooks/{route_name}", adapter._handle_webhook)
    return app


# ---------------------------------------------------------------------------
# Agent-mode delivery: send() must resolve the bound profile's adapter
# ---------------------------------------------------------------------------

class TestProfileBoundAgentDelivery:

    def _adapter(self, delivery, runner):
        adapter = _make_adapter()
        adapter.gateway_runner = runner
        chat_id = _adapter_with_delivery(adapter, delivery)
        return adapter, chat_id

    @pytest.mark.asyncio
    async def test_profile_bound_route_delivers_via_profile_adapter(self):
        """Route bound to trainer + deliver:discord posts via TRAINER's bot, even though
        the primary (first-connected) Discord adapter belongs to accountant."""
        accountant, trainer = _discord("accountant"), _discord("trainer")
        runner = _profile_runner(
            primary_discord=accountant,
            profile_discords={"trainer": {Platform.DISCORD: trainer}},
        )
        adapter, chat_id = self._adapter(
            {"deliver": "discord", "deliver_extra": {"chat_id": "-100123"}, "profile": "trainer"},
            runner,
        )

        result = await adapter.send(chat_id, "workout summary")

        assert result.success is True
        trainer.send.assert_awaited_once()
        accountant.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unprofiled_route_still_uses_primary_adapter(self):
        """No profile in the delivery entry → primary adapter (existing behaviour)."""
        accountant, trainer = _discord("accountant"), _discord("trainer")
        runner = _profile_runner(
            primary_discord=accountant,
            profile_discords={"trainer": {Platform.DISCORD: trainer}},
        )
        adapter, chat_id = self._adapter(
            {"deliver": "discord", "deliver_extra": {"chat_id": "-100123"}},
            runner,
        )

        result = await adapter.send(chat_id, "ci summary")

        assert result.success is True
        accountant.send.assert_awaited_once()
        trainer.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bound_profile_without_adapter_fails_closed(self):
        """Trainer bound but its Discord never connected → not-connected error, never a
        silent fallback through the primary (accountant) bot."""
        accountant = _discord("accountant")
        runner = _profile_runner(primary_discord=accountant, profile_discords={"trainer": {}})
        adapter, chat_id = self._adapter(
            {"deliver": "discord", "deliver_extra": {"chat_id": "-100123"}, "profile": "trainer"},
            runner,
        )

        result = await adapter.send(chat_id, "workout summary")

        assert result.success is False
        assert "not connected" in (result.error or "")
        accountant.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# The route profile must reach _delivery_info via _dispatch_agent_run
# ---------------------------------------------------------------------------

class TestDispatchRecordsRouteProfile:

    @pytest.mark.asyncio
    async def test_dispatch_agent_run_stores_route_profile(self):
        adapter = _make_adapter(routes={"ai-coach-events": {"deliver": "discord"}})
        adapter.handle_message = AsyncMock()

        class _Req:
            method = "POST"

        adapter._dispatch_agent_run(
            _Req(), {"deliver": "discord"}, "ai-coach-events", "trainer",
            {"n": 1}, "tick", "push", "d-1", time.time(),
        )

        info = adapter._delivery_info["webhook:ai-coach-events:d-1"]
        assert info["deliver"] == "discord"
        assert info["profile"] == "trainer"


# ---------------------------------------------------------------------------
# deliver_only routes share the same egress helper
# ---------------------------------------------------------------------------

class TestDeliverOnlyProfileDelivery:

    @pytest.mark.asyncio
    async def test_deliver_only_honors_route_profile(self):
        accountant, trainer = _discord("accountant"), _discord("trainer")
        runner = _profile_runner(
            primary_discord=accountant,
            profile_discords={"trainer": {Platform.DISCORD: trainer}},
        )
        adapter = _make_adapter()
        adapter.gateway_runner = runner

        await adapter._handle_deliver_only(
            "outbound msg", {},
            {"deliver": "discord", "profile": "trainer", "deliver_extra": {"chat_id": "-100123"}},
            "notify", "event", "d-9",
        )

        trainer.send.assert_awaited_once()
        accountant.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deliver_only_without_profile_uses_primary(self):
        accountant, trainer = _discord("accountant"), _discord("trainer")
        runner = _profile_runner(
            primary_discord=accountant,
            profile_discords={"trainer": {Platform.DISCORD: trainer}},
        )
        adapter = _make_adapter()
        adapter.gateway_runner = runner

        await adapter._handle_deliver_only(
            "outbound msg", {},
            {"deliver": "discord", "deliver_extra": {"chat_id": "-100123"}},
            "notify", "event", "d-9",
        )

        accountant.send.assert_awaited_once()
        trainer.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# HTTP boundary: /p/<profile>/ prefix flows end-to-end
# ---------------------------------------------------------------------------

class TestProfilePrefixHttpDelivery:

    @pytest.mark.asyncio
    async def test_url_profile_prefix_delivers_via_profile_adapter(self, monkeypatch, tmp_path):
        """POST /p/trainer/webhooks/ai-coach-events → final response posts from the
        trainer bot, not the primary (accountant) bot."""
        accountant, trainer = _discord("accountant"), _discord("trainer")
        runner = _profile_runner(
            primary_discord=accountant,
            profile_discords={"trainer": {Platform.DISCORD: trainer}},
        )
        routes = {"ai-coach-events": {
            "secret": _INSECURE_NO_AUTH, "prompt": "tick", "deliver": "discord",
            "deliver_extra": {"chat_id": "-100123"}, "profile": "trainer",
        }}
        adapter = _make_adapter(routes=routes)
        adapter.gateway_runner = runner

        # Simulate the runner side of the real flow: the agent turn's final response is
        # delivered through WebhookAdapter.send() keyed on the session chat_id.
        async def _fake_agent_turn(event):
            await adapter.send(event.source.chat_id, "workout summary")

        adapter.handle_message = _fake_agent_turn

        # _resolve_request_profile / _profile_scope import these lazily from the module.
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex, profile_allowlist=None: [("default", "/x"), ("trainer", "/x")],
        )
        trainer_home = tmp_path / "profiles" / "trainer"
        trainer_home.mkdir(parents=True)
        (trainer_home / ".env").write_text("", encoding="utf-8")
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: tmp_path / "profiles" / name)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/p/trainer/webhooks/ai-coach-events",
                json={"n": 1},
                headers={"X-GitHub-Delivery": "d-http-1"},
            )
            assert resp.status == 202

        # handle_message is spawned as a fire-and-forget task — let it finish.
        for _ in range(100):
            if trainer.send.await_count or accountant.send.await_count:
                break
            await asyncio.sleep(0.01)

        # The route profile reached the delivery record…
        info = adapter._delivery_info["webhook:ai-coach-events:d-http-1"]
        assert info["profile"] == "trainer"
        # …and egress used the profile's adapter, not the primary bot.
        trainer.send.assert_awaited_once()
        accountant.send.assert_not_awaited()
