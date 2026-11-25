"""Tests for the capability-gated platform action facade (#64176, action half).

Covers:
* gate default-off: no grant → ``capability_not_granted`` structured error,
  no adapter touched
* capability grant honored via ``granted_capabilities`` AND via the legacy
  ``allow_platform_actions`` config key
* unknown platform / unregistered adapter / disconnected adapter → structured
  errors, never exceptions
* verbs route to the right adapter primitives (telegram ``_set_reaction`` /
  ``rename_dm_topic``; discord ``rename_thread``)
* adapter-layer exceptions surface as ``action_failed`` results, never raise
* ``ctx.platform_actions`` facade is bound to the plugin id
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from hermes_cli.platform_actions import PlatformActions


def _grant(granted: bool):
    """Patch the capability check the facade performs."""
    return patch(
        "hermes_cli.plugin_capabilities.plugin_capability_granted",
        return_value=granted,
    )


def _runner_with(adapters: dict):
    runner = SimpleNamespace(adapters=adapters)
    return patch("gateway.run._gateway_runner_ref", lambda: runner)


def _multiplex_runner_with(*, default: dict, profiles: dict, active_profile: str = "default", config=None):
    """A runner using the REAL GatewayAuthorizationMixin resolution ladder."""
    from gateway.authz_mixin import GatewayAuthorizationMixin

    runner = GatewayAuthorizationMixin.__new__(GatewayAuthorizationMixin)
    runner.adapters = default
    runner._profile_adapters = profiles
    runner._active_profile_name = lambda: active_profile
    if config is not None:
        runner.config = config
    return patch("gateway.run._gateway_runner_ref", lambda: runner)


def _telegram_adapter(connected=True):
    a = MagicMock()
    a.platform = Platform.TELEGRAM
    a.is_connected = connected
    a._set_reaction = AsyncMock(return_value=True)
    a.rename_dm_topic = AsyncMock(return_value=None)
    return a


def _discord_adapter(connected=True):
    a = MagicMock()
    a.platform = Platform.DISCORD
    a.is_connected = connected
    a.rename_thread = AsyncMock(return_value=True)
    return a


class TestGateDefaultOff:
    def test_no_grant_returns_structured_error(self):
        actions = PlatformActions("some-plugin")
        adapter = _telegram_adapter()

        with _grant(False), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(
                actions.add_reaction("telegram", "123", "456", "\U0001F44D")
            )

        assert result["ok"] is False
        assert result["error"] == "capability_not_granted"
        adapter._set_reaction.assert_not_awaited()

    def test_default_config_is_off_via_real_capability_check(self):
        """No patching of the check itself: an empty config entry denies."""
        actions = PlatformActions("some-plugin")
        with patch(
            "hermes_cli.plugin_capabilities._plugin_entry", return_value={}
        ):
            result = asyncio.run(
                actions.set_thread_title("telegram", "1", "2", "t")
            )
        assert result == {
            "ok": False,
            "error": "capability_not_granted",
            "detail": result["detail"],
        }

    def test_legacy_allow_platform_actions_key_grants(self):
        actions = PlatformActions("some-plugin")
        adapter = _telegram_adapter()
        with patch(
            "hermes_cli.plugin_capabilities._plugin_entry",
            return_value={"allow_platform_actions": True},
        ), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(
                actions.add_reaction("telegram", "123", "456", "\U0001F44D")
            )
        assert result["ok"] is True

    def test_granted_capabilities_list_grants(self):
        actions = PlatformActions("some-plugin")
        adapter = _telegram_adapter()
        with patch(
            "hermes_cli.plugin_capabilities._plugin_entry",
            return_value={"granted_capabilities": ["gateway.platform_actions"]},
        ), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(
                actions.add_reaction("telegram", "123", "456", "\U0001F44D")
            )
        assert result["ok"] is True

    def test_capability_check_failure_fails_closed(self):
        actions = PlatformActions("some-plugin")
        with patch(
            "hermes_cli.plugin_capabilities.plugin_capability_granted",
            side_effect=RuntimeError("corrupt config"),
        ):
            result = asyncio.run(
                actions.add_reaction("telegram", "1", "2", "x")
            )
        assert result["error"] == "capability_not_granted"


class TestStructuredErrors:
    def test_no_gateway_runner(self):
        actions = PlatformActions("p")
        with _grant(True), patch("gateway.run._gateway_runner_ref", lambda: None):
            result = asyncio.run(actions.add_reaction("telegram", "1", "2", "x"))
        assert result["error"] == "gateway_unavailable"

    def test_unknown_platform(self):
        actions = PlatformActions("p")
        with _grant(True), _runner_with({}):
            result = asyncio.run(actions.add_reaction("smoke-signals", "1", "2", "x"))
        assert result["error"] == "unknown_platform"

    def test_adapter_not_registered(self):
        actions = PlatformActions("p")
        with _grant(True), _runner_with({}):
            result = asyncio.run(actions.add_reaction("telegram", "1", "2", "x"))
        assert result["error"] == "adapter_not_registered"

    def test_adapter_disconnected(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter(connected=False)
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.add_reaction("telegram", "1", "2", "x"))
        assert result["error"] == "adapter_disconnected"
        adapter._set_reaction.assert_not_awaited()

    @pytest.mark.parametrize("bad", ["", "   ", None, 123])
    def test_invalid_arguments(self, bad):
        actions = PlatformActions("p")
        with _grant(True):
            result = asyncio.run(actions.add_reaction("telegram", "1", "2", bad))
        assert result["error"] == "invalid_argument"

    def test_adapter_exception_becomes_action_failed(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        adapter._set_reaction = AsyncMock(side_effect=RuntimeError("api down"))
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.add_reaction("telegram", "1", "2", "x"))
        assert result["ok"] is False
        assert result["error"] == "action_failed"
        assert "api down" in result["detail"]

    def test_unsupported_platform_action(self):
        actions = PlatformActions("p")
        adapter = MagicMock()
        adapter.platform = Platform.SLACK
        adapter.is_connected = True
        with _grant(True), _runner_with({Platform.SLACK: adapter}):
            result = asyncio.run(actions.add_reaction("slack", "1", "2", "x"))
        assert result["error"] == "unsupported_platform_action"


class TestVerbRouting:
    def test_telegram_add_reaction_routes_to_set_reaction(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(
                actions.add_reaction("telegram", "-100123", "456", "\U0001F44D")
            )
        assert result == {"ok": True, "action": "add_reaction"}
        adapter._set_reaction.assert_awaited_once_with("-100123", "456", "\U0001F44D")

    def test_telegram_set_reaction_false_is_action_failed(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        adapter._set_reaction = AsyncMock(return_value=False)
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.add_reaction("telegram", "1", "2", "x"))
        assert result["error"] == "action_failed"

    def test_telegram_set_thread_title_routes_to_rename_dm_topic(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(
                actions.set_thread_title("telegram", "123", "42", "New title")
            )
        assert result == {"ok": True, "action": "set_thread_title"}
        adapter.rename_dm_topic.assert_awaited_once_with("123", 42, "New title")

    def test_discord_set_thread_title_routes_to_rename_thread(self):
        actions = PlatformActions("p")
        adapter = _discord_adapter()
        with _grant(True), _runner_with({Platform.DISCORD: adapter}):
            result = asyncio.run(
                actions.set_thread_title("discord", "555", "321", "Renamed")
            )
        assert result == {"ok": True, "action": "set_thread_title"}
        adapter.rename_thread.assert_awaited_once_with("321", "Renamed")

    def test_discord_rename_false_is_action_failed(self):
        actions = PlatformActions("p")
        adapter = _discord_adapter()
        adapter.rename_thread = AsyncMock(return_value=False)
        with _grant(True), _runner_with({Platform.DISCORD: adapter}):
            result = asyncio.run(
                actions.set_thread_title("discord", "555", "321", "Renamed")
            )
        assert result["error"] == "action_failed"

    def test_discord_add_reaction_fetches_and_reacts(self):
        actions = PlatformActions("p")
        adapter = _discord_adapter()
        message = MagicMock()
        message.add_reaction = AsyncMock()
        channel = MagicMock()
        channel.fetch_message = AsyncMock(return_value=message)
        client = MagicMock()
        client.get_channel = MagicMock(return_value=channel)
        adapter._client = client
        with _grant(True), _runner_with({Platform.DISCORD: adapter}):
            result = asyncio.run(
                actions.add_reaction("discord", "555", "456", "\U0001F44D")
            )
        assert result == {"ok": True, "action": "add_reaction"}
        channel.fetch_message.assert_awaited_once_with(456)
        message.add_reaction.assert_awaited_once_with("\U0001F44D")

    def test_discord_add_reaction_non_numeric_ids(self):
        actions = PlatformActions("p")
        adapter = _discord_adapter()
        adapter._client = MagicMock()
        with _grant(True), _runner_with({Platform.DISCORD: adapter}):
            result = asyncio.run(
                actions.add_reaction("discord", "not-a-number", "456", "x")
            )
        assert result["error"] == "invalid_argument"


class TestMultiplexProfileRouting:
    """A plugin acting during a secondary profile's turn must act through THAT
    profile's adapter, never the default profile's — the fail-closed contract
    of GatewayAuthorizationMixin._authorization_adapter (#85245)."""

    def test_secondary_profile_routes_to_its_own_adapter_not_default(self):
        actions = PlatformActions("p")
        default_adapter = _telegram_adapter()
        team_b_adapter = _telegram_adapter()
        with (
            _grant(True),
            _multiplex_runner_with(
                default={Platform.TELEGRAM: default_adapter},
                profiles={"team-b": {Platform.TELEGRAM: team_b_adapter}},
            ),
            patch("hermes_cli.profiles.get_active_profile_name", return_value="team-b"),
        ):
            result = asyncio.run(actions.add_reaction("telegram", "1", "2", "x"))
        assert result["ok"] is True
        team_b_adapter._set_reaction.assert_awaited_once()
        default_adapter._set_reaction.assert_not_awaited()

    @pytest.mark.parametrize(
        "resolver",
        [
            {"return_value": "team-b"},              # stamped profile, no registry entry
            {"side_effect": RuntimeError("boom")},  # profile resolution itself fails
        ],
        ids=["no-registry-entry", "resolution-error"],
    )
    def test_unresolvable_profile_fails_closed_never_default_bot(self, resolver):
        actions = PlatformActions("p")
        default_adapter = _telegram_adapter()
        with (
            _grant(True),
            _multiplex_runner_with(default={Platform.TELEGRAM: default_adapter}, profiles={}),
            patch("hermes_cli.profiles.get_active_profile_name", **resolver),
        ):
            result = asyncio.run(actions.add_reaction("telegram", "1", "2", "x"))
        assert result["error"] == "adapter_not_registered"
        default_adapter._set_reaction.assert_not_awaited()

    def test_shared_bot_satellite_acts_only_on_routed_chats(self, tmp_path):
        """A bot-less satellite borrows the primary bot only for a chat a route maps to it (#147602)."""
        from gateway.config import GatewayConfig
        from gateway.profile_routing import parse_profile_routes

        actions = PlatformActions("p")
        primary = _telegram_adapter()
        cfg = GatewayConfig()
        cfg.multiplex_profiles = True
        cfg.profile_routes = parse_profile_routes(
            [{"name": "sec-room", "platform": "telegram", "profile": "sec", "chat_id": "100"}])
        with (
            _grant(True),
            _multiplex_runner_with(default={Platform.TELEGRAM: primary}, profiles={"sec": {}}, config=cfg),
            patch("hermes_cli.profiles.get_active_profile_name", return_value="sec"),
            patch("gateway.run._multiplex_profile_homes", return_value=[("sec", tmp_path / "sec")]),
        ):
            assert asyncio.run(actions.add_reaction("telegram", "100", "2", "x"))["ok"] is True
            primary._set_reaction.reset_mock()
            result = asyncio.run(actions.add_reaction("telegram", "200", "2", "x"))
        assert result["error"] == "adapter_not_registered"
        primary._set_reaction.assert_not_awaited()

    def test_topic_routed_satellite_reacts_to_its_own_turn_message(self, tmp_path):
        """A reaction names no thread, so on a topic-pinned route the turn's own message is matched in
        the bound origin's topic; another message, or the same id in another chat, is not (#147602)."""
        from gateway.config import GatewayConfig
        from gateway.profile_routing import parse_profile_routes
        from gateway.session_context import clear_session_vars, set_session_vars

        actions = PlatformActions("p")
        primary = _telegram_adapter()
        slack = SimpleNamespace(platform=Platform.SLACK, is_connected=True)
        cfg = GatewayConfig()
        cfg.multiplex_profiles = True
        cfg.profile_routes = parse_profile_routes([
            {"name": "topic", "platform": "telegram", "profile": "sec", "chat_id": "100", "thread_id": "7"},
            {"name": "elsewhere", "platform": "telegram", "profile": "sec", "chat_id": "200", "thread_id": "7"},
            {"name": "other-platform", "platform": "slack", "profile": "sec", "chat_id": "100", "thread_id": "7"}])
        tokens = set_session_vars(profile="sec", platform="telegram", chat_id="100", thread_id="7", message_id="2")
        try:
            with (
                _grant(True),
                _multiplex_runner_with(default={Platform.TELEGRAM: primary, Platform.SLACK: slack},
                                       profiles={"sec": {}}, config=cfg),
                patch("hermes_cli.profiles.get_active_profile_name", return_value="sec"),
                patch("gateway.run._multiplex_profile_homes", return_value=[("sec", tmp_path / "sec")]),
            ):
                own = asyncio.run(actions.add_reaction("telegram", "100", "2", "x"))
                other_message = asyncio.run(actions.add_reaction("telegram", "100", "3", "x"))
                other_chat = asyncio.run(actions.add_reaction("telegram", "200", "2", "x"))
                other_platform = asyncio.run(actions.add_reaction("slack", "100", "2", "x"))
        finally:
            clear_session_vars(tokens)
        assert own["ok"] is True
        assert other_message["error"] == other_chat["error"] == other_platform["error"] == "adapter_not_registered"
        primary._set_reaction.assert_awaited_once()

    def test_shared_bot_satellite_cannot_rename_foreign_discord_thread(self, tmp_path):
        """Discord renames by global thread id: a routed chat_id must not unlock an arbitrary thread
        for a bot-less satellite; only a thread its route pins is renamable (#147602)."""
        from gateway.config import GatewayConfig
        from gateway.profile_routing import parse_profile_routes

        actions = PlatformActions("p")
        primary = _discord_adapter()
        cfg = GatewayConfig()
        cfg.multiplex_profiles = True
        cfg.profile_routes = parse_profile_routes([
            {"name": "d", "platform": "discord", "profile": "sec", "chat_id": "111"},
            {"name": "dt", "platform": "discord", "profile": "sec", "chat_id": "111", "thread_id": "333"},
        ])
        with (
            _grant(True),
            _multiplex_runner_with(default={Platform.DISCORD: primary}, profiles={"sec": {}}, config=cfg),
            patch("hermes_cli.profiles.get_active_profile_name", return_value="sec"),
            patch("gateway.run._multiplex_profile_homes", return_value=[("sec", tmp_path / "sec")]),
        ):
            result = asyncio.run(actions.set_thread_title("discord", "111", "999", "pwned"))
            assert result["ok"] is not True
            primary.rename_thread.assert_not_awaited()
            assert asyncio.run(actions.set_thread_title("discord", "111", "333", "ok"))["ok"] is True
        primary.rename_thread.assert_awaited_once_with("333", "ok")


class TestPluginContextWiring:
    def test_ctx_platform_actions_bound_to_plugin_id(self):
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

        manager = PluginManager()
        ctx = PluginContext(
            PluginManifest(name="actions-fixture", source="user"), manager,
        )
        facade = ctx.platform_actions
        assert isinstance(facade, PlatformActions)
        assert facade._plugin_id == "actions-fixture"
        # Cached: property returns the same instance.
        assert ctx.platform_actions is facade
