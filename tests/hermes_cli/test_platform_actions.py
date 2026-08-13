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
from hermes_cli.platform_actions import CAPABILITY_ID, PlatformActions
from hermes_cli.plugin_capabilities import CAPABILITY_REGISTRY


def _grant(granted: bool):
    """Patch the capability check the facade performs."""
    return patch(
        "hermes_cli.plugin_capabilities.plugin_capability_granted",
        return_value=granted,
    )


def _runner_with(adapters: dict):
    runner = SimpleNamespace(adapters=adapters)
    return patch("gateway.run._gateway_runner_ref", lambda: runner)


class _MultiplexRunner:
    """Minimal stand-in for GatewayRunner's real multiplex adapter split —
    mirrors GatewayAuthorizationMixin._authorization_adapter's fail-closed
    contract (default profile -> self.adapters; secondary profile -> its own
    entry in self._profile_adapters, never falling back to another
    profile's adapter of the same platform)."""

    def __init__(self, adapters: dict, profile_adapters: dict, active_profile: str = "default"):
        self.adapters = adapters
        self._profile_adapters = profile_adapters
        self._active_profile = active_profile

    def _active_profile_name(self):
        return self._active_profile

    def _authorization_adapter(self, platform, profile=None):
        profile_name = (profile or "").strip() or None
        if profile_name and profile_name != "default":
            if profile_name == self._active_profile:
                return self.adapters.get(platform)
            if profile_name in self._profile_adapters:
                return self._profile_adapters[profile_name].get(platform)
            return None
        return self.adapters.get(platform)


def _multiplex_runner_with(*, default: dict, profiles: dict, active_profile: str = "default"):
    runner = _MultiplexRunner(default, profiles, active_profile)
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


class TestCapabilityRegistry:
    def test_gateway_platform_actions_registered(self):
        spec = CAPABILITY_REGISTRY.get("gateway.platform_actions")
        assert spec is not None
        assert spec.legacy_path == ("allow_platform_actions",)
        assert spec.description

    def test_facade_uses_registered_id(self):
        assert CAPABILITY_ID == "gateway.platform_actions"


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
    """A plugin scoped to a secondary profile must act through THAT
    profile's adapter, never the default profile's — the same invariant
    gateway/authz_mixin.py's _authorization_adapter enforces for every
    other adapter-resolution path in this codebase."""

    def test_secondary_profile_routes_to_its_own_adapter_not_default(self):
        actions = PlatformActions("p")
        default_adapter = _telegram_adapter()
        team_b_adapter = _telegram_adapter()
        with (
            _grant(True),
            _multiplex_runner_with(
                default={Platform.TELEGRAM: default_adapter},
                profiles={"team-b": {Platform.TELEGRAM: team_b_adapter}},
                active_profile="default",
            ),
            patch(
                "hermes_cli.profiles.get_active_profile_name",
                return_value="team-b",
            ),
        ):
            result = asyncio.run(
                actions.add_reaction("telegram", "1", "2", "x")
            )
        assert result["ok"] is True
        team_b_adapter._set_reaction.assert_awaited_once()
        default_adapter._set_reaction.assert_not_awaited()

    def test_secondary_profile_with_no_registry_entry_fails_closed(self):
        """A stamped secondary profile whose adapter isn't registered must
        refuse — never silently fall back to the default profile's bot."""
        actions = PlatformActions("p")
        default_adapter = _telegram_adapter()
        with (
            _grant(True),
            _multiplex_runner_with(
                default={Platform.TELEGRAM: default_adapter},
                profiles={},
                active_profile="default",
            ),
            patch(
                "hermes_cli.profiles.get_active_profile_name",
                return_value="team-b",
            ),
        ):
            result = asyncio.run(
                actions.add_reaction("telegram", "1", "2", "x")
            )
        assert result["error"] == "adapter_not_registered"
        default_adapter._set_reaction.assert_not_awaited()

    def test_default_profile_still_routes_to_default_adapter(self):
        """Healthy path unchanged: the default profile keeps using
        runner.adapters via the same _authorization_adapter call."""
        actions = PlatformActions("p")
        default_adapter = _telegram_adapter()
        with (
            _grant(True),
            _multiplex_runner_with(
                default={Platform.TELEGRAM: default_adapter},
                profiles={},
                active_profile="default",
            ),
            patch(
                "hermes_cli.profiles.get_active_profile_name",
                return_value="default",
            ),
        ):
            result = asyncio.run(
                actions.add_reaction("telegram", "1", "2", "x")
            )
        assert result["ok"] is True
        default_adapter._set_reaction.assert_awaited_once()

    def test_active_profile_matching_secondary_name_uses_default_adapters(self):
        """A profile that IS the process's own active profile resolves via
        runner.adapters (not _profile_adapters), mirroring
        _authorization_adapter's active-profile shortcut."""
        actions = PlatformActions("p")
        default_adapter = _telegram_adapter()
        with (
            _grant(True),
            _multiplex_runner_with(
                default={Platform.TELEGRAM: default_adapter},
                profiles={},
                active_profile="team-b",
            ),
            patch(
                "hermes_cli.profiles.get_active_profile_name",
                return_value="team-b",
            ),
        ):
            result = asyncio.run(
                actions.add_reaction("telegram", "1", "2", "x")
            )
        assert result["ok"] is True
        default_adapter._set_reaction.assert_awaited_once()


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
