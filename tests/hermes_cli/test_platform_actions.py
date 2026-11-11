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


def _multiplex_runner_with(*, default: dict, profiles: dict, active_profile: str = "default"):
    """A runner using the REAL GatewayAuthorizationMixin resolution ladder."""
    from gateway.authz_mixin import GatewayAuthorizationMixin

    runner = GatewayAuthorizationMixin.__new__(GatewayAuthorizationMixin)
    runner.adapters = default
    runner._profile_adapters = profiles
    runner._active_profile_name = lambda: active_profile
    return patch("gateway.run._gateway_runner_ref", lambda: runner)


def _telegram_adapter(connected=True):
    a = MagicMock()
    a.platform = Platform.TELEGRAM
    a.is_connected = connected
    a._set_reaction = AsyncMock(return_value=True)
    a.rename_dm_topic = AsyncMock(return_value=None)
    a.send_widget = AsyncMock(return_value={"message_id": 501, "date": 0})
    a.edit_widget_data = AsyncMock(return_value=True)
    a.answer_widget_query = AsyncMock(return_value=None)
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


class TestTelegramWidgetVerbs:
    """Bot API 10.4 HTML widgets: the verbs route to the Telegram adapter's widget calls, carry the
    identifiers a plugin needs next, and refuse malformed input before touching the adapter."""

    def test_send_widget_returns_the_message_id_as_text(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.send_widget(
                "telegram", "88", "<b>hi</b>", chat_data='{"n":1}', caption="Dashboard", disable_notification=True))
        assert result == {"ok": True, "action": "send_widget", "message_id": "501"}
        adapter.send_widget.assert_awaited_once_with(
            "88", "<b>hi</b>", chat_data='{"n":1}', public_data=None, caption="Dashboard",
            allow_queries=True, disable_notification=True, thread_id=None, width=None, height=None)

    def test_send_widget_passes_the_widget_size(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.send_widget("telegram", "88", "<b>hi</b>", width=320, height=240))
        assert result["ok"] is True
        assert adapter.send_widget.await_args.kwargs == {
            "chat_data": None, "public_data": None, "caption": None, "allow_queries": True,
            "disable_notification": False, "thread_id": None, "width": 320, "height": 240}

    @pytest.mark.parametrize("size", [{"width": 0}, {"height": -1}, {"width": "320"}, {"height": 2.5},
                                      {"width": True}])
    def test_the_widget_size_must_be_a_positive_integer(self, size):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.send_widget("telegram", "88", "<b>hi</b>", **size))
        assert result["error"] == "invalid_argument"
        adapter.send_widget.assert_not_awaited()

    def test_send_widget_without_a_message_back_is_action_failed(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        adapter.send_widget = AsyncMock(return_value=True)
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.send_widget("telegram", "88", "<b>hi</b>"))
        assert result["error"] == "action_failed"

    @pytest.mark.parametrize("bad", ["", "   ", None, 123])
    def test_send_widget_needs_html(self, bad):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.send_widget("telegram", "88", bad))
        assert result["error"] == "invalid_argument"
        adapter.send_widget.assert_not_awaited()

    def test_optional_packets_must_be_text(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.send_widget("telegram", "88", "<b>hi</b>", chat_data={"n": 1}))
        assert result["error"] == "invalid_argument"
        adapter.send_widget.assert_not_awaited()

    def test_edit_widget_data_routes_and_reports_a_change(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.edit_widget_data("telegram", "88", "472", chat_data='{"n":2}'))
        assert result == {"ok": True, "action": "edit_widget_data", "modified": True}
        adapter.edit_widget_data.assert_awaited_once_with("88", "472", chat_data='{"n":2}', public_data=None)

    def test_edit_widget_data_with_the_same_packet_is_success_not_modified(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        adapter.edit_widget_data = AsyncMock(return_value=False)
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.edit_widget_data("telegram", "88", "472", public_data=""))
        assert result == {"ok": True, "action": "edit_widget_data", "modified": False}

    def test_edit_widget_data_needs_a_packet(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.edit_widget_data("telegram", "88", "472"))
        assert result["error"] == "invalid_argument"
        adapter.edit_widget_data.assert_not_awaited()

    def test_answer_widget_query_routes(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(True), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.answer_widget_query("telegram", "q1", data='{"ok":1}'))
        assert result == {"ok": True, "action": "answer_widget_query"}
        adapter.answer_widget_query.assert_awaited_once_with("q1", '{"ok":1}')

    def test_widget_verbs_are_telegram_only(self):
        actions = PlatformActions("p")
        adapter = _discord_adapter()
        with _grant(True), _runner_with({Platform.DISCORD: adapter}):
            results = [asyncio.run(call) for call in (
                actions.send_widget("discord", "1", "<b>x</b>"),
                actions.edit_widget_data("discord", "1", "2", chat_data="{}"),
                actions.answer_widget_query("discord", "q1"),
            )]
        assert [result["error"] for result in results] == ["unsupported_platform_action"] * 3

    def test_widget_verbs_are_gated_like_the_others(self):
        actions = PlatformActions("p")
        adapter = _telegram_adapter()
        with _grant(False), _runner_with({Platform.TELEGRAM: adapter}):
            result = asyncio.run(actions.send_widget("telegram", "88", "<b>hi</b>"))
        assert result["error"] == "capability_not_granted"
        adapter.send_widget.assert_not_awaited()


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
