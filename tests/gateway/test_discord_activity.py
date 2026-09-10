"""Tests for Discord bot activity (rich presence) feature.

Covers:
- _load_discord_activity_config: enabled false/true, type/state, case
  handling, missing fields, non-dict activity, config read errors
- _apply_activity: template expansion ({{model}}, {{profile}}), type
  mapping, disabled skip, client=None, change_presence exception,
  async change_presence actually awaited, config disabled clears
- Watchdog diff guard: no re-send while rendered state is unchanged
- _refresh_discord_activity on GatewayRunner: adapter lookup, no-op when absent
- Config defaults: discord.activity present and enabled=False
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_discord_connect import _ensure_discord_mock

_ensure_discord_mock()

import discord  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

_CONFIG_MODULE = "hermes_cli.config"
_PROFILES_MODULE = "hermes_cli.profiles"


def _make_adapter(client: MagicMock | None = None) -> DiscordAdapter:
    # Pin the watchdog interval to the default regardless of any local
    # config.yaml — __init__ reads discord.activity_check_interval_seconds.
    with patch(f"{_CONFIG_MODULE}.read_raw_config", return_value={}):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._client = client
    adapter._last_activity_state = None
    return adapter


def _mock_discord_cfg(activity: dict) -> dict:
    return {"discord": {"activity": activity}}


def _client_with_async_presence() -> MagicMock:
    """Client mock whose change_presence is an AsyncMock (real API is a coroutine)."""
    client = MagicMock()
    client.change_presence = AsyncMock()
    return client


# _load_discord_activity_config tests


class TestLoadDiscordActivityConfig:

    def test_disabled_by_default(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({})):
            result = adapter._load_discord_activity_config()
        assert result == {}

    def test_disabled_explicit_false(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({"enabled": False})):
            result = adapter._load_discord_activity_config()
        assert result == {}

    def test_enabled_true_with_watching_type(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "type": "watching",
                       "state": "Qwen/Qwen3.6-27B",
                   })):
            result = adapter._load_discord_activity_config()
        assert result == {"type": "watching", "state": "Qwen/Qwen3.6-27B"}

    @pytest.mark.parametrize("atype", ["playing", "listening", "competing"])
    def test_all_activity_types_accepted(self, atype):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "type": atype, "state": "hermes-agent",
                   })):
            result = adapter._load_discord_activity_config()
        assert result == {"type": atype, "state": "hermes-agent"}

    def test_missing_type_returns_empty(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "state": "something"
                   })):
            result = adapter._load_discord_activity_config()
        assert result == {}

    def test_missing_state_returns_empty(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "type": "watching"
                   })):
            result = adapter._load_discord_activity_config()
        assert result == {}

    def test_activity_not_a_dict_returns_empty(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({"activity": "invalid"})):
            result = adapter._load_discord_activity_config()
        assert result == {}

    def test_no_activity_section_returns_empty(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({})):
            result = adapter._load_discord_activity_config()
        assert result == {}

    def test_config_read_error_returns_empty(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   side_effect=Exception("config file unreadable")):
            result = adapter._load_discord_activity_config()
        assert result == {}

    def test_type_case_insensitive(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "type": "  Watching  ", "state": "test",
                   })):
            result = adapter._load_discord_activity_config()
        assert result["type"] == "watching"

    def test_state_whitespace_stripped(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "type": "watching",
                       "state": "  my status  ",
                   })):
            result = adapter._load_discord_activity_config()
        assert result["state"] == "my status"

    def test_details_preserved(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "type": "watching",
                       "state": "test", "details": "on {{profile}}",
                   })):
            result = adapter._load_discord_activity_config()
        assert result["details"] == "on {{profile}}"


# _apply_activity tests


class TestApplyActivity:

    def setup_method(self):
        discord.Activity.reset_mock()
        at = discord.ActivityType
        if hasattr(at, "playing") and hasattr(at.playing, "_mock_name"):
            at.playing.reset_mock()
            at.watching.reset_mock()
            at.listening.reset_mock()
            at.competing.reset_mock()
        else:
            at.reset_mock()

    def _assert_activity_created(self, expected_type, expected_name,
                                 expected_details=None):
        assert discord.Activity.call_count == 1
        call_kwargs = discord.Activity.call_args.kwargs
        assert call_kwargs["name"] == expected_name
        assert call_kwargs["type"] is getattr(discord.ActivityType, expected_type, None)
        if expected_details is not None:
            assert call_kwargs["details"] == expected_details
        else:
            assert "details" not in call_kwargs

    def _assert_change_presence_awaited(self, client: MagicMock):
        assert client.change_presence.await_count == 1
        call_kwargs = client.change_presence.call_args.kwargs
        assert "activity" in call_kwargs

    def _do_apply(self, read_cfg, load_cfg, profile="default"):
        client = _client_with_async_presence()
        adapter = _make_adapter(client)
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg(read_cfg)):
            with patch(f"{_CONFIG_MODULE}.load_config_readonly",
                       return_value=load_cfg):
                with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                           return_value=profile):
                    asyncio.run(adapter._apply_activity())
        return client

    def test_enabled_sends_presence(self):
        client = self._do_apply(
            {"enabled": True, "type": "watching", "state": "test"},
            {"model": {"default": "gpt-4"}},
            profile="coder",
        )
        self._assert_activity_created("watching", "test")
        self._assert_change_presence_awaited(client)

    def test_change_presence_is_awaited(self):
        """Regression: change_presence is a coroutine in discord.py — if the
        adapter calls it without await, the presence is never sent and this
        AsyncMock would show zero awaited calls."""
        client = self._do_apply(
            {"enabled": True, "type": "watching", "state": "test"},
            {},
        )
        assert client.change_presence.await_count == 1
        assert client.change_presence.call_count == 1

    def test_disabled_does_not_send_presence(self):
        client = _client_with_async_presence()
        adapter = _make_adapter(client)
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({"enabled": False})):
            asyncio.run(adapter._apply_activity())
        assert client.change_presence.await_count == 0
        assert discord.Activity.call_count == 0

    def test_model_template_resolved(self):
        client = self._do_apply(
            {"enabled": True, "type": "watching", "state": "{{model}}"},
            {"model": {"default": "Qwen3.6-27B"}},
        )
        self._assert_activity_created("watching", "Qwen3.6-27B")

    def test_profile_template_resolved_from_active_profile(self):
        client = self._do_apply(
            {"enabled": True, "type": "watching", "state": "{{profile}}"},
            {},
            profile="coder",
        )
        self._assert_activity_created("watching", "coder")

    def test_both_templates_resolved(self):
        client = self._do_apply(
            {"enabled": True, "type": "playing",
             "state": "{{model}} on {{profile}}"},
            {"model": {"default": "gpt-4"}},
            profile="coder",
        )
        self._assert_activity_created("playing", "gpt-4 on coder")

    def test_details_template_resolved(self):
        client = self._do_apply(
            {"enabled": True, "type": "watching",
             "state": "hermes", "details": "on {{profile}}"},
            {"model": {"default": "gpt-4"}},
            profile="coder",
        )
        self._assert_activity_created("watching", "hermes",
                                      expected_details="on coder")

    def test_long_model_name_truncated_to_128(self):
        """Discord rejects activity names > 128 chars (HTTP 400); a long
        custom model ID must be truncated, not sent as-is."""
        long_model = "custom-provider/" + "x" * 200
        client = self._do_apply(
            {"enabled": True, "type": "watching", "state": "{{model}}"},
            {"model": {"default": long_model}},
        )
        call_kwargs = discord.Activity.call_args.kwargs
        assert len(call_kwargs["name"]) == 128
        self._assert_change_presence_awaited(client)

    def test_truncation_applies_to_details(self):
        client = self._do_apply(
            {"enabled": True, "type": "watching",
             "state": "hermes", "details": "d" * 200},
            {},
        )
        call_kwargs = discord.Activity.call_args.kwargs
        assert len(call_kwargs["details"]) == 128

    def test_missing_model_empty_state_cleared_not_sent(self):
        """Regression: state "{{model}}" with no model.default renders empty;
        Discord rejects empty activity names, so the adapter must skip the
        API call (and, if a prior activity exists, clear it) instead of
        retrying every watchdog cycle with a swallowed 400."""
        client = self._do_apply(
            {"enabled": True, "type": "watching", "state": "{{model}}"},
            {},
        )
        assert client.change_presence.await_count == 0
        assert discord.Activity.call_count == 0

    def test_empty_state_clears_prior_activity(self):
        async def _test():
            client = _client_with_async_presence()
            adapter = _make_adapter(client)
            adapter._last_activity_state = ("watching", "old-model", "")
            with patch(f"{_CONFIG_MODULE}.read_raw_config",
                       return_value=_mock_discord_cfg({
                           "enabled": True, "type": "watching",
                           "state": "{{model}}"
                       })):
                with patch(f"{_CONFIG_MODULE}.load_config_readonly", return_value={}):
                    with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                               return_value="default"):
                        await adapter._apply_activity()
            client.change_presence.assert_awaited_with(activity=None)
            # Second pass: diff guard sees the same empty state, no API call.
            with patch(f"{_CONFIG_MODULE}.read_raw_config",
                       return_value=_mock_discord_cfg({
                           "enabled": True, "type": "watching",
                           "state": "{{model}}"
                       })):
                with patch(f"{_CONFIG_MODULE}.load_config_readonly", return_value={}):
                    with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                               return_value="default"):
                        await adapter._apply_activity()
            assert client.change_presence.await_count == 1
        asyncio.run(_test())

    def test_unknown_type_does_not_crash(self):
        client = self._do_apply(
            {"enabled": True, "type": "streaming", "state": "test"},
            {},
        )
        assert client.change_presence.await_count == 0
        assert discord.Activity.call_count == 0

    def test_client_none_does_not_crash(self):
        adapter = _make_adapter(client=None)
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "type": "watching", "state": "test"
                   })):
            with patch(f"{_CONFIG_MODULE}.load_config_readonly", return_value={}):
                with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                           return_value="default"):
                    asyncio.run(adapter._apply_activity())

    def test_change_presence_exception_does_not_crash(self):
        client = _client_with_async_presence()
        client.change_presence.side_effect = RuntimeError("Discord API error")
        adapter = _make_adapter(client)
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "type": "watching", "state": "test"
                   })):
            with patch(f"{_CONFIG_MODULE}.load_config_readonly", return_value={}):
                with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                           return_value="default"):
                    asyncio.run(adapter._apply_activity())
        # Failure must not poison the cache — next cycle should retry.
        assert adapter._last_activity_state is None

    @pytest.mark.parametrize("atype", ["playing", "watching", "listening", "competing"])
    def test_all_activity_types_map_correctly(self, atype):
        client = self._do_apply(
            {"enabled": True, "type": atype, "state": "hermes"}, {},
        )
        self._assert_activity_created(atype, "hermes")

    def test_load_config_readonly_exception_handled(self):
        client = _client_with_async_presence()
        adapter = _make_adapter(client)
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({
                       "enabled": True, "type": "watching",
                       "state": "{{model}} on {{profile}}"
                   })):
            with patch(f"{_CONFIG_MODULE}.load_config_readonly",
                       side_effect=RuntimeError("config unreadable")):
                with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                           return_value="coder"):
                    asyncio.run(adapter._apply_activity())
        self._assert_activity_created("watching", " on coder")


# Watchdog diff guard tests (re-entry through _apply_activity)


class TestWatchdogDiffGuard:

    def setup_method(self):
        discord.Activity.reset_mock()

    def test_updates_when_state_changes(self):
        async def _test():
            adapter = _make_adapter(_client_with_async_presence())
            with patch(f"{_CONFIG_MODULE}.read_raw_config",
                       return_value=_mock_discord_cfg({
                           "enabled": True, "type": "watching", "state": "{{model}}"
                       })):
                with patch(f"{_CONFIG_MODULE}.load_config_readonly",
                           return_value={"model": {"default": "gpt-4o"}}):
                    with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                               return_value="coder"):
                        await adapter._apply_activity()
            assert adapter._last_activity_state == ("watching", "gpt-4o", "")
            assert discord.Activity.call_count == 1
        asyncio.run(_test())

    def test_skips_when_unchanged(self):
        async def _test():
            adapter = _make_adapter(_client_with_async_presence())
            adapter._last_activity_state = ("watching", "gpt-4o", "")
            with patch(f"{_CONFIG_MODULE}.read_raw_config",
                       return_value=_mock_discord_cfg({
                           "enabled": True, "type": "watching", "state": "{{model}}"
                       })):
                with patch(f"{_CONFIG_MODULE}.load_config_readonly",
                           return_value={"model": {"default": "gpt-4o"}}):
                    with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                               return_value="coder"):
                        await adapter._apply_activity()
            assert discord.Activity.call_count == 0
            assert adapter._client.change_presence.await_count == 0
        asyncio.run(_test())

    def test_updates_when_type_changes(self):
        async def _test():
            adapter = _make_adapter(_client_with_async_presence())
            adapter._last_activity_state = ("watching", "gpt-4o", "")
            with patch(f"{_CONFIG_MODULE}.read_raw_config",
                       return_value=_mock_discord_cfg({
                           "enabled": True, "type": "playing", "state": "{{model}}"
                       })):
                with patch(f"{_CONFIG_MODULE}.load_config_readonly",
                           return_value={"model": {"default": "gpt-4o"}}):
                    with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                               return_value="coder"):
                        await adapter._apply_activity()
            assert discord.Activity.call_count == 1
        asyncio.run(_test())

    def test_disabled_clears_activity(self):
        async def _test():
            adapter = _make_adapter(_client_with_async_presence())
            adapter._last_activity_state = ("watching", "old-model", "")
            with patch(f"{_CONFIG_MODULE}.read_raw_config",
                       return_value=_mock_discord_cfg({"enabled": False})):
                await adapter._apply_activity()
            adapter._client.change_presence.assert_awaited_with(activity=None)
            assert adapter._last_activity_state is None
        asyncio.run(_test())

    def test_unknown_type_logs_warning_and_skips(self):
        async def _test():
            adapter = _make_adapter(_client_with_async_presence())
            with patch(f"{_CONFIG_MODULE}.read_raw_config",
                       return_value=_mock_discord_cfg({
                           "enabled": True, "type": "streaming", "state": "test"
                       })):
                with patch(f"{_CONFIG_MODULE}.load_config_readonly", return_value={}):
                    with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                               return_value="default"):
                        await adapter._apply_activity()
            assert discord.Activity.call_count == 0
            assert adapter._client.change_presence.await_count == 0
        asyncio.run(_test())


# _refresh_discord_activity tests


class TestActivityWatchdogInterval:

    def test_default_interval_is_60(self):
        adapter = _make_adapter()
        assert adapter._activity_watchdog_interval == 60

    def test_interval_configurable(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value={"discord": {"activity_check_interval_seconds": 300}}):
            assert adapter._load_discord_int_config(
                "activity_check_interval_seconds", 60, minimum=5
            ) == 300

    def test_interval_clamped_to_minimum(self):
        adapter = _make_adapter()
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value={"discord": {"activity_check_interval_seconds": 1}}):
            assert adapter._load_discord_int_config(
                "activity_check_interval_seconds", 60, minimum=5
            ) == 5


class TestRefreshDiscordActivity:

    def test_noop_when_no_adapters(self):
        from gateway.run import GatewaySlashCommandsMixin

        class DummyRunner(GatewaySlashCommandsMixin):
            pass

        runner = DummyRunner()
        runner.adapters = {}
        runner._refresh_discord_activity()

    def test_noop_when_no_discord_adapter(self):
        from gateway.run import GatewaySlashCommandsMixin
        from gateway.config import Platform

        class DummyRunner(GatewaySlashCommandsMixin):
            pass

        runner = DummyRunner()
        runner.adapters = {Platform.TELEGRAM: MagicMock()}
        runner._refresh_discord_activity()

    def test_calls_adapter_method(self):
        from gateway.run import GatewaySlashCommandsMixin
        from gateway.config import Platform

        class DummyRunner(GatewaySlashCommandsMixin):
            pass

        adapter = MagicMock()
        adapter._apply_activity = AsyncMock()

        runner = DummyRunner()
        runner.adapters = {Platform.DISCORD: adapter}

        async def _run():
            runner._refresh_discord_activity()
            await asyncio.sleep(0)

        asyncio.run(_run())
        adapter._apply_activity.assert_awaited_once()


# _clear_activity tests


class TestClearActivity:

    def test_none_client_noop(self):
        adapter = _make_adapter(client=None)
        asyncio.run(adapter._clear_activity())

    def test_clears_and_resets_cache(self):
        client = _client_with_async_presence()
        adapter = _make_adapter(client)
        adapter._last_activity_state = ("watching", "some-status", "")
        asyncio.run(adapter._clear_activity())
        client.change_presence.assert_awaited_with(activity=None)
        assert adapter._last_activity_state is None


# _cancel_activity_watchdog_task tests


class TestCancelWatchdogTask:

    def test_no_task_to_cancel(self):
        async def _test():
            adapter = _make_adapter()
            await adapter._cancel_activity_watchdog_task()
        asyncio.run(_test())

    def test_cancels_running_task(self):
        async def _test():
            adapter = _make_adapter()
            adapter._running = True

            async def _dummy_loop():
                while True:
                    await asyncio.sleep(100)

            adapter._activity_watchdog_task = asyncio.create_task(_dummy_loop())
            await asyncio.sleep(0)
            await adapter._cancel_activity_watchdog_task()

            assert adapter._activity_watchdog_task is None
            assert adapter._last_activity_state is None
        asyncio.run(_test())


# Integration tests


class TestOnReadyIntegration:

    def test_on_ready_apply_activity_sends_presence(self):
        """on_ready awaits _apply_activity; simulate that path end to end."""
        client = _client_with_async_presence()
        adapter = _make_adapter(client)

        async def _test():
            with patch(f"{_CONFIG_MODULE}.read_raw_config",
                       return_value=_mock_discord_cfg({
                           "enabled": True, "type": "watching", "state": "test"
                       })):
                with patch(f"{_CONFIG_MODULE}.load_config_readonly", return_value={}):
                    with patch(f"{_PROFILES_MODULE}.get_active_profile_name",
                               return_value="default"):
                        await adapter._apply_activity()
        asyncio.run(_test())
        assert client.change_presence.await_count == 1

    def test_on_ready_no_crash_when_disabled(self):
        adapter = _make_adapter(_client_with_async_presence())
        with patch(f"{_CONFIG_MODULE}.read_raw_config",
                   return_value=_mock_discord_cfg({"enabled": False})):
            asyncio.run(adapter._apply_activity())


# DEFAULT_CONFIG sanity


class TestDefaultConfigActivity:

    def test_default_config_has_activity(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        assert "discord" in DEFAULT_CONFIG
        assert "activity" in DEFAULT_CONFIG["discord"]
        assert not DEFAULT_CONFIG["discord"]["activity"].get("enabled", False)

    def test_default_config_has_details_field(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        assert "details" in DEFAULT_CONFIG["discord"]["activity"]
        assert DEFAULT_CONFIG["discord"]["activity"]["details"] == ""

    def test_default_config_keys(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        keys = set(DEFAULT_CONFIG["discord"]["activity"].keys())
        assert keys == {"enabled", "type", "state", "details"}
