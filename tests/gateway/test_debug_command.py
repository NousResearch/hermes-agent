"""Tests for gateway /debug runtime override behavior."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.discord import DiscordAdapter  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402


def _make_event(text="/debug") -> MessageEvent:
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="alan",
        chat_type="group",
    )
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner(discord_extra: dict | None = None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="***",
                extra=discord_extra or {},
            )
        }
    )
    runner.adapters = {
        Platform.DISCORD: DiscordAdapter(runner.config.platforms[Platform.DISCORD])
    }
    runner._runtime_debug_overrides = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._session_send_policies = {}
    runner._session_docks = {}
    runner._session_db = None
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    return runner


@pytest.mark.asyncio
async def test_debug_show_lists_supported_keys_when_no_overrides():
    runner = _make_runner()

    result = await runner._handle_debug_command(_make_event("/debug"))

    assert "Runtime Debug Overrides" in result
    assert "None" in result
    assert "`agent.system_prompt`" in result
    assert "`discord.auto_thread`" in result


@pytest.mark.asyncio
async def test_debug_set_show_reasoning_updates_runtime_cache(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setattr("gateway.run._hermes_home", hermes_home)
    runner = _make_runner()

    result = await runner._handle_debug_command(
        _make_event("/debug set display.show_reasoning true")
    )

    assert runner._show_reasoning is True
    assert "`true`" not in result  # format_yaml_block uses yaml-style, not inline code
    assert "Runtime override set for `display.show_reasoning`" in result
    assert "true" in result.lower()


@pytest.mark.asyncio
async def test_debug_set_and_unset_reasoning_effort_restores_base_config(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "agent:\n  reasoning_effort: low\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.run._hermes_home", hermes_home)
    runner = _make_runner()

    await runner._handle_debug_command(_make_event("/debug set agent.reasoning_effort xhigh"))
    unset_result = await runner._handle_debug_command(_make_event("/debug unset agent.reasoning_effort"))

    assert runner._reasoning_config == {"enabled": True, "effort": "low"}
    assert "Removed runtime override for `agent.reasoning_effort`" in unset_result
    assert "low" in unset_result


@pytest.mark.asyncio
async def test_debug_set_discord_policy_override_updates_live_adapter():
    runner = _make_runner({"auto_thread": True, "require_mention": True})
    adapter = runner.adapters[Platform.DISCORD]

    result = await runner._handle_debug_command(
        _make_event("/debug set discord.auto_thread false")
    )

    assert adapter._get_discord_policy().auto_thread is False
    assert adapter._get_discord_policy().require_mention is True
    assert runner._runtime_debug_overrides["discord.auto_thread"] is False
    assert "Runtime override set for `discord.auto_thread`" in result


@pytest.mark.asyncio
async def test_debug_reset_clears_active_overrides(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setattr("gateway.run._hermes_home", hermes_home)
    runner = _make_runner()

    await runner._handle_debug_command(_make_event("/debug set display.show_reasoning true"))
    await runner._handle_debug_command(_make_event("/debug set discord.require_mention false"))
    result = await runner._handle_debug_command(_make_event("/debug reset"))

    assert runner._runtime_debug_overrides == {}
    assert runner._show_reasoning is False
    assert runner.adapters[Platform.DISCORD]._get_discord_policy().require_mention is True
    assert "Removed `2` override(s)" in result
