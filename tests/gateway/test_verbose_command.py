"""Tests for gateway /verbose command (config-gated tool progress cycling)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/verbose", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner without calling __init__."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


class TestVerboseCommand:
    """Tests for _handle_verbose_command in the gateway."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self, tmp_path, monkeypatch):
        """When tool_progress_command is false, /verbose returns an info message."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("display:\n  tool_progress: all\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        result = await runner._handle_verbose_command(_make_event())

        assert "not enabled" in result.lower()
        assert "tool_progress_command" in result

    @pytest.mark.asyncio
    async def test_enabled_cycles_mode(self, tmp_path, monkeypatch):
        """When enabled, /verbose cycles tool_progress mode per-platform."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "display:\n  tool_progress_command: true\n  tool_progress: all\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        result = await runner._handle_verbose_command(_make_event())

        # all -> verbose
        assert "VERBOSE" in result
        assert "telegram" in result.lower()  # per-platform feedback

        # Verify config was saved to display.platforms.telegram
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["display"]["platforms"]["telegram"]["tool_progress"] == "verbose"

    @pytest.mark.asyncio
    async def test_cycles_through_all_modes(self, tmp_path, monkeypatch):
        """Calling /verbose repeatedly cycles through all four modes."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "display:\n  tool_progress_command: true\n  tool_progress: 'off'\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        runner = _make_runner()

        # off -> new -> all -> verbose -> off
        expected = ["new", "all", "verbose", "off"]
        for mode in expected:
            result = await runner._handle_verbose_command(_make_event())
            saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            actual = saved["display"]["platforms"]["telegram"]["tool_progress"]
            assert actual == mode, \
                f"Expected {mode}, got {actual}"

    @pytest.mark.asyncio
    async def test_defaults_to_all_when_no_tool_progress_set(self, tmp_path, monkeypatch):
        """When tool_progress is not in config, defaults to 'all' then cycles to verbose."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "display:\n  tool_progress_command: true\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        result = await runner._handle_verbose_command(_make_event())

        # Telegram default is "all" (high tier) → cycles to verbose
        assert "VERBOSE" in result
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["display"]["platforms"]["telegram"]["tool_progress"] == "verbose"

    @pytest.mark.asyncio
    async def test_per_platform_isolation(self, tmp_path, monkeypatch):
        """Cycling /verbose on Telegram doesn't change Slack's setting.

        Without a global tool_progress, each platform uses its built-in
        default: Telegram = 'all' (high tier), Slack = 'off' (quiet Slack default).
        """
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        # No global tool_progress → built-in platform defaults apply
        config_path.write_text(
            "display:\n  tool_progress_command: true\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        runner = _make_runner()

        # Cycle on Telegram
        await runner._handle_verbose_command(
            _make_event(platform=Platform.TELEGRAM)
        )
        # Cycle on Slack
        await runner._handle_verbose_command(
            _make_event(platform=Platform.SLACK)
        )

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        platforms = saved["display"]["platforms"]
        # Telegram: all -> verbose (high tier default = all)
        assert platforms["telegram"]["tool_progress"] == "verbose"
        # Slack: off -> new (first /verbose cycle from quiet default)
        assert platforms["slack"]["tool_progress"] == "new"

    @pytest.mark.asyncio
    async def test_no_config_file_returns_disabled(self, tmp_path, monkeypatch):
        """When config.yaml doesn't exist, command reports disabled."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        # No config.yaml

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        result = await runner._handle_verbose_command(_make_event())
        assert "not enabled" in result.lower()

    def test_verbose_is_in_gateway_known_commands(self):
        """The /verbose command is recognized by the gateway dispatch."""
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "verbose" in GATEWAY_KNOWN_COMMANDS
