"""Tests for user-defined quick commands that bypass the agent loop."""
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch
from rich.text import Text
import pytest


# ── CLI tests ──────────────────────────────────────────────────────────────

class TestCLIQuickCommands:
    """Test quick command dispatch in HermesCLI.process_command."""

    @staticmethod
    def _printed_plain(call_arg):
        if isinstance(call_arg, Text):
            return call_arg.plain
        return str(call_arg)

    def _make_cli(self, quick_commands):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.config = {"quick_commands": quick_commands}
        cli.console = MagicMock()
        cli.agent = None
        cli.conversation_history = []
        # session_id is accessed by the fallback skill/fuzzy-match path in
        # process_command; without it, tests that exercise `/alias args`
        # can trip an AttributeError when cross-test state leaks a skill
        # command matching the alias target.
        cli.session_id = "test-session"
        return cli

    def test_exec_command_runs_and_prints_output(self):
        cli = self._make_cli({"dn": {"type": "exec", "command": "echo daily-note"}})
        result = cli.process_command("/dn")
        assert result is True
        cli.console.print.assert_called_once()
        printed = self._printed_plain(cli.console.print.call_args[0][0])
        assert printed == "daily-note"

    def test_exec_command_uses_chat_console_when_tui_is_live(self):
        cli = self._make_cli({"dn": {"type": "exec", "command": "echo daily-note"}})
        cli._app = object()
        live_console = MagicMock()

        with patch("cli.ChatConsole", return_value=live_console):
            result = cli.process_command("/dn")

        assert result is True
        live_console.print.assert_called_once()
        printed = self._printed_plain(live_console.print.call_args[0][0])
        assert printed == "daily-note"
        cli.console.print.assert_not_called()








    def test_prompt_command_seeds_rendered_prompt_as_the_literal_turn(self):
        """type: prompt sends the configured text (with {args}) as the next turn via the one-shot seed
        (as /prompt), never through ``_pending_input``, which re-parses it: a prompt starting with
        ``/help`` would run /help instead of reaching the model."""
        import queue
        cli = self._make_cli({"rn": {"type": "prompt", "prompt": "Draft release notes for {args}."},
                              "lit": {"type": "prompt", "prompt": "/help is what users type first"}})
        cli._pending_input = queue.Queue()
        cli._pending_agent_seed = None
        assert cli.process_command("/rn v2.3") is True
        assert cli._pending_agent_seed == "Draft release notes for v2.3."
        assert cli._pending_input.empty()
        cli._pending_agent_seed = None
        assert cli.process_command("/lit") is True
        assert cli._pending_agent_seed == "/help is what users type first" and cli._pending_input.empty()
        cli.console.print.assert_not_called()

    def test_quick_command_takes_priority_over_skill_commands(self):
        """Quick commands must be checked before skill slash commands."""
        cli = self._make_cli({"mygif": {"type": "exec", "command": "echo overridden"}})
        with patch("cli._skill_commands", {"/mygif": {"name": "gif-search"}}):
            cli.process_command("/mygif")
        cli.console.print.assert_called_once()
        printed = self._printed_plain(cli.console.print.call_args[0][0])
        assert printed == "overridden"




# ── Gateway tests ──────────────────────────────────────────────────────────

class TestGatewayQuickCommands:
    """Test quick command dispatch in GatewayRunner._handle_message."""

    def _make_event(self, command, args=""):
        event = MagicMock()
        event.get_command.return_value = command
        event.get_command_args.return_value = args
        event.text = f"/{command} {args}".strip()
        event.source = MagicMock()
        event.source.user_id = "test_user"
        event.source.user_name = "Test User"
        event.source.platform.value = "telegram"
        event.source.chat_type = "dm"
        event.source.chat_id = "123"
        return event

    @pytest.mark.asyncio
    async def test_exec_command_returns_output(self):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"limits": {"type": "exec", "command": "echo ok"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("limits")
        result = await runner._handle_message(event)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_exec_command_does_not_leak_credentials(self):
        """Quick command exec must sanitize env — API keys must not appear in output."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"leak": {"type": "exec", "command": "env"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("leak")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-secret-12345"}):
            result = await runner._handle_message(event)

        assert "sk-or-secret-12345" not in result, \
            "Quick command leaked OPENROUTER_API_KEY — exec runs without env sanitization"

    @pytest.mark.asyncio
    async def test_exec_command_output_is_redacted(self, monkeypatch):
        """Quick command output must redact sensitive patterns before returning."""
        from gateway.run import GatewayRunner

        # Ensure redaction is active regardless of host HERMES_REDACT_SECRETS state
        # or test ordering
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"token": {"type": "exec", "command": "echo sk-ant-api03-supersecretkey1234567890"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("token")
        result = await runner._handle_message(event)

        assert "supersecretkey1234567890" not in result, \
            "Quick command output not redacted — raw API key returned to user"


    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        from gateway.run import GatewayRunner
        import asyncio
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"slow": {"type": "exec", "command": "sleep 100"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("slow")
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await runner._handle_message(event)
        assert result is not None
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_gateway_config_object_supports_quick_commands(self):
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            quick_commands={"limits": {"type": "exec", "command": "echo ok"}}
        )
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("limits")
        result = await runner._handle_message(event)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_prompt_command_in_a_busy_chat_follows_busy_input_mode(self, monkeypatch):
        """A saved prompt typed while the agent runs is expanded BEFORE busy routing: steer mode
        injects the rendered prompt (never the raw ``/rn v2.3``), and it is not also queued."""
        import gateway.run as _gr
        from gateway.run import GatewayRunner
        from gateway.platforms.base import build_session_key
        monkeypatch.setattr(_gr, "_load_gateway_config", dict)
        runner = object.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"rn": {"type": "prompt", "prompt": "Draft release notes for {args}."}}}
        runner._running_agents, runner._running_agents_ts, runner._pending_messages = {}, {}, {}
        runner._busy_ack_ts, runner._draining, runner._busy_input_mode = {}, False, "steer"
        runner._busy_text_mode, runner.adapters, runner.session_store = "interrupt", {}, None
        runner._is_user_authorized = lambda _source: True
        runner._check_slash_access = MagicMock(return_value=None)
        from gateway.config import Platform
        from gateway.platforms.event import MessageEvent, MessageType
        from gateway.session import SessionSource
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="private", user_id="u1")
        event = MessageEvent(text="/rn v2.3", message_type=MessageType.COMMAND, source=source, message_id="m1")
        adapter = MagicMock(_pending_messages={}, _text_debounce={}, _send_with_retry=AsyncMock())
        adapter.config.extra = {}
        runner.adapters[event.source.platform] = adapter
        sk = build_session_key(event.source)
        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent

        with patch("gateway.platforms.base.merge_pending_message_event") as merge:
            assert await runner._handle_active_session_busy_message(event, sk) is True

        agent.steer.assert_called_once()
        assert agent.steer.call_args.args[0].endswith("Draft release notes for v2.3.")
        agent.interrupt.assert_not_called()
        merge.assert_not_called()
        assert sk not in adapter._pending_messages

    @pytest.mark.asyncio
    async def test_prompt_command_rewrites_event_into_agent_turn(self):
        """Messaging platforms: the prompt replaces the slash text and dispatch falls through to the agent."""
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"rn": {"type": "prompt", "prompt": "Draft release notes."}}}
        runner._draining = False
        runner._check_slash_access = MagicMock(return_value=None)
        event = self._make_event("rn", "v2.3")

        handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(event, event.source, "rn")

        assert (handled, result, command) == (False, None, None)
        assert event.text == "Draft release notes.\n\nv2.3"
