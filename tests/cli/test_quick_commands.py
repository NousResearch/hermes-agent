"""Tests for user-defined quick commands that bypass the agent loop."""
import os
import subprocess
from unittest.mock import MagicMock, patch
from rich.text import Text
import pytest


# ── CLI tests ──────────────────────────────────────────────────────────────

class TestCLIQuickCommands:
    """Test custom command dispatch via commands.custom (legacy quick_commands path removed)."""

    @staticmethod
    def _printed_plain(call_arg):
        if isinstance(call_arg, Text):
            return call_arg.plain
        return str(call_arg)

    def _make_cli(self, commands):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.config = {"commands": {"custom": commands}}
        cli.console = MagicMock()
        cli.agent = None
        cli.conversation_history = []
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

    def test_exec_command_stderr_shown_on_no_stdout(self):
        cli = self._make_cli({"err": {"type": "exec", "command": "echo error >&2"}})
        result = cli.process_command("/err")
        assert result is True
        # stderr fallback — should print something
        cli.console.print.assert_called_once()

    def test_exec_command_no_output_shows_fallback(self):
        cli = self._make_cli({"empty": {"type": "exec", "command": "true"}})
        cli.process_command("/empty")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "no output" in args.lower()

    def test_alias_command_routes_to_target(self):
        """Alias custom commands rewrite to the target command."""
        cli = self._make_cli({"shortcut": {"type": "alias", "command": "/help"}})
        with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
            cli.process_command("/shortcut")
            # Should recursively call process_command with /help
            spy.assert_any_call("/help")

    def test_alias_command_passes_args(self):
        """Alias custom commands forward user arguments to the target."""
        cli = self._make_cli({"sc": {"type": "alias", "command": "/context"}})
        with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
            cli.process_command("/sc some args")
            spy.assert_any_call("/context some args")

    def test_alias_no_target_shows_error(self):
        cli = self._make_cli({"broken": {"type": "alias", "command": ""}})
        cli.process_command("/broken")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "no target defined" in args.lower()

    def test_unsupported_type_shows_error(self):
        cli = self._make_cli({"bad": {"type": "prompt", "command": "echo hi"}})
        cli.process_command("/bad")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "unsupported type" in args.lower()

    def test_missing_command_field_shows_error(self):
        cli = self._make_cli({"oops": {"type": "exec"}})
        cli.process_command("/oops")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "no command defined" in args.lower()

    def test_quick_command_takes_priority_over_skill_commands(self):
        """Quick commands must be checked before skill slash commands."""
        cli = self._make_cli({"mygif": {"type": "exec", "command": "echo overridden"}})
        with patch("cli._skill_commands", {"/mygif": {"name": "gif-search"}}):
            cli.process_command("/mygif")
        cli.console.print.assert_called_once()
        printed = self._printed_plain(cli.console.print.call_args[0][0])
        assert printed == "overridden"

    def test_unknown_command_still_shows_error(self):
        cli = self._make_cli({})
        with patch("cli._cprint") as mock_cprint:
            cli.process_command("/nonexistent")
            mock_cprint.assert_called()
            printed = " ".join(str(c) for c in mock_cprint.call_args_list)
            assert "unknown command" in printed.lower()

    def test_timeout_shows_error(self):
        cli = self._make_cli({"slow": {"type": "exec", "command": "sleep 100"}})
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sleep", 30)):
            cli.process_command("/slow")
        cli.console.print.assert_called_once()
        args = cli.console.print.call_args[0][0]
        assert "timed out" in args.lower()


class TestCLICustomCommands:
    """Test custom command dispatch from the new commands.custom structure."""

    @staticmethod
    def _printed_plain(call_arg):
        if isinstance(call_arg, Text):
            return call_arg.plain
        return str(call_arg)

    def _make_cli(self, custom_cmds):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.config = {"commands": {"custom": custom_cmds}}
        cli.console = MagicMock()
        cli.agent = None
        cli.conversation_history = []
        cli.session_id = "test-session"
        return cli

    def test_custom_commands_new_structure_dispatch(self):
        """A command defined in commands.custom should be dispatched."""
        cli = self._make_cli({"dn": {"type": "exec", "command": "echo daily-note"}})
        result = cli.process_command("/dn")
        assert result is True
        cli.console.print.assert_called_once()
        printed = self._printed_plain(cli.console.print.call_args[0][0])
        assert printed == "daily-note"

    def test_custom_commands_override_quick_commands(self):
        """commands.custom is the only source of custom commands."""
        cli = self._make_cli(
            {"dn": {"type": "exec", "command": "echo new-format"}},
        )
        result = cli.process_command("/dn")
        assert result is True
        printed = self._printed_plain(cli.console.print.call_args[0][0])
        assert printed == "new-format", \
            "commands.custom must take priority over legacy quick_commands"

    def test_custom_commands_alias_reads_command(self):
        """Alias in commands.custom must read the 'command' field, not 'target'."""
        cli = self._make_cli({"go": {"type": "alias", "command": "/help"}})
        with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
            cli.process_command("/go")
            spy.assert_any_call("/help")

    def test_custom_commands_alias_rejects_target_field(self):
        """Alias in commands.custom must NOT read the legacy 'target' field."""
        cli = self._make_cli({"go": {"type": "alias", "target": "/help", "command": "/context"}})
        with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
            cli.process_command("/go")
            spy.assert_any_call("/context")

    def test_custom_commands_disabled_shows_error(self):
        """A disabled commands.custom command should show an error."""
        cli = self._make_cli({"off": {"type": "exec", "command": "echo hi", "enabled": False}})
        cli.process_command("/off")
        cli.console.print.assert_called_once()
        printed = self._printed_plain(cli.console.print.call_args[0][0])
        assert "disabled" in printed.lower()

    def test_custom_commands_cli_visibility_gate_blocks(self):
        """visible.cli=False must block the command on the CLI."""
        cli = self._make_cli({"hide": {"type": "exec", "command": "echo secret", "visible": {"cli": False}}})
        cli.process_command("/hide")
        cli.console.print.assert_called_once()
        printed = self._printed_plain(cli.console.print.call_args[0][0])
        assert "not available" in printed.lower()

    def test_custom_commands_cli_visibility_gate_allows_default(self):
        """Without visible.cli the command must work on the CLI."""
        cli = self._make_cli({"ok": {"type": "exec", "command": "echo visible"}})
        result = cli.process_command("/ok")
        assert result is True


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
        runner.config = {
            "commands": {"custom": {"limits": {"type": "exec", "command": "echo ok"}}}
        }
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
        runner.config = {
            "commands": {"custom": {"leak": {"type": "exec", "command": "env"}}}
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("leak")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "«redacted:sk-…»"}):
            result = await runner._handle_message(event)

        assert "«redacted:sk-…»" not in result, \
            "Quick command leaked OPENROUTER_API_KEY — exec runs without env sanitization"

    @pytest.mark.asyncio
    async def test_exec_command_output_is_redacted(self, monkeypatch):
        """Quick command output must redact sensitive patterns before returning."""
        from gateway.run import GatewayRunner

        # Ensure redaction is active regardless of host HERMES_REDACT_SECRETS state
        # or test ordering
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {"custom": {"token": {"type": "exec", "command": "echo «redacted:sk-…»"}}}
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("token")
        result = await runner._handle_message(event)

        assert "supersecretkey1234567890" not in result, \
            "Quick command output not redacted — raw API key returned to user"

    @pytest.mark.asyncio
    async def test_unsupported_type_returns_error(self):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {"custom": {"bad": {"type": "prompt", "command": "echo hi"}}}
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("bad")
        result = await runner._handle_message(event)
        assert result is not None
        assert "unsupported type" in result.lower()

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        from gateway.run import GatewayRunner
        import asyncio
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {"custom": {"slow": {"type": "exec", "command": "sleep 100"}}}
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("slow")
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await runner._handle_message(event)
        assert result is not None
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_gateway_config_object_supports_custom_commands(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {"custom": {"limits": {"type": "exec", "command": "echo ok"}}}
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("limits")
        result = await runner._handle_message(event)
        assert result == "ok"


class TestGatewayCustomCommands:
    """Test custom command dispatch from the new commands.custom structure."""

    def _make_event(self, command, args="", platform="telegram"):
        event = MagicMock()
        event.get_command.return_value = command
        event.get_command_args.return_value = args
        event.text = f"/{command} {args}".strip()
        event.source = MagicMock()
        event.source.user_id = "test_user"
        event.source.user_name = "Test User"
        event.source.platform.value = platform
        event.source.chat_type = "dm"
        event.source.chat_id = "123"
        return event

    @pytest.mark.asyncio
    async def test_custom_commands_new_structure_dispatch(self):
        """A command defined in commands.custom should be dispatched by gateway."""
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {
                "custom": {"limits": {"type": "exec", "command": "echo ok"}}
            }
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("limits")
        result = await runner._handle_message(event)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_custom_commands_override_quick_commands(self):
        """commands.custom must win over quick_commands in gateway dispatch."""
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {"custom": {"test": {"type": "exec", "command": "echo new-format"}}},
            "quick_commands": {"test": {"type": "exec", "command": "echo old-format"}},
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("test")
        result = await runner._handle_message(event)
        assert result == "new-format", \
            "commands.custom must take priority over legacy quick_commands"

    @pytest.mark.asyncio
    async def test_custom_commands_alias_reads_command(self):
        """Alias in commands.custom must read 'command' field in gateway."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {
                "custom": {"go": {"type": "alias", "command": "/bogus"}}
            }
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=False)
        runner.session_store = MagicMock()
        runner.hooks = MagicMock()

        event = self._make_event("go")
        result = await runner._handle_message(event)
        # The alias rewrites /go to /bogus — verify the rewrite happened
        assert event.text == "/bogus"

    @pytest.mark.asyncio
    async def test_custom_commands_visibility_gate_blocks(self):
        """Platform visibility=False must block the command."""
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {
                "custom": {
                    "hide": {
                        "type": "exec",
                        "command": "echo secret",
                        "visible": {"telegram": False},
                    }
                }
            }
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("hide", platform="telegram")
        result = await runner._handle_message(event)
        assert "not available" in result.lower()
        assert "telegram" in result.lower()

    @pytest.mark.asyncio
    async def test_custom_commands_visibility_gate_allows_other_platforms(self):
        """Platform visibility=False must only block that platform."""
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {
                "custom": {
                    "show": {
                        "type": "exec",
                        "command": "echo visible",
                        "visible": {"telegram": False},
                    }
                }
            }
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        # Should work on discord despite being blocked on telegram
        event = self._make_event("show", platform="discord")
        result = await runner._handle_message(event)
        assert result == "visible"

    @pytest.mark.asyncio
    async def test_custom_commands_visibility_gate_defaults_true(self):
        """Without visible key, the command must work."""
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {
            "commands": {
                "custom": {
                    "ok": {"type": "exec", "command": "echo allowed"}
                }
            }
        }
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("ok", platform="telegram")
        result = await runner._handle_message(event)
        assert result == "allowed"
