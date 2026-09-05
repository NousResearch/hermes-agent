"""Tests for user-defined quick commands that bypass the agent loop."""
import os
import subprocess
from unittest.mock import MagicMock, patch
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


# ── Exec arg-forwarding (PR #9942) ──────────────────────────────────────
# Gateway/CLI/TUI exec quick commands must forward user args with argv
# boundaries preserved (split-then-quote, never blob-quote).

class TestGatewayExecArgsForwarding(TestGatewayQuickCommands):
    """Gateway exec quick commands append args with boundaries preserved."""

    @pytest.mark.asyncio
    async def test_exec_command_passes_args_tokenized(self):
        """Gateway exec quick commands append args with boundaries preserved."""
        import shlex
        from unittest.mock import AsyncMock

        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"search": {"type": "exec", "command": "echo"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        captured: dict[str, str] = {}
        fake_proc = AsyncMock()
        fake_proc.communicate.return_value = (b"ok", b"")

        async def fake_shell(cmd, **kwargs):
            captured["cmd"] = cmd
            return fake_proc

        with patch("asyncio.create_subprocess_shell", side_effect=fake_shell):
            result = await runner._handle_message(self._make_event("search", args="--foo bar"))

        assert result == "ok"
        assert shlex.split(captured["cmd"]) == ["echo", "--foo", "bar"]

    @pytest.mark.asyncio
    async def test_exec_command_metachars_stay_boundary_safe(self):
        """Gateway: shell metacharacters must stay inside quoted tokens."""
        import shlex
        from unittest.mock import AsyncMock

        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"search": {"type": "exec", "command": "echo"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        captured: dict[str, str] = {}
        fake_proc = AsyncMock()
        fake_proc.communicate.return_value = (b"ok", b"")

        async def fake_shell(cmd, **kwargs):
            captured["cmd"] = cmd
            return fake_proc

        for dangerous in ["'; rm -rf /'", "a && id", "x | cat", "%PATH%"]:
            with patch("asyncio.create_subprocess_shell", side_effect=fake_shell):
                await runner._handle_message(self._make_event("search", args=dangerous))
            assert shlex.split(captured["cmd"]) == ["echo"] + shlex.split(dangerous)





# ── TUI tests ──────────────────────────────────────────────────────────────

class TestTUIQuickCommands:
    """Test TUI command.dispatch exec quick commands (arg forwarding)."""

    @staticmethod
    def _dispatch(name, arg):
        # Handlers are registered on tui_gateway.server's _methods registry
        # (method_ctx rebinds at install); methods_tools itself has no dict.
        import tui_gateway.server as server

        return server._methods["command.dispatch"](None, {"name": name, "arg": arg, "session_id": ""})

    @staticmethod
    def _patch_cfg(quick_commands):
        # _load_cfg is lazily imported from tui_gateway.server at call time —
        # patch the source module, not the consumer.
        return patch("tui_gateway.server._load_cfg", return_value={"quick_commands": quick_commands})

    def test_tui_exec_forwards_args_tokenized(self):
        import shlex
        import subprocess as sp
        from unittest.mock import patch

        captured: dict[str, str] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return sp.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with self._patch_cfg({"search": {"type": "exec", "command": "echo"}}):
            with patch("subprocess.run", side_effect=fake_run):
                self._dispatch("/search", "--foo bar")

        assert shlex.split(captured["cmd"]) == ["echo", "--foo", "bar"]

    def test_tui_exec_no_args_unchanged(self):
        import subprocess as sp
        from unittest.mock import patch

        captured: dict[str, str] = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return sp.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with self._patch_cfg({"limits": {"type": "exec", "command": "echo ok"}}):
            with patch("subprocess.run", side_effect=fake_run):
                self._dispatch("/limits", "")

        assert captured["cmd"] == "echo ok"


class TestExecQuickCommandArgs:
    """Test that exec quick commands receive user arguments."""

    def test_exec_command_receives_args(self):
        """Exec quick commands should pass user arguments to the shell command."""
        from unittest.mock import patch
        import subprocess

        cli = TestCLIQuickCommands()._make_cli({
            "search": {"type": "exec", "command": "echo"}
        })

        # Process command with arguments
        result = cli.process_command("/search hello world")
        assert result is True

        # Verify the command was called with arguments
        printed = TestCLIQuickCommands._printed_plain(cli.console.print.call_args[0][0])
        assert "hello world" in printed

    def test_exec_command_passes_args_tokenized(self):
        """User args must be appended per-token so argv boundaries survive."""
        import shlex

        cli = TestCLIQuickCommands()._make_cli({"search": {"type": "exec", "command": "echo"}})
        captured: dict[str, str] = {}

        def fake_run(args, **kwargs):
            captured["cmd"] = args
            return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            result = cli.process_command("/search --foo bar")

        assert result is True
        assert shlex.split(captured["cmd"]) == ["echo", "--foo", "bar"]

    def test_exec_command_metachars_stay_boundary_safe(self):
        """Shell metacharacters in user args must arrive quoted, not as operators."""
        import shlex

        cli = TestCLIQuickCommands()._make_cli({"search": {"type": "exec", "command": "echo"}})
        captured: dict[str, str] = {}

        def fake_run(args, **kwargs):
            captured["cmd"] = args
            return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

        for dangerous in ["'; rm -rf /'", "a && id", "x | cat", "%PATH%", "a; touch /tmp/pwned"]:
            with patch("subprocess.run", side_effect=fake_run):
                cli.process_command(f"/search {dangerous}")
            assert shlex.split(captured["cmd"]) == ["echo"] + shlex.split(dangerous)

    def test_exec_command_no_args_still_works(self):
        """Exec quick commands without arguments should still execute."""
        cli = TestCLIQuickCommands()._make_cli({
            "test": {"type": "exec", "command": "echo ok"}
        })

        result = cli.process_command("/test")
        assert result is True
        printed = TestCLIQuickCommands._printed_plain(cli.console.print.call_args[0][0])
        assert "ok" in printed
