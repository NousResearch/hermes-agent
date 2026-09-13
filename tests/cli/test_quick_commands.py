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
    async def test_timeout_kills_the_command_process_tree(self, tmp_path):
        """A timed-out exec quick command must not keep running under the gateway, nor its children."""
        import asyncio
        import contextlib
        import sys
        import time
        import psutil
        from gateway.run import GatewayRunner

        pid_file = tmp_path / "pids"
        script = tmp_path / "tree.py"
        script.write_text(
            "import os, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"with open({str(pid_file)!r}, 'w', encoding='utf-8') as f:\n"
            "    f.write(f'{os.getpid()}\\n{child.pid}\\n')\n"
            "time.sleep(120)\n",
            encoding="utf-8",
        )

        def _pids():
            try:
                return [int(p) for p in pid_file.read_text(encoding="utf-8").split()]
            except (OSError, ValueError):
                return []

        real_wait_for = asyncio.wait_for

        async def _expire_once_tree_is_up(aw, timeout):
            # Behave like wait_for() hitting its 30 s cap, without waiting 30 s: cancel the inner
            # communicate() and raise TimeoutError once the command's process tree exists.
            if getattr(aw, "__qualname__", "") != "Process.communicate":
                return await real_wait_for(aw, timeout)
            task = asyncio.ensure_future(aw)
            deadline = time.monotonic() + 15
            while len(_pids()) < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise asyncio.TimeoutError

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"hang": {"type": "exec", "command": f'"{sys.executable}" "{script}"'}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        with patch("asyncio.wait_for", _expire_once_tree_is_up):
            result = await runner._handle_message(self._make_event("hang"))

        assert "timed out" in result.lower()
        pids = _pids()
        assert len(pids) == 2, "the quick command never started its process tree"

        def _running(pid):
            # Read-only probe: waiting on the direct child here would steal asyncio's reap.
            try:
                return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False

        alive = pids
        deadline = time.monotonic() + 5
        while alive and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            alive = [pid for pid in alive if _running(pid)]
        for pid in alive:  # never leak the sleepers into the rest of the run, even on failure
            with contextlib.suppress(psutil.NoSuchProcess):
                psutil.Process(pid).kill()
        assert not alive, f"timed-out quick command left processes running: {alive}"

    @pytest.mark.asyncio
    async def test_completed_command_is_not_killed(self):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"limits": {"type": "exec", "command": "echo ok"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        with patch("agent.deadline.kill_process_tree") as kill_tree:
            result = await runner._handle_message(self._make_event("limits"))
        assert result == "ok"
        kill_tree.assert_not_called()

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
