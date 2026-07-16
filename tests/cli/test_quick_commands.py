"""Tests for user-defined quick commands that bypass the agent loop."""
import asyncio
import os
import subprocess
import sys
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
        """Alias quick commands rewrite to the target command."""
        cli = self._make_cli({"shortcut": {"type": "alias", "target": "/help"}})
        with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
            cli.process_command("/shortcut")
            # Should recursively call process_command with /help
            spy.assert_any_call("/help")

    def test_alias_command_passes_args(self):
        """Alias quick commands forward user arguments to the target."""
        cli = self._make_cli({"sc": {"type": "alias", "target": "/context"}})
        with patch.object(cli, "process_command", wraps=cli.process_command) as spy:
            cli.process_command("/sc some args")
            spy.assert_any_call("/context some args")

    def test_alias_no_target_shows_error(self):
        cli = self._make_cli({"broken": {"type": "alias", "target": ""}})
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

    def test_argv_text_is_one_item_without_shell_or_credentials(self):
        cli = self._make_cli({
            "remember": {
                "type": "argv",
                "command": ["/opt/companion/bin/remember-spool-append", "--type", "fact"],
                "argument_mode": "text",
            }
        })
        completed = subprocess.CompletedProcess([], 0, stdout="saved\n", stderr="")
        with patch.dict(os.environ, {"PATH": "/usr/bin", "BOT_TOKEN": "secret"}), \
             patch(
                 "hermes_cli.quick_commands.run_bounded_argv",
                 return_value=completed,
             ) as run:
            result = cli.process_command("/remember hello; echo not-a-shell")

        assert result is True
        assert run.call_args.args[0] == [
            "/opt/companion/bin/remember-spool-append",
            "--type",
            "fact",
            "hello; echo not-a-shell",
        ]
        assert run.call_args.kwargs["env"]["PATH"] == "/usr/bin"
        assert "BOT_TOKEN" not in run.call_args.kwargs["env"]
        assert self._printed_plain(cli.console.print.call_args[0][0]) == "saved"

    @pytest.mark.parametrize(
        "qcmd",
        [
            "not-a-mapping",
            {"type": "argv", "command": "echo unsafe"},
            {"type": "argv", "command": []},
            {"type": "argv", "command": ["/bin/echo", "  "]},
            {"type": "argv", "command": ["/bin/echo"], "argument_mode": "words"},
        ],
    )
    def test_argv_malformed_config_is_user_facing(self, qcmd):
        cli = self._make_cli({"broken": qcmd})
        with patch("hermes_cli.quick_commands.run_bounded_argv") as run:
            cli.process_command("/broken hello")

        run.assert_not_called()
        assert "quick command" in str(cli.console.print.call_args[0][0]).lower()

    def test_argv_text_rejects_blank_and_oversized_input(self):
        cli = self._make_cli({
            "remember": {
                "type": "argv",
                "command": ["/bin/echo"],
                "argument_mode": "text",
            }
        })
        with patch("subprocess.run") as run:
            cli.process_command("/remember   ")
            blank = str(cli.console.print.call_args[0][0]).lower()
            cli.console.reset_mock()
            cli.process_command("/remember " + ("é" * 4097))
            oversized = str(cli.console.print.call_args[0][0]).lower()

        run.assert_not_called()
        assert "requires text" in blank
        assert "8192 utf-8 bytes" in oversized

    def test_argv_output_is_bounded(self):
        cli = self._make_cli({
            "qstatus": {"type": "argv", "command": ["/bin/echo"]}
        })
        completed = subprocess.CompletedProcess([], 0, stdout="x" * 65537, stderr="")
        with patch(
            "hermes_cli.quick_commands.run_bounded_argv", return_value=completed
        ):
            cli.process_command("/qstatus")

        assert "65536 utf-8 bytes" in str(cli.console.print.call_args[0][0]).lower()

    def test_argv_nonzero_exit_is_not_reported_as_success(self):
        cli = self._make_cli({
            "remember": {"type": "argv", "command": ["/bin/false"]}
        })
        completed = subprocess.CompletedProcess([], 9, stdout="", stderr="spool failed")
        with patch(
            "hermes_cli.quick_commands.run_bounded_argv", return_value=completed
        ):
            cli.process_command("/remember")

        rendered = str(cli.console.print.call_args[0][0]).lower()
        assert "failed" in rendered
        assert "spool failed" in rendered

    def test_argv_output_redaction_is_forced(self, monkeypatch):
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        cli = self._make_cli({
            "qstatus": {"type": "argv", "command": ["/bin/echo"]}
        })
        secret = "sk-ant-api03-supersecretkey1234567890"
        completed = subprocess.CompletedProcess([], 0, stdout=secret, stderr="")
        with patch(
            "hermes_cli.quick_commands.run_bounded_argv", return_value=completed
        ):
            cli.process_command("/qstatus")

        rendered = self._printed_plain(cli.console.print.call_args[0][0])
        assert "supersecretkey1234567890" not in rendered


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
        event.source.thread_id = None
        event.message_id = "msg-42"
        event.platform_update_id = 77
        return event

    def _make_runner(self, qcmd):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"remember": qcmd}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)
        runner._draining = False
        return runner

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
    async def test_unsupported_type_returns_error(self):
        from gateway.run import GatewayRunner
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {"quick_commands": {"bad": {"type": "prompt", "command": "echo hi"}}}
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
        runner.config = {"quick_commands": {"slow": {"type": "exec", "command": "sleep 100"}}}
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._is_user_authorized = MagicMock(return_value=True)

        event = self._make_event("slow")

        async def timeout_and_close(awaitable, timeout):
            del timeout
            awaitable.close()
            raise asyncio.TimeoutError

        with patch("asyncio.wait_for", new=timeout_and_close):
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
    async def test_argv_passes_one_text_item_and_exact_provenance_env(self):
        runner = self._make_runner({
            "type": "argv",
            "command": ["/opt/companion/bin/remember-spool-append", "--type", "fact"],
            "argument_mode": "text",
            "destination_alias": "owner",
        })
        event = self._make_event("remember", "hello; echo not-a-shell")
        proc = MagicMock(returncode=0)

        with patch.dict(os.environ, {"PATH": "/usr/bin", "BOT_TOKEN": "secret"}), \
             patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as create, \
             patch(
                 "hermes_cli.quick_commands.communicate_bounded_async",
                 AsyncMock(return_value=(b"saved\n", b"")),
             ):
            result = await runner._handle_message(event)

        assert result == "saved"
        assert create.await_args.args == (
            "/opt/companion/bin/remember-spool-append",
            "--type",
            "fact",
            "hello; echo not-a-shell",
        )
        child_env = create.await_args.kwargs["env"]
        assert child_env["HERMES_QUICK_COMMAND_PLATFORM"] == "telegram"
        assert child_env["HERMES_QUICK_COMMAND_MESSAGE_ID"] == "msg-42"
        assert child_env["HERMES_QUICK_COMMAND_UPDATE_ID"] == "77"
        assert child_env["HERMES_QUICK_COMMAND_DESTINATION_ALIAS"] == "owner"
        assert child_env["PATH"] == "/usr/bin"
        assert "BOT_TOKEN" not in child_env
        assert "shell" not in create.await_args.kwargs

    @pytest.mark.asyncio
    async def test_multiplex_argv_uses_routed_profile_config_and_access(
        self, tmp_path
    ):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.session import SessionSource

        qcmd = {
            "type": "argv",
            "command": ["/opt/primary-only"],
            "destination_alias": "owner",
        }
        runner = self._make_runner(qcmd)
        runner.config = GatewayConfig(
            multiplex_profiles=True,
            quick_commands={"remember": qcmd},
        )
        event = self._make_event("remember")
        event.source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="test_user",
            profile="secondary",
        )
        secondary_home = tmp_path / "profiles" / "secondary"
        secondary_cfg = GatewayConfig(
            multiplex_profiles=True,
            quick_commands={"remember": qcmd},
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="***",
                    extra={
                        "allow_admin_from": ["secondary-admin"],
                        "user_allowed_commands": [],
                    },
                )
            },
        )
        runner._resolve_profile_home_for_source = MagicMock(
            return_value=secondary_home
        )

        with (
            patch("gateway.config.load_gateway_config", return_value=secondary_cfg),
            patch("asyncio.create_subprocess_exec", AsyncMock()) as create,
        ):
            result = await runner._handle_message(event)

        create.assert_not_awaited()
        assert "admin-only" in result

    @pytest.mark.asyncio
    async def test_multiplex_secondary_cannot_run_primary_only_argv(self, tmp_path):
        from gateway.config import GatewayConfig, Platform
        from gateway.session import SessionSource

        qcmd = {"type": "argv", "command": ["/opt/primary-only"]}
        runner = self._make_runner(qcmd)
        runner.config = GatewayConfig(
            multiplex_profiles=True,
            quick_commands={"remember": qcmd},
        )
        runner._handle_message_with_agent = AsyncMock(return_value="agent fallback")
        event = self._make_event("remember")
        event.source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="test_user",
            profile="secondary",
        )
        secondary_home = tmp_path / "profiles" / "secondary"
        secondary_cfg = GatewayConfig(
            multiplex_profiles=True,
            quick_commands={},
        )
        runner._resolve_profile_home_for_source = MagicMock(
            return_value=secondary_home
        )

        with (
            patch("gateway.config.load_gateway_config", return_value=secondary_cfg),
            patch("asyncio.create_subprocess_exec", AsyncMock()) as create,
        ):
            result = await runner._handle_message(event)

        create.assert_not_awaited()
        assert "unknown command" in result.lower()

    @pytest.mark.asyncio
    async def test_multiplex_secondary_only_alias_reaches_builtin(self, tmp_path):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.session import SessionSource

        runner = self._make_runner({})
        runner.config = GatewayConfig(
            multiplex_profiles=True,
            quick_commands={},
        )
        runner._handle_status_command = AsyncMock(return_value="secondary status")
        runner._handle_message_with_agent = AsyncMock(return_value="agent fallback")
        event = self._make_event("secondary-status")
        event.source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="secondary-user",
            profile="secondary",
        )
        secondary_home = tmp_path / "profiles" / "secondary"
        secondary_cfg = GatewayConfig(
            multiplex_profiles=True,
            quick_commands={
                "secondary-status": {"type": "alias", "target": "/status"}
            },
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="***",
                    extra={
                        "allow_admin_from": ["secondary-admin"],
                        "user_allowed_commands": ["status"],
                    },
                )
            },
        )
        runner._resolve_profile_home_for_source = MagicMock(
            return_value=secondary_home
        )

        with patch("gateway.config.load_gateway_config", return_value=secondary_cfg):
            result = await runner._handle_message(event)

        assert result == "secondary status"
        runner._handle_status_command.assert_awaited_once_with(event)
        runner._handle_message_with_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiplex_primary_alias_cannot_override_secondary_argv(
        self, tmp_path
    ):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.session import SessionSource

        runner = self._make_runner({})
        runner.config = GatewayConfig(
            multiplex_profiles=True,
            quick_commands={"remember": {"type": "alias", "target": "/stop"}},
        )
        event = self._make_event("remember")
        event.source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="secondary-user",
            profile="secondary",
        )
        secondary_home = tmp_path / "profiles" / "secondary"
        secondary_cfg = GatewayConfig(
            multiplex_profiles=True,
            quick_commands={
                "remember": {"type": "argv", "command": ["/opt/secondary-save"]}
            },
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="***",
                    extra={
                        "allow_admin_from": ["secondary-admin"],
                        "user_allowed_commands": ["remember"],
                    },
                )
            },
        )
        runner._resolve_profile_home_for_source = MagicMock(
            return_value=secondary_home
        )
        proc = MagicMock(returncode=0)

        with (
            patch("gateway.config.load_gateway_config", return_value=secondary_cfg),
            patch(
                "asyncio.create_subprocess_exec", AsyncMock(return_value=proc)
            ) as create,
            patch(
                "hermes_cli.quick_commands.communicate_bounded_async",
                AsyncMock(return_value=(b"saved in secondary\n", b"")),
            ),
        ):
            result = await runner._handle_message(event)

        assert result == "saved in secondary"
        assert create.await_args.args == ("/opt/secondary-save",)

    @pytest.mark.asyncio
    async def test_multiplex_builtin_access_uses_secondary_policy(self, tmp_path):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.session import SessionSource

        runner = self._make_runner({})
        runner.config = GatewayConfig(
            multiplex_profiles=True,
            quick_commands={},
        )
        event = self._make_event("stop")
        event.source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="secondary-user",
            profile="secondary",
        )
        secondary_home = tmp_path / "profiles" / "secondary"
        secondary_cfg = GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="***",
                    extra={
                        "allow_admin_from": ["secondary-admin"],
                        "user_allowed_commands": [],
                    },
                )
            },
        )
        runner._resolve_profile_home_for_source = MagicMock(
            return_value=secondary_home
        )

        with patch("gateway.config.load_gateway_config", return_value=secondary_cfg):
            result = await runner._handle_message(event)

        assert result is not None
        assert "/stop is admin-only here" in result

    @pytest.mark.asyncio
    @pytest.mark.live_system_guard_bypass
    async def test_gateway_argv_reader_hang_reaps_forked_descendant(
        self, monkeypatch, tmp_path
    ):
        pid_path = tmp_path / "descendant.pid"
        heartbeat_path = tmp_path / "descendant.heartbeat"
        descendant = (
            "import pathlib,sys,time\n"
            "path=pathlib.Path(sys.argv[1])\n"
            "while True:\n"
            " path.write_text(str(time.time_ns()))\n"
            " time.sleep(0.01)\n"
        )
        script = (
            "import pathlib,subprocess,sys,time\n"
            "heartbeat=pathlib.Path(sys.argv[2])\n"
            "child=subprocess.Popen([sys.executable,'-c',sys.argv[3],str(heartbeat)])\n"
            "deadline=time.monotonic()+2\n"
            "while not heartbeat.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
        )
        runner = self._make_runner({
            "type": "argv",
            "command": [
                sys.executable,
                "-c",
                script,
                str(pid_path),
                str(heartbeat_path),
                descendant,
            ],
        })
        monkeypatch.setattr(
            "hermes_cli.quick_commands.QUICK_COMMAND_TIMEOUT_SECONDS", 0.2
        )

        result = await runner._handle_message(self._make_event("remember"))

        assert "timed out" in result.lower()
        descendant_pid = int(pid_path.read_text())
        before = heartbeat_path.read_text()
        await asyncio.sleep(0.1)
        assert heartbeat_path.read_text() == before, (
            f"forked argv descendant {descendant_pid} survived timeout"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "qcmd",
        [
            "not-a-mapping",
            {"type": "argv", "command": "echo unsafe", "destination_alias": "owner"},
            {"type": "argv", "command": [], "destination_alias": "owner"},
            {"type": "argv", "command": ["/bin/echo"], "destination_alias": ""},
            {"type": "argv", "command": ["/bin/echo"], "destination_alias": "bad alias"},
            {"type": "argv", "command": ["/bin/echo"], "argument_mode": "words", "destination_alias": "owner"},
        ],
    )
    async def test_gateway_argv_malformed_config_is_user_facing(self, qcmd):
        runner = self._make_runner(qcmd)
        with patch("asyncio.create_subprocess_exec", AsyncMock()) as create:
            result = await runner._handle_message(self._make_event("remember", "hello"))

        create.assert_not_awaited()
        assert "quick command" in result.lower()

    @pytest.mark.asyncio
    async def test_gateway_argv_timeout_is_user_facing(self):
        runner = self._make_runner({
            "type": "argv",
            "command": ["/bin/echo"],
            "destination_alias": "owner",
        })
        proc = MagicMock()

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), \
             patch(
                 "hermes_cli.quick_commands.communicate_bounded_async",
                 AsyncMock(side_effect=asyncio.TimeoutError),
             ):
            result = await runner._handle_message(self._make_event("remember"))

        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_gateway_argv_output_is_bounded(self):
        runner = self._make_runner({
            "type": "argv",
            "command": ["/bin/echo"],
            "destination_alias": "owner",
        })
        from hermes_cli.quick_commands import QuickCommandOutputError

        proc = MagicMock()
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), \
             patch(
                 "hermes_cli.quick_commands.communicate_bounded_async",
                 AsyncMock(
                     side_effect=QuickCommandOutputError(
                         "output exceeds 65536 UTF-8 bytes"
                     )
                 ),
             ):
            result = await runner._handle_message(self._make_event("remember"))

        assert "65536 utf-8 bytes" in result.lower()

    @pytest.mark.asyncio
    async def test_gateway_argv_nonzero_exit_is_not_reported_as_success(self):
        runner = self._make_runner({
            "type": "argv",
            "command": ["/bin/false"],
            "destination_alias": "owner",
        })
        proc = MagicMock(returncode=7)
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), \
             patch(
                 "hermes_cli.quick_commands.communicate_bounded_async",
                 AsyncMock(return_value=(b"", b"spool failed")),
             ):
            result = await runner._handle_message(self._make_event("remember"))

        assert "failed" in result.lower()
        assert "spool failed" in result.lower()

    @pytest.mark.asyncio
    async def test_gateway_argv_without_destination_alias_remains_generic(self):
        runner = self._make_runner({
            "type": "argv",
            "command": ["/bin/echo"],
        })
        proc = MagicMock(returncode=0)
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as create, \
             patch(
                 "hermes_cli.quick_commands.communicate_bounded_async",
                 AsyncMock(return_value=(b"ok", b"")),
             ):
            result = await runner._handle_message(self._make_event("remember"))

        assert result == "ok"
        assert "HERMES_QUICK_COMMAND_DESTINATION_ALIAS" not in create.await_args.kwargs["env"]
