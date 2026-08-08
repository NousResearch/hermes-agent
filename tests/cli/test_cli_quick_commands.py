"""Tests for classic CLI quick-command execution."""

import io
import os
import shlex
import subprocess
from unittest.mock import MagicMock, patch

import pytest

import cli as cli_mod
from cli import HermesCLI


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {
        "quick_commands": {
            "safe": {"type": "exec", "command": "echo safe"},
            "danger": {"type": "exec", "command": "rm -rf /tmp/demo"},
        }
    }
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "sess-quick-command-test"
    cli_obj._pending_input = MagicMock()
    cli_obj._pending_resume_sessions = None
    cli_obj._agent_running = False
    cli_obj._app = None
    return cli_obj


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        wait_error: Exception | None = None,
    ):
        self.returncode = returncode
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.wait_error = wait_error
        self.pid = 12345
        self.killed = False
        self.wait_timeout = None

    def wait(self, timeout=None):
        self.wait_timeout = timeout
        if self.wait_error is not None:
            raise self.wait_error
        return self.returncode

    def kill(self):
        self.killed = True


def _assert_quick_command_popen_kwargs(call_kwargs, *, cwd):
    assert call_kwargs["shell"] is True
    assert call_kwargs["stdout"] is cli_mod.subprocess.PIPE
    assert call_kwargs["stderr"] is cli_mod.subprocess.PIPE
    assert call_kwargs["cwd"] == cwd
    assert call_kwargs["stdin"] is cli_mod.subprocess.DEVNULL
    assert "env" in call_kwargs
    assert call_kwargs.get("text") in (None, False)
    assert call_kwargs.get("universal_newlines") in (None, False)
    assert call_kwargs.get("encoding") is None
    assert call_kwargs.get("errors") is None
    for key, value in cli_mod._quick_command_popen_kwargs().items():
        assert call_kwargs[key] == value


def test_quick_command_rejects_empty_input():
    with patch.object(cli_mod.subprocess, "Popen") as popen_mock:
        result = cli_mod._run_quick_command("   ")

    assert result == {"ok": False, "message": "empty command"}
    popen_mock.assert_not_called()


def test_quick_command_rejects_non_string_input():
    with patch.object(cli_mod.subprocess, "Popen") as popen_mock:
        result = cli_mod._run_quick_command(["echo safe"])

    assert result == {"ok": False, "message": "quick command must be a string"}
    popen_mock.assert_not_called()


def test_quick_command_normalizes_before_execution(monkeypatch):
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    with patch.object(
        cli_mod.subprocess,
        "Popen",
        return_value=_FakeProcess(stdout=b"ok\n"),
    ) as popen_mock:
        result = cli_mod._run_quick_command("\n  echo safe  \n")

    assert result == {"ok": True, "output": "ok", "returncode": 0}
    assert popen_mock.call_args.args == ("echo safe",)


def test_quick_command_handles_timeout():
    with patch.object(
        cli_mod.subprocess,
        "Popen",
        return_value=_FakeProcess(
            wait_error=subprocess.TimeoutExpired("sleep 100", 30)
        ),
    ), patch("cli._terminate_quick_command_tree") as terminate_mock:
        fake_proc = cli_mod.subprocess.Popen.return_value
        result = cli_mod._run_quick_command("sleep 100")

    assert result == {"ok": False, "message": "command timed out (30s)"}
    assert fake_proc.wait_timeout == 30
    terminate_mock.assert_called_once_with(fake_proc)


def test_quick_command_runs_in_terminal_cwd(monkeypatch):
    terminal_cwd = "/tmp/hermes-quick-command-cwd"
    monkeypatch.setenv("TERMINAL_CWD", terminal_cwd)

    with patch.object(
        cli_mod.subprocess,
        "Popen",
        return_value=_FakeProcess(stdout=b"ok\n"),
    ) as popen_mock:
        result = cli_mod._run_quick_command("pwd")

    assert result == {"ok": True, "output": "ok", "returncode": 0}
    _assert_quick_command_popen_kwargs(popen_mock.call_args.kwargs, cwd=terminal_cwd)


def test_quick_command_sanitizes_subprocess_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abc123def456ghi789jkl012mno345")
    monkeypatch.setenv("HERMES_QUICK_COMMAND_TEST_MARKER", "kept")

    with patch.object(
        cli_mod.subprocess,
        "Popen",
        return_value=_FakeProcess(stdout=b"ok\n"),
    ) as popen_mock:
        result = cli_mod._run_quick_command("env")

    assert result == {"ok": True, "output": "ok", "returncode": 0}
    child_env = popen_mock.call_args.kwargs["env"]
    assert child_env["HERMES_QUICK_COMMAND_TEST_MARKER"] == "kept"
    assert "OPENAI_API_KEY" not in child_env


def test_quick_command_redacts_output_before_returning():
    secret = "sk-proj-abc123def456ghi789jkl012mno345"

    with patch.object(
        cli_mod.subprocess,
        "Popen",
        return_value=_FakeProcess(stdout=f"OPENAI_API_KEY={secret}\n".encode()),
    ):
        result = cli_mod._run_quick_command("show-secret")

    assert result["ok"] is True
    assert secret not in result["output"]
    assert "OPENAI_API_KEY=" in result["output"]


def test_quick_command_truncates_large_output(monkeypatch):
    monkeypatch.setattr(cli_mod, "_QUICK_COMMAND_OUTPUT_LIMIT_PER_STREAM", 8)

    with patch.object(
        cli_mod.subprocess,
        "Popen",
        return_value=_FakeProcess(stdout=b"abcdefghijklmnop"),
    ):
        result = cli_mod._run_quick_command("printf lots")

    assert result["ok"] is True
    assert (
        result["output"]
        == "abcdefgh\n[output truncated after 8 bytes from this stream]"
    )


def test_process_command_exec_quick_command_prints_stdout_and_stderr():
    cli_obj = _make_cli()

    with patch("cli._ensure_skill_commands", return_value={}), patch(
        "cli.get_skill_bundles", return_value=[]
    ), patch.object(
        cli_mod.subprocess,
        "Popen",
        return_value=_FakeProcess(returncode=0, stdout=b"stdout\n", stderr=b"stderr\n"),
    ) as popen_mock, patch.dict(os.environ, {"TERMINAL_CWD": ""}):
        assert cli_obj.process_command("/safe") is True

    popen_mock.assert_called_once()
    assert popen_mock.call_args.args == ("echo safe",)
    _assert_quick_command_popen_kwargs(popen_mock.call_args.kwargs, cwd=os.getcwd())
    printed = "\n".join(str(call.args[0]) for call in cli_obj.console.print.call_args_list)
    assert "stdout" in printed
    assert "stderr" in printed


@pytest.mark.skipif(os.name == "nt", reason="regression uses a POSIX shell command")
def test_process_command_executes_operator_configured_command_without_agent_guard(
    tmp_path,
):
    from tools.approval import detect_dangerous_command

    cli_obj = _make_cli()
    target = tmp_path / "operator-target"
    target.mkdir()
    command = f"rm -rf {shlex.quote(str(target))}"
    is_dangerous, _, _ = detect_dangerous_command(command)
    assert is_dangerous is True
    cli_obj.config["quick_commands"]["operator"] = {
        "type": "exec",
        "command": command,
    }

    with patch("cli._ensure_skill_commands", return_value={}), patch(
        "cli.get_skill_bundles", return_value=[]
    ):
        assert cli_obj.process_command("/operator") is True

    assert not target.exists()
    printed = "\n".join(str(call.args[0]) for call in cli_obj.console.print.call_args_list)
    assert "Command returned no output" in printed


def test_process_command_reports_nonzero_quick_command_as_error():
    cli_obj = _make_cli()

    with patch("cli._ensure_skill_commands", return_value={}), patch(
        "cli.get_skill_bundles", return_value=[]
    ), patch.object(
        cli_mod.subprocess,
        "Popen",
        return_value=_FakeProcess(returncode=7, stdout=b"stdout\n", stderr=b"stderr\n"),
    ):
        assert cli_obj.process_command("/safe") is True

    printed = "\n".join(str(call.args[0]) for call in cli_obj.console.print.call_args_list)
    assert "Quick command error" in printed
    assert "stdout" in printed
    assert "stderr" in printed
