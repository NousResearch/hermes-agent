"""Exercise classic CLI quick commands through their actual subprocess boundary."""
import os
import shlex
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from cli import HermesCLI


def _python_command(code):
    args = [sys.executable, "-c", code]
    return subprocess.list2cmdline(args) if sys.platform == "win32" else shlex.join(args)


def _cli_result(command):
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"quick_commands": {"qc": {"type": "exec", "command": command}}}
    cli.console = MagicMock()
    cli.agent = None
    cli.conversation_history = []
    cli.session_id = "quick-command-regression"
    assert cli.process_command("/qc") is True
    return "\n".join(str(call.args[0]) for call in cli.console.print.call_args_list)


def test_quick_command_respects_terminal_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    assert str(tmp_path) in _cli_result(_python_command("import os; print(os.getcwd())"))


def test_quick_command_preserves_both_streams():
    output = _cli_result(_python_command("import sys; print('stdout-marker'); print('stderr-marker', file=sys.stderr)"))
    assert "stdout-marker" in output and "stderr-marker" in output


def test_quick_command_reports_exit_status():
    output = _cli_result(_python_command("raise SystemExit(7)"))
    assert "Quick command error" in output and "7" in output


def test_quick_command_closes_stdin(monkeypatch):
    original = subprocess.Popen
    calls = []

    def spawn(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    assert "stdin-check" in _cli_result(_python_command("print('stdin-check')"))
    assert calls[-1]["stdin"] is subprocess.DEVNULL


def test_quick_command_caps_large_output():
    output = _cli_result(_python_command("import sys; sys.stdout.write('x' * 200000)"))
    assert len(output) < 70000
    assert "truncated" in output


def test_quick_command_sanitizes_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abc123def456ghi789jkl012mno345")
    monkeypatch.setenv("HERMES_QUICK_COMMAND_TEST_MARKER", "visible-marker")
    code = "import os; print(os.environ.get('OPENAI_API_KEY', 'absent-key')); print(os.environ['HERMES_QUICK_COMMAND_TEST_MARKER'])"
    output = _cli_result(_python_command(code))
    assert "absent-key" in output and "visible-marker" in output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX operator command")
def test_quick_command_preserves_operator_shell_contract(tmp_path):
    from tools.approval import detect_dangerous_command

    target = tmp_path / "operator-owned-target"
    target.mkdir()
    (target / "file").write_text("owned test data")
    command = "rm -rf " + shlex.quote(str(target)) + " && echo operator-command-ran"
    assert detect_dangerous_command(command)
    assert "operator-command-ran" in _cli_result(command)
    assert not target.exists()


@pytest.mark.parametrize("command", ["", "   ", ["echo", "hi"]])
def test_quick_command_rejects_malformed_command(command):
    from hermes_cli.quick_command_runner import run_quick_command

    assert run_quick_command(command)["ok"] is False


@pytest.mark.parametrize("detached_launcher", [False, True])
def test_quick_command_timeout_terminates_children_and_drains_pipes(tmp_path, detached_launcher):
    import threading
    import time
    import psutil
    from hermes_cli.quick_command_runner import run_quick_command

    pid_path = tmp_path / "child.pid"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); "
            f"Path({str(pid_path)!r}).write_text(str(child.pid)); "
            + ("print('launcher-finished')" if detached_launcher else "time.sleep(20)"))
    start = time.monotonic()
    result = run_quick_command(_python_command(code), timeout=0.5)
    assert "timed out" in result["message"]
    assert time.monotonic() - start < 5
    assert pid_path.exists()
    try:
        child = psutil.Process(int(pid_path.read_text()))
        assert child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    assert not any(t.name == "quick-command-output" and t.is_alive() for t in threading.enumerate())


def test_quick_command_cancellation_cleans_up_before_propagating(monkeypatch):
    import threading
    import hermes_cli.quick_command_runner as runner

    original_join = threading.Thread.join
    original_spawn = subprocess.Popen
    spawned = []
    interrupted = False

    def spawn(*args, **kwargs):
        proc = original_spawn(*args, **kwargs)
        spawned.append(proc)
        return proc

    def join(thread, *args, **kwargs):
        nonlocal interrupted
        if thread.name == "quick-command-output" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_join(thread, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(threading.Thread, "join", join)
    with pytest.raises(KeyboardInterrupt):
        runner.run_quick_command(_python_command("import time; time.sleep(20)"))
    assert spawned[0].poll() is not None
    assert not any(t.name == "quick-command-output" and t.is_alive() for t in threading.enumerate())


def test_quick_command_reader_start_failure_closes_process_and_pipes(monkeypatch):
    import threading
    import hermes_cli.quick_command_runner as runner

    original = threading.Thread.start
    count = 0

    def start(thread):
        nonlocal count
        if thread.name == "quick-command-output":
            count += 1
            if count == 2:
                raise RuntimeError("reader could not start")
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    result = runner.run_quick_command(_python_command("import time; time.sleep(20)"))
    assert result == {"ok": False, "message": "reader could not start"}
    assert not any(t.name == "quick-command-output" and t.is_alive() for t in threading.enumerate())


@pytest.mark.parametrize("returncode", [0, 7])
def test_quick_command_redacts_both_success_and_failure_output(returncode):
    from hermes_cli.quick_command_runner import run_quick_command

    secret = "sk-proj-abc123def456ghi789jkl012mno345"
    result = run_quick_command(_python_command(f"print('OPENAI_API_KEY={secret}'); raise SystemExit({returncode})"))
    assert secret not in (result.get("output") or result["message"])
    assert result["returncode"] == returncode


def test_quick_command_spawn_error_is_redacted(monkeypatch):
    from hermes_cli.quick_command_runner import run_quick_command

    secret = "sk-proj-abc123def456ghi789jkl012mno345"

    def spawn(*args, **kwargs):
        raise OSError("OPENAI_API_KEY=" + secret)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    result = run_quick_command("configured-command")
    assert result["ok"] is False and secret not in result["message"]


def test_quick_command_hides_windows_console(monkeypatch):
    import io
    from types import SimpleNamespace
    import hermes_cli.quick_command_runner as runner

    flags = 0x08000000
    calls = []

    def spawn(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stdout=io.BytesIO(b"hidden-console\n"), stderr=io.BytesIO(), wait=lambda timeout: 0)

    monkeypatch.setattr(runner, "IS_WINDOWS", True)
    monkeypatch.setattr(runner, "windows_hide_flags", lambda: flags)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    assert runner.run_quick_command("echo hidden-console")["ok"] is True
    assert calls[0]["creationflags"] == flags
    assert "start_new_session" not in calls[0]
