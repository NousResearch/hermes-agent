"""Cron script timeouts must terminate the whole subprocess tree."""

import signal
import subprocess
from unittest.mock import MagicMock

import cron.scheduler as scheduler
import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _script(hermes_env):
    path = hermes_env / "scripts" / "hung.py"
    path.write_text("print('started')\n", encoding="utf-8")
    return path


def _timed_out_process():
    proc = MagicMock()
    proc.pid = 4242
    proc.returncode = None
    proc.communicate.side_effect = [
        subprocess.TimeoutExpired(["python", "hung.py"], 5),
        ("", ""),
    ]
    return proc


def test_posix_timeout_kills_script_process_group(hermes_env, monkeypatch):
    _script(hermes_env)
    proc = _timed_out_process()
    killpg_calls = []

    monkeypatch.setattr(scheduler, "_get_script_timeout", lambda: 5)
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda *a, **kw: proc)
    monkeypatch.setattr(
        scheduler.subprocess,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(a[0], kw.get("timeout", 5))
        ),
    )
    monkeypatch.setattr(scheduler.sys, "platform", "darwin")
    monkeypatch.setattr(
        scheduler.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig))
    )

    ok, message = scheduler._run_job_script("hung.py")

    assert ok is False
    assert "timed out" in message
    assert killpg_calls == [(proc.pid, signal.SIGKILL)]
    assert proc.communicate.call_args_list[0].kwargs == {"timeout": 5}
    assert proc.communicate.call_args_list[1].kwargs == {"timeout": 1.0}


def test_windows_timeout_uses_taskkill_tree(hermes_env, monkeypatch):
    _script(hermes_env)
    proc = _timed_out_process()
    taskkill_calls = []

    monkeypatch.setattr(scheduler, "_get_script_timeout", lambda: 5)
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda *a, **kw: proc)
    monkeypatch.setattr(scheduler.sys, "platform", "win32")
    monkeypatch.setattr(
        scheduler,
        "_windows_cron_python_invocation",
        lambda _python: ("python.exe", {}),
    )

    def fake_run(argv, **kwargs):
        if argv[0] == "taskkill":
            taskkill_calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0)
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 5))

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)

    ok, message = scheduler._run_job_script("hung.py")

    assert ok is False
    assert "timed out" in message
    assert taskkill_calls[0][0] == [
        "taskkill", "/T", "/F", "/PID", str(proc.pid)
    ]
