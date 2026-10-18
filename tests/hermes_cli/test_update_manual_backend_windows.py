"""Windows update recovery of manual backends (regression for #131864)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import MagicMock

import psutil
import pytest

from hermes_cli import dashboard_procs, main_dashboard
from hermes_cli._subprocess_compat import (
    windows_detach_flags,
    windows_detach_flags_without_breakaway,
    windows_hide_flags,
)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("mode, outcome", [
    ("serve", "running"),
    ("dashboard", "running"),
    ("serve", "breakaway-denied"),
    ("dashboard", "spawn-failed"),
    ("serve", "exited"),
    ("serve", "access-denied"),
    ("serve", "gone"),
    ("serve", "empty-argv"),
    ("serve", "port-zero"),
    ("dashboard", "port-zero"),
    ("serve", "foreign-home"),
    ("serve", "explicit-stop"),
])
def test_windows_cleanup_preserves_manual_launch_and_recovery_outcome(
    tmp_path, monkeypatch, mode, outcome,
):
    root = tmp_path / "Hermes Data"
    home = root / "profiles" / "research"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DESKTOP_CHILD_PID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pid = 5555
    argv = [
        r"C:\Runtime Dir\python.exe", "-m", "hermes_cli.main",
        "--profile", "research", mode, "--host", "::1", "--port",
        "0" if outcome == "port-zero" else "9119", "--skip-build",
    ]
    stopped = False

    class Process:
        def cmdline(self):
            assert not stopped, "launch arguments must be captured before stopping"
            if outcome == "access-denied":
                raise psutil.AccessDenied(pid)
            if outcome == "gone":
                raise psutil.NoSuchProcess(pid)
            return [] if outcome == "empty-argv" else list(argv)

        def environ(self):
            assert not stopped, "profile ownership must be captured before stopping"
            source = tmp_path / "Other Install" if outcome == "foreign-home" else root
            return {"HERMES_HOME": str(source)}

    monkeypatch.setattr(psutil, "Process", lambda target: Process())
    monkeypatch.setattr(main_dashboard, "_find_stale_dashboard_pids", lambda **kw: [pid])

    def no_posix_probe(*args, **kwargs):
        pytest.fail("Windows recovery must not probe systemd or launchd")

    for name in ("_get_pid_cgroup_path", "_get_systemd_service_for_pid", "_loaded_launchd_backend_jobs"):
        monkeypatch.setattr(main_dashboard, name, no_posix_probe)

    def stop(pids, killed, failed):
        nonlocal stopped
        assert pids == [pid]
        stopped = True
        killed.extend(pids)

    monkeypatch.setattr(dashboard_procs, "_kill_pids_windows", stop)
    attempts = []

    def spawn(command, **kwargs):
        assert stopped
        assert Path(kwargs["stdout"].name) == home / "logs" / "dashboard-restart.log"
        attempts.append((list(command), kwargs))
        if outcome == "spawn-failed":
            raise OSError("cannot start backend")
        if outcome == "breakaway-denied" and len(attempts) == 1:
            raise PermissionError("parent job forbids breakaway")
        return MagicMock(poll=lambda: 1 if outcome == "exited" else None, returncode=1)

    monkeypatch.setattr(main_dashboard.subprocess, "Popen", spawn)
    monkeypatch.setattr(main_dashboard.time, "sleep", lambda _: None)
    result = dashboard_procs._kill_stale_dashboard_processes(
        restart_managed=outcome != "explicit-stop", scope_home=str(home),
    )

    assert result["killed"] == [pid]
    assert result["failed"] == []
    unrecovered = outcome in {
        "spawn-failed", "exited", "access-denied", "gone", "empty-argv", "explicit-stop",
    }
    assert result["unrecovered"] == ([pid] if unrecovered else [])
    if outcome in {"running", "breakaway-denied", "spawn-failed", "exited"}:
        expected = argv + (["--no-open"] if mode == "dashboard" else [])
        assert all(command == expected for command, _ in attempts)
        assert attempts[0][1]["creationflags"] == windows_detach_flags()
        assert "start_new_session" not in attempts[0][1]
        if outcome in {"breakaway-denied", "spawn-failed"}:
            assert len(attempts) == 2
            assert attempts[1][1]["creationflags"] == windows_detach_flags_without_breakaway()
        else:
            assert len(attempts) == 1
    else:
        assert attempts == []


@pytest.mark.platforms("windows")
def test_windows_cleanup_restarts_a_live_child_with_exact_argv(tmp_path, monkeypatch):
    """Use real psutil, taskkill, file I/O and Popen on test-owned process trees."""
    monkeypatch.delenv("HERMES_DESKTOP_CHILD_PID", raising=False)
    homes = [tmp_path / "profiles" / name for name in ("research", "review")]
    for home in homes:
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
    script = tmp_path / "resident backend.py"
    script.write_text(
        "import json, os, sys, time\n"
        "with open(sys.argv[1], 'a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps({'pid': os.getpid(), 'args': sys.argv[1:], "
        "'home': os.environ['HERMES_HOME']}) + '\\n')\n"
        "    stream.flush()\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    popen = subprocess.Popen
    children = []

    def spawn(command, **kwargs):
        child = popen(command, **kwargs)
        if command == argv:
            children.append(child)
        return child

    def launches(count):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rows = [json.loads(line) for line in record.read_text(encoding="utf-8-sig").splitlines()] \
                if record.exists() else []
            if len(rows) >= count:
                return rows
            time.sleep(0.05)
        pytest.fail(f"expected {count} backend launches")

    try:
        monkeypatch.setattr(main_dashboard.subprocess, "Popen", spawn)
        for index, home in enumerate((homes[0], homes[1], homes[0])):
            monkeypatch.setenv("HERMES_HOME", str(home))
            record = tmp_path / f"launch records {index}.jsonl"
            argv = [sys.executable, str(script), str(record), "serve", "--host", "127.0.0.1",
                    "--port", "9119", "--skip-build"]
            original = spawn(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=windows_hide_flags())
            before = launches(1)[0]
            assert main_dashboard._dashboard_cmdline_for_pid(original.pid) == argv
            monkeypatch.setattr(main_dashboard, "_find_stale_dashboard_pids", lambda **kw: [original.pid])
            result = dashboard_procs._kill_stale_dashboard_processes(
                restart_managed=True, scope_home=str(home),
            )
            assert result["killed"] == [original.pid]
            assert result["failed"] == []
            assert result["unrecovered"] == []
            after = launches(2)[1]
            assert after["pid"] != before["pid"]
            assert after["args"] == before["args"] == argv[2:]
            assert after["home"] == before["home"] == str(home)
            assert original.wait(timeout=5) is not None
            assert children[-1].poll() is None
            assert (home / "logs" / "dashboard-restart.log").is_file()
            dashboard_procs._kill_pids_windows([children[-1].pid], [], [])
            children[-1].wait(timeout=5)
    finally:
        # The venv launcher can own a worker; stop the whole test-owned tree.
        alive = [child.pid for child in children if child.poll() is None]
        if alive:
            dashboard_procs._kill_pids_windows(alive, [], [])
        for child in children:
            child.wait(timeout=5)
