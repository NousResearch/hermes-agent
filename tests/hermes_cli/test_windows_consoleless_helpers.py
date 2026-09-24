"""Regression coverage for helper processes spawned by console-less Windows backends."""

import subprocess

import pytest

from hermes_cli import gateway, gitlock, update_cmd, update_cmd_git


@pytest.mark.windows_only
def test_backend_helper_spawns_hide_their_console(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append((command, kwargs))
        if command[0] == "powershell.exe":
            stdout = "MISSING\n"
        elif command[0] == "tasklist":
            stdout = '"Image Name","PID"\n'
        elif "--abbrev-ref" in command:
            stdout = "main\n"
        elif "--short" in command:
            stdout = "abc123\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(gateway.shutil, "which", lambda name: "powershell.exe" if name == "powershell" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gateway._windows_scheduled_task_state("HermesGateway") == "MISSING"
    assert gitlock._git_proc_running() is False
    assert update_cmd._git_run(["git"], ["status"], cwd=tmp_path).returncode == 0
    assert update_cmd_git._branch_head_label(["git"], tmp_path) == "main @ abc123"

    assert len(calls) == 5
    assert all(kwargs.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW for _, kwargs in calls)
