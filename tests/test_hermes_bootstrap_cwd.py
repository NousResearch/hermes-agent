"""hermes_bootstrap keeps a process alive when its inherited cwd is unreadable.

Kanban workers inherit the dispatcher's cwd; on a stalled SMB mount getcwd()
raises EINTR and the worker used to die before parsing its argv (rc=0, no
tool call -> protocol_violation).
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import hermes_bootstrap


def _getcwd_failing(times: int, err: OSError):
    calls = {"n": 0}

    def getcwd():
        calls["n"] += 1
        if calls["n"] <= times:
            raise err
        return "/"

    return getcwd, calls


def test_transient_eintr_is_retried_and_the_cwd_kept(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    getcwd, calls = _getcwd_failing(2, InterruptedError(errno.EINTR, "Interrupted system call"))
    hermes_bootstrap.ensure_readable_cwd(getcwd=getcwd, sleep=lambda _s: None)
    assert calls["n"] == 3
    assert Path.cwd() == tmp_path


def test_persistent_eintr_falls_back_to_hermes_home_and_says_so(tmp_path, monkeypatch, capsys):
    work, home = tmp_path / "work", tmp_path / "home"
    work.mkdir(); home.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HERMES_HOME", str(home))
    getcwd, _ = _getcwd_failing(99, InterruptedError(errno.EINTR, "Interrupted system call"))
    hermes_bootstrap.ensure_readable_cwd(getcwd=getcwd, sleep=lambda _s: None)
    assert Path.cwd() == home
    assert f"continuing from {home}" in capsys.readouterr().err


def test_missing_hermes_home_falls_back_to_user_home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "absent"))
    monkeypatch.setenv("HOME", str(tmp_path))
    getcwd, _ = _getcwd_failing(99, OSError(errno.ETIMEDOUT, "Operation timed out"))
    hermes_bootstrap.ensure_readable_cwd(getcwd=getcwd, sleep=lambda _s: None)
    assert Path.cwd() == tmp_path


def test_deleted_cwd_still_recovers_to_the_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    getcwd, _ = _getcwd_failing(1, FileNotFoundError(errno.ENOENT, "gone"))
    hermes_bootstrap.ensure_readable_cwd(getcwd=getcwd, sleep=lambda _s: None)
    assert Path(os.getcwd()) == hermes_bootstrap._root


def test_terminal_default_cwd_survives_eintr(monkeypatch, tmp_path):
    from tools import terminal_tool_config

    def eintr():
        raise InterruptedError(errno.EINTR, "Interrupted system call")

    monkeypatch.setattr(terminal_tool_config.os, "getcwd", eintr)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    assert terminal_tool_config._safe_getcwd() == str(tmp_path)
