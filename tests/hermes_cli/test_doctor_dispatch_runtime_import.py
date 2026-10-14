"""Doctor's kanban dispatch foreign-cwd import check.

The failure mode this guards: the dispatcher spawns workers as
``python -m hermes_cli.main`` from the task workspace with the import context
built by ``_propagate_module_import_root``. The check must reproduce that exact
spawn condition -- the dispatcher's own argv and env, from a real foreign cwd --
so a pin that cannot deliver ``hermes_cli`` fails loudly in doctor instead of
auto-blocking every task on the board.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import doctor_state


def _fake_run_factory(returncode: int, stdout: str, stderr: str):
    calls = {}

    def fake_run(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return fake_run, calls


def test_probe_uses_dispatcher_spawn_context(monkeypatch, capsys):
    fake_run, calls = _fake_run_factory(0, "Hermes Agent v0.21.5\n", "")
    monkeypatch.setattr(doctor_state.subprocess, "run", fake_run)

    finding = doctor_state._check_dispatch_runtime_import(False)

    assert finding.issues == []
    argv = calls["args"][0]
    # The dispatcher's module form: this interpreter, -m hermes_cli.main, --version.
    assert argv[:3] == [sys.executable, "-m", "hermes_cli.main"]
    assert argv[-1] == "--version"
    # Must probe from a foreign cwd (a temp dir), never the checkout cwd.
    assert calls["kwargs"]["cwd"] is not None and calls["kwargs"]["cwd"] != str(Path.cwd())
    # The child gets the propagation result as its environment, not None.
    assert isinstance(calls["kwargs"]["env"], dict)
    assert calls["kwargs"]["env"].get("PATH") == os.environ.get("PATH")
    assert "via the dispatcher's import pin" in capsys.readouterr().out


def test_foreign_cwd_failure_reports_issue(monkeypatch, capsys):
    fake_run, _calls = _fake_run_factory(
        1, "", "Traceback (most recent call last):\nModuleNotFoundError: No module named 'hermes_cli'\n")
    monkeypatch.setattr(doctor_state.subprocess, "run", fake_run)

    finding = doctor_state._check_dispatch_runtime_import(False)

    assert finding.issues, "failure must append a repair issue"
    assert "cannot import hermes_cli via the dispatcher's import pin" in capsys.readouterr().out


def test_launch_failure_reports_issue(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr(doctor_state.subprocess, "run", boom)

    finding = doctor_state._check_dispatch_runtime_import(False)

    assert finding.issues, "a probe that cannot launch must append a repair issue"
    assert "probe failed to launch" in capsys.readouterr().out
