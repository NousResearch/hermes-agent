"""Permission failures in install state must explain the actual access problem."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from pm.environments import install_state_dir, install_state_permission_message


@pytest.mark.parametrize("phase", ["preparation", "activation"])
def test_bootstrap_reports_unwritable_install_once(tmp_path, monkeypatch, phase):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = Path(__file__).resolve().parents[2]
    target = install_state_dir(root) / (
        "pm-runtime/.prepare.lock" if phase == "preparation" else ".install.lock"
    )
    code = """
import errno
import sys
import pm.environments as environments
target, phase = sys.argv[1:3]
from hermes_cli import _early_recovery, venv_sync

def denied(*args, **kwargs):
    raise PermissionError(errno.EACCES, "Permission denied", target)

venv_sync.prepare_launch = denied if phase == "preparation" else lambda *_: None
_early_recovery.recover_if_needed = lambda *_: False
environments.activate_dependencies = denied if phase == "activation" else lambda *_: None
sys.argv = ["hermes", "-z", "hi"]
import hermes_bootstrap
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(target), phase],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, result.stderr
    assert result.stderr.count("hermes: ") == 1
    assert "install state is not writable by this user" in result.stderr
    assert str(target) in result.stderr
    assert "source-update completion failed" not in result.stderr
    assert "run `hermes update`" not in result.stderr
    assert "hermes pm repair" not in result.stderr
    assert "Traceback" not in result.stderr


def test_pm_repair_reports_unwritable_runtime_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = Path(__file__).resolve().parents[2]
    target = install_state_dir(root) / "pm-runtime" / ".prepare.lock"
    code = """
import errno
import sys
from pm import runtime

runtime.is_runtime = lambda: False
def denied(*args):
    raise PermissionError(errno.EACCES, "Permission denied", sys.argv[1])
runtime.run_cli = denied
from pm.cli import main
raise SystemExit(main(["repair"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(target)],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, result.stderr
    assert "install state is not writable by this user" in result.stderr
    assert str(target) in result.stderr
    assert "Traceback" not in result.stderr
    lease = install_state_dir(root) / "environments" / "generation" / ".leases" / "reader"
    assert install_state_permission_message(root, PermissionError(13, "Permission denied", str(lease)))
    outside = tmp_path / "other" / ".prepare.lock"
    exc = PermissionError(13, "Permission denied", str(outside))
    assert install_state_permission_message(root, exc) is None

    from hermes_cli import _early_recovery

    project = tmp_path / "source"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    denied_path = install_state_dir(project) / ".install.lock"
    recovery = ModuleType("pm.recovery")

    def fail_repair(*_):
        raise PermissionError(13, "Permission denied", str(denied_path))

    recovery.repair_dependencies = fail_repair
    monkeypatch.setitem(sys.modules, "pm.recovery", recovery)
    with pytest.raises(PermissionError):
        _early_recovery.recover_if_needed(project, argv=["pm", "repair"], explicit=True)
    assert "run `hermes pm repair`" not in capsys.readouterr().err
