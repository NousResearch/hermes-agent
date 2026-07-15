"""Regression tests for the Windows update import boundary.

`hermes update` mutates the same venv it runs from. On Windows, importing
native-extension packages before dependency sync can keep their `.pyd` files
locked and leave the install half-updated. The update import path therefore
must stay PyYAML-free until the install step has completed.
"""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import main as cli_main


def test_minimal_pause_stops_active_gateway_without_yaml_import(monkeypatch):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            return ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"]

        def ppid(self):
            return 1

        def kill(self):
            raise AssertionError("taskkill should handle the gateway")

    fake_psutil = SimpleNamespace(Process=FakeProcess)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(
        cli_main,
        "_detect_venv_python_processes",
        lambda: [
            (101, "pythonw.exe", "pythonw.exe -m hermes_cli.main gateway run")
        ],
    )
    calls = []
    monkeypatch.setattr(
        cli_main.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv)
        or SimpleNamespace(returncode=0),
    )

    real_import = builtins.__import__

    def block_yaml(name, *args, **kwargs):
        if name == "yaml" or name.startswith("yaml."):
            raise AssertionError(f"unexpected yaml import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_yaml)
    token = cli_main._pause_windows_gateways_for_update_minimal()

    assert calls == [["taskkill", "/PID", "101", "/T", "/F"]]
    assert token == {
        "resume_needed": True,
        "profiles": {},
        "unmapped_pids": [101],
        "unmapped": [
            {
                "pid": 101,
                "argv": [
                    "pythonw.exe",
                    "-m",
                    "hermes_cli.main",
                    "gateway",
                    "run",
                ],
            }
        ],
    }


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file-lock regression")
def test_windows_update_import_does_not_import_yaml():
    repo = Path(__file__).resolve().parents[2]
    code = r"""
import importlib.abc
import sys
from types import SimpleNamespace

sys.argv = ["hermes", "update", "--yes"]


class BlockYaml(importlib.abc.MetaPathFinder):
    seen = 0

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "yaml" or fullname.startswith("yaml."):
            self.seen += 1
            raise AssertionError(f"unexpected yaml import: {fullname}")
        return None


blocker = BlockYaml()
sys.meta_path.insert(0, blocker)

import hermes_cli.main as main

assert main._windows_update_import_minimal()
assert blocker.seen == 0

# The real pause seam must remain dependency-light on this path too.
main._detect_venv_python_processes = lambda: []
assert main._pause_windows_gateways_for_update() is None
assert blocker.seen == 0

main._install_hangup_protection = lambda gateway_mode=False: {}
main._finalize_update_output = lambda state: None
main._cmd_update_impl = lambda args, gateway_mode: None
main.cmd_update(SimpleNamespace(check=False, gateway=False))
assert blocker.seen == 0
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
