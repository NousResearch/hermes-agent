"""Regression tests for the Windows update import boundary.

`hermes update` mutates the same venv it runs from. On Windows, importing
native-extension packages before dependency sync can keep their `.pyd` files
locked and leave the install half-updated. The update import path therefore
must stay PyYAML-free until the install step has completed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


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
