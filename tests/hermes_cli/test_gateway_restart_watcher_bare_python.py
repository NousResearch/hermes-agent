"""The detached gateway restart watcher must survive on an interpreter without the dependency env.

``hermes update`` finishes on the bare store Python (dependencies come from ``hermes_bootstrap``,
not site-packages) and spawns the watcher as ``sys.executable -c <watcher>``. A watcher that imports
a third-party-backed Hermes module (``gateway.status`` -> ``utils`` -> ``ruamel``) dies before it
relaunches the gateway, leaving a manually started gateway down after every update.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import gateway

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell wrapper stands in for the bare interpreter")


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_restart_watcher_relaunches_from_an_interpreter_without_site_packages(tmp_path, monkeypatch):
    # A python that sees the checkout but no third-party packages, like the bare store Python.
    bare = tmp_path / "bare-python"
    bare.write_text(f'#!/bin/sh\nunset PYTHONPATH\nexec "{sys.executable}" -S "$@"\n', encoding="utf-8")
    bare.chmod(0o755)
    probe = subprocess.run([str(bare), "-c", "import ruamel.yaml"], cwd=REPO, capture_output=True, text=True)
    assert probe.returncode != 0, "premise: the stand-in interpreter must not see site-packages"

    marker = tmp_path / "respawned"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "executable", str(bare))
    relaunch = [str(bare), "-c", f"open({str(marker)!r}, 'w').write('ok')"]

    assert gateway._spawn_gateway_restart_watcher(_dead_pid(), relaunch, host=False)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.1)
    stdio = home / "logs" / "gateway-stdio.log"
    assert marker.exists(), (
        "the restart watcher never relaunched the gateway command"
        + (f"; stdio log:\n{stdio.read_text(errors='replace')}" if stdio.exists() else ""))
    assert os.path.getsize(marker) == 2
