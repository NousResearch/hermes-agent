"""Packaged-shell pre-gate for the Desktop updater (#118070).

A system-package shell (AppImage/deb/rpm) lives outside the checkout, so
``hermes update`` can replace the backend but never the running GUI. The
post-update skew warning fires only AFTER the backend is already replaced,
stranding Desktop on a newer backend the old shell cannot answer
(approvals/clarify go dark). The updater must refuse BEFORE touching the
backend and keep the compatible one.

Drives the real orchestrator (like ``repro.sh launch``) against a fake
install whose ``hermes`` stub records every invocation.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

POSIX_SH = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "desktop-update" / "posix.sh"

pytestmark = pytest.mark.platforms("linux")


def _stub_install(root: Path, home: Path) -> Path:
    """Fake legacy install whose ``hermes`` stub logs invocations, exits 0."""
    bin_dir = root / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    marker = home / "hermes-invoked.log"
    stub = bin_dir / "hermes"
    stub.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{marker}"\n'
        "exit 0\n"
    )
    stub.chmod(0o755)
    return marker


def _run_update(root: Path, home: Path, target: str) -> dict:
    env = dict(os.environ, HERMES_HOME=str(home))
    subprocess.run(
        ["bash", str(POSIX_SH), "--daemonized",
         "--no-ui", "--no-marker-cleanup", "--desktop-pid", "0",
         "--install-root", str(root), "--relaunch-target", target],
        capture_output=True, text=True, check=False, env=env, timeout=120,
    )
    return json.loads((home / ".hermes-update-result.json").read_text())


def test_packaged_shell_refuses_before_touching_backend(tmp_path):
    root = tmp_path / "hermes-agent"
    home = tmp_path / "home"
    home.mkdir()
    marker = _stub_install(root, home)
    pkg = tmp_path / "opt" / "Hermes"
    pkg.mkdir(parents=True)
    (pkg / "hermes").touch()

    result = _run_update(root, home, str(pkg / "hermes"))

    assert not marker.exists(), "backend update ran under a packaged shell"
    assert result["ok"] is True and result["manual"] is True
    assert "was not changed" in result["message"]
    assert "Nothing was changed" in result["message"]
    assert "apt upgrade hermes-desktop" in result["message"], "per-manager repair hint"


def test_checkout_run_still_updates(tmp_path):
    """Dev/checkout runs (target inside INSTALL_ROOT) keep today's flow."""
    root = tmp_path / "hermes-agent"
    home = tmp_path / "home"
    home.mkdir()
    marker = _stub_install(root, home)
    dev_target = root / "node_modules" / "electron" / "dist" / "electron"
    dev_target.parent.mkdir(parents=True)
    dev_target.touch()

    result = _run_update(root, home, str(dev_target))

    assert marker.exists(), "checkout-run backend update must still proceed"
    assert result["manual"] is True
    assert "was not changed" in result["message"]
