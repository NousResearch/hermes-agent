"""Regression for #125350: -SkipSetup must keep binding after the staged-installer rework.

The staged-installer rework (92686159d1) dropped the ``-SkipSetup`` switch from
install.ps1's ``param()`` block, so wrappers written against the old spelling
(hermes-desktop passes ``-SkipSetup -NonInteractive -HermesHome ...``) die at
parameter binding with ``NamedParameterNotFound`` before any stage runs. The
switch is back as a deprecated alias that folds into ``-NonInteractive``.

``-ShowResolvedPaths`` is the installer's side-effect-free contract: it still
exercises real parameter binding and the whole prologue, then prints the
resolved-path report and exits 0.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER = REPO_ROOT / "scripts" / "install.ps1"


def test_skipsetup_still_binds(tmp_path):
    powershell = shutil.which("powershell")
    assert powershell
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(INSTALLER), "-SkipSetup", "-NonInteractive", "-ShowResolvedPaths",
         "-HermesHome", str(tmp_path / "home"), "-InstallDir", str(tmp_path / "install")],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        "-SkipSetup failed to bind:\n" + result.stderr + result.stdout)
    report = json.loads(result.stdout)
    assert report, "expected the resolved-paths JSON report on stdout"
