"""Regression tests for #96112: installer-spawned npm pack must see tsc.cmd.

TypeScript is already on disk after workspace npm install; the failure is the
environment the bootstrap installer hands to npm/cmd.exe (duplicate Path/PATH
keys, missing PATHEXT, node_modules\\.bin not on PATH).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_install_ps1_defines_npm_lifecycle_path_helper() -> None:
    text = _install_ps1()
    assert "function Ensure-NpmLifecyclePath" in text
    assert "function Test-DesktopBuildIsMissingTsc" in text
    assert re.search(
        r'\[Environment\]::SetEnvironmentVariable\(\s*"Path"',
        text,
    ), "Ensure-NpmLifecyclePath must write the Process Path key"
    assert ".CMD" in text, "PATHEXT default must include .CMD so tsc.cmd resolves"


def test_install_desktop_unifies_path_before_npm_pack() -> None:
    text = _install_ps1()
    assert re.search(
        r"function Install-Desktop \{[\s\S]*?Ensure-NpmLifecyclePath[\s\S]*?\$npmExe run pack",
        text,
    ), "Install-Desktop must unify Path before npm run pack"


def test_install_desktop_does_not_blame_electron_for_missing_tsc() -> None:
    text = _install_ps1()
    assert re.search(
        r"Test-DesktopBuildIsMissingTsc[\s\S]{0,400}ELECTRON_MIRROR",
        text,
    ), "tsc-not-on-PATH failures must skip the Electron-mirror retry"
