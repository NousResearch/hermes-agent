"""Regression tests for #48130: Windows npm lifecycle scripts need node on PATH.

The desktop installer can resolve ``npm.cmd`` while postinstall hooks fail with
``'node' is not recognized`` because child ``cmd.exe`` processes do not inherit
a PATH that includes ``node.exe``'s directory.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_install_ps1_defines_ensure_node_exe_on_path_helper() -> None:
    text = _install_ps1()
    assert "function Ensure-NodeExeOnPath" in text
    assert re.search(
        r"\$env:Path\s*=\s*\"\$nodeExeDir;\$env:Path\"",
        text,
    ), "Ensure-NodeExeOnPath must prepend node.exe's directory to PATH"


def test_test_node_prepends_node_dir_before_success() -> None:
    text = _install_ps1()
    assert re.search(
        r"if \(Test-NodeVersionOk \$version\) \{[\s\S]{0,200}?Ensure-NodeExeOnPath",
        text,
    ), "Test-Node must call Ensure-NodeExeOnPath when a system Node passes the version floor"


def test_install_node_deps_prepends_node_dir_before_npm() -> None:
    text = _install_ps1()
    assert re.search(
        r"function Install-NodeDeps \{[\s\S]{0,900}?Ensure-NodeExeOnPath[\s\S]{0,900}?Resolve npm explicitly",
        text,
    ), "Install-NodeDeps must call Ensure-NodeExeOnPath before invoking npm"
