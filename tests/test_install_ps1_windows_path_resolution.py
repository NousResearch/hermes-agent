"""Regression: Windows installer must prefer the fresh managed hermes shim.

Issue #54919 reported every ``hermes`` command failing immediately with uv's
"trampoline failed to spawn Python child process" launcher error on native
Windows. The installer already verified that ``venv\\Scripts\\hermes.exe``
existed, but it only prepended the path if it was absent and never removed
stale Hermes-owned entries or checked which launcher ``hermes`` actually
resolved to afterwards.

These source-level tests lock the installer contract:

1. rebuild PATH so the current managed install wins over stale launchers under
   the same install root; and
2. verify which ``hermes`` shim resolves after the rewrite, surfacing a clear
   warning if another launcher still wins.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_windows_installer_rebuilds_path_instead_of_append_if_missing() -> None:
    text = _install_ps1()
    assert "function Merge-HermesPathEntries" in text, (
        "install.ps1 must rebuild PATH entries so the current managed Hermes "
        "launcher wins over stale shims"
    )
    assert '$currentPath -notlike "*$hermesBin*"' not in text, (
        "install.ps1 must not use the naive 'append if missing' PATH guard; "
        "it leaves stale Hermes launchers ahead of the fresh install"
    )
    assert (
        "StartsWith($installRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)"
        in text
    ), (
        "PATH rebuild must drop stale entries under the managed install root "
        "using a case-insensitive path-prefix check"
    )


def test_windows_installer_rewrites_both_user_and_session_path() -> None:
    text = _install_ps1()
    assert (
        "$newUserPath = Merge-HermesPathEntries $currentPath $hermesBin $InstallDir"
        in text
    )
    assert (
        "$env:Path = Merge-HermesPathEntries $env:Path $hermesBin $InstallDir" in text
    ), (
        "install.ps1 must rewrite the current session PATH too, not just the "
        "registry value for future shells"
    )


def test_windows_installer_verifies_resolved_hermes_launcher() -> None:
    text = _install_ps1()
    assert "Get-Command hermes" in text, (
        "install.ps1 must verify which hermes launcher resolves after PATH setup"
    )
    assert "$expectedHermesExe" in text
    assert "hermes currently resolves to a different launcher" in text, (
        "installer should warn explicitly when another hermes shim still wins "
        "after PATH normalization"
    )
