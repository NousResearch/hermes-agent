"""Install-Desktop must pin npm's Win32 cwd to the Hermes checkout.

Launching Hermes-Setup.exe from another project (e.g. D:\\Work) leaves the
process cwd there. Push-Location updates $PWD, but npm.cmd inherits
[Environment]::CurrentDirectory, so it reads the wrong package.json:

  No workspaces found: --workspace=ui-tui --workspace=web
  Missing script: "pack"

The installer then misdiagnoses that as a blocked Electron download and
retries npmmirror (#46785).
"""

from __future__ import annotations

import re
from pathlib import Path

INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


def _source() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    match = re.search(
        rf"^function {re.escape(name)} \{{.*?^\}}", source, re.MULTILINE | re.DOTALL
    )
    assert match, f"could not extract function {name} from install.ps1"
    return match.group(0)


def test_install_desktop_syncs_win32_cwd_before_npm() -> None:
    """npm.cmd children must see the checkout directory, not the launch cwd."""
    source = _source()
    sync = _function_body(source, "Sync-Win32ProcessCwd")
    assert "[System.IO.Directory]::SetCurrentDirectory" in sync

    desktop = _function_body(source, "Install-Desktop")
    push_then_sync = re.search(
        r"Push-Location \$InstallDir\s+.*?Sync-Win32ProcessCwd",
        desktop,
        re.DOTALL,
    )
    assert push_then_sync, "npm ci must run after Sync-Win32ProcessCwd at $InstallDir"
    pack_sync = re.search(
        r"Push-Location \$desktopDir\s+.*?Sync-Win32ProcessCwd",
        desktop,
        re.DOTALL,
    )
    assert pack_sync, "npm run pack must run after Sync-Win32ProcessCwd at $desktopDir"
    ci_idx = desktop.find("Sync-Win32ProcessCwd")
    pack_idx = desktop.find("& $npmExe run pack")
    assert ci_idx != -1 and pack_idx != -1 and ci_idx < pack_idx


def test_install_desktop_skips_electron_mirror_on_wrong_cwd() -> None:
    """A missing pack script is the wrong package.json, not GitHub blocking Electron."""
    source = _source()
    detector = _function_body(source, "Test-DesktopNpmFailureIsWrongCwd")
    assert "Missing script" in detector
    assert "No workspaces found" in detector

    desktop = _function_body(source, "Install-Desktop")
    assert "Test-DesktopNpmFailureIsWrongCwd" in desktop
    skip_gate = re.search(
        r"if \(Test-DesktopNpmFailureIsWrongCwd \$errSoFar\) \{[\s\S]{0,400}?"
        r"Skipping Electron redownload",
        desktop,
    )
    assert skip_gate, "wrong-cwd npm output must skip the npmmirror retry"
