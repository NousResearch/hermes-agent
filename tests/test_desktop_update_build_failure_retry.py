"""The hand-off must recognise every Desktop build-failure message (#107685).

``hermes update`` calls a Desktop build failure non-fatal and exits 0, so the
Desktop-driven hand-off re-reads its output to decide whether to spend its one
rebuild retry. It matched the literal "Desktop build failed" only, which the
packaged-app stage's "Desktop GUI build failed" never contains -- so the retry
could not fire for the failure class that actually needs it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DESKTOP_UPDATE = REPO_ROOT / "scripts" / "desktop-update"
RETRY_POLICY = DESKTOP_UPDATE / "retry-policy.ps1"
WINDOWS_PS1 = DESKTOP_UPDATE / "windows.ps1"
POSIX_SH = DESKTOP_UPDATE / "posix.sh"


@pytest.mark.windows_only
def test_policy_matches_both_build_failure_shapes() -> None:
    policy = str(RETRY_POLICY).replace("'", "''")
    command = f"""
        . '{policy}'
        $gui = [char]0x2717 + ' Desktop GUI build failed'
        $deps = '  ' + [char]0x26A0 + ' Desktop build failed (run hermes desktop to retry)'
        @{{
            gui = [bool](Test-HermesDesktopBuildFailed -Output $gui)
            deps = [bool](Test-HermesDesktopBuildFailed -Output $deps)
            clean = [bool](Test-HermesDesktopBuildFailed -Output 'Desktop packaged app ready')
            empty = [bool](Test-HermesDesktopBuildFailed -Output '')
        }} | ConvertTo-Json -Compress
    """
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout) == {"gui": True, "deps": True, "clean": False, "empty": False}


def test_windows_handoff_reads_the_detector_from_the_updated_checkout() -> None:
    text = WINDOWS_PS1.read_text(encoding="utf-8")

    # The in-flight script cannot fix its own matching rules, but the companion
    # policy is dot-sourced from the checkout the update just wrote.
    assert "Test-HermesDesktopBuildFailed -Output $res.Output" in text
    assert 'Desktop( GUI)? build failed' in text
    assert '-match "Desktop build failed"' not in text


def test_posix_handoff_matches_the_gui_build_failure() -> None:
    text = POSIX_SH.read_text(encoding="utf-8")

    assert 'grep -qE "Desktop( GUI)? build failed"' in text
    assert 'grep -q "Desktop build failed"' not in text


def test_verification_failure_does_not_advise_repair_or_antivirus() -> None:
    messages = [
        line
        for line in WINDOWS_PS1.read_text(encoding="utf-8").splitlines()
        if "$finalMsg" in line and "verif" in line.lower()
    ]

    assert messages
    for line in messages:
        # "Repair the installation and review antivirus quarantine" is destructive
        # advice for a condition that means "the receipt could not read the app".
        assert "antivirus" not in line.lower()
        assert "repair the installation" not in line.lower()
