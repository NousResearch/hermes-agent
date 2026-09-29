"""A denied store mutation names the operation, path, and remediation (#124052).

A non-admin run whose earlier bootstrap ran elevated hits admin-owned store files
and fails with the raw localized Win32 message ("Odmowa dostepu" / "Access denied"):
no path, no operation — undiagnosable. install.ps1's store mutations rethrow naming
all three so the stage frame (and the user) can act instead of reinstalling blind.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"


def _dot_sourced(body: str) -> subprocess.CompletedProcess:
    script = f'$ErrorActionPreference = "Stop"; . "{INSTALLER}"; {body}'
    return subprocess.run(
        [shutil.which("powershell"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, timeout=120)


def _deny(denied: Path) -> None:
    """A self-deny ACE: an owner can set one without elevation, reproducing the
    access-denied class (Win32 error 5) deterministically on the test host."""
    subprocess.run(
        ["icacls", str(denied), "/deny", f"{os.environ['USERNAME']}:(OI)(CI)(F)"],
        capture_output=True, text=True, check=True)


def _allow(denied: Path) -> None:
    subprocess.run(
        ["icacls", str(denied), "/remove", os.environ["USERNAME"]],
        capture_output=True, text=True)


def test_denied_store_mutation_names_operation_path_and_remediation(tmp_path):
    denied = tmp_path / "store" / "tools" / "git-entry"
    denied.mkdir(parents=True)
    _deny(denied)

    try:
        result = _dot_sourced(
            f'$entry = "{denied}"; '
            'try { Invoke-StoreMutation "replace" $entry { Remove-Item -Recurse -Force $entry } } '
            'catch { "caught: $_" }')
    finally:
        _allow(denied)  # restore access so tmp_path cleanup can remove the tree

    assert result.returncode == 0, result.stderr
    assert f'replace "{denied}" failed (UnauthorizedAccessException)' in result.stdout
    assert f'icacls "{denied}"' in result.stdout
    assert "caught: cannot replace" in result.stdout


def test_allowed_store_mutation_prints_no_diagnostics(tmp_path):
    entry = tmp_path / "store" / "tools" / "git-entry"
    entry.mkdir(parents=True)

    result = _dot_sourced(
        f'$entry = "{entry}"; '
        'Invoke-StoreMutation "replace" $entry { Remove-Item -Recurse -Force $entry }; '
        '"removed=$(-not (Test-Path $entry))"')

    assert result.returncode == 0, result.stderr
    assert "removed=True" in result.stdout
    assert "failed" not in result.stdout
