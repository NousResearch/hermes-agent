"""install.ps1's self-contained PowerShell suites, under Windows PowerShell 5.1 and pwsh 7.

``scripts/tests/*.ps1`` drive the real installer as a subprocess and exit
non-zero on any failed assertion. install.ps1 is delivered via ``irm | iex``
into whatever shell the user already has, and 5.1 is what ships with Windows,
so each suite runs under both: a construct that only parses under 7 is a broken
installer for most of the people hitting it.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("windows")
SUITES = Path(__file__).resolve().parents[3] / "scripts" / "tests"


def _shell(name: str) -> str:
    if name == "powershell":
        return str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    return shutil.which("pwsh") or pytest.fail("native lane requires PowerShell 7")


@pytest.mark.parametrize("shell", ["powershell", "pwsh"])
@pytest.mark.parametrize("suite", sorted(p.name for p in SUITES.glob("test-install-ps1-*.ps1")))
def test_suite_passes(shell, suite):
    result = subprocess.run([_shell(shell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-File", str(SUITES / suite)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
