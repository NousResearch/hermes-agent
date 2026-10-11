"""Pinned git staging must not need a bzip2-capable tar (#122512, #122774).

Stock Windows 10 ships a System32 tar.exe without a bzip2 filter, so the
old .tar.bz2 pin died at stage=prerequisites ("tar.exe: Error opening
archive: Can't initialize filter; unable to run program \"bzip2 -d\"").
Get-PinnedGit now stages the PortableGit self-extractor, which needs
neither tar nor bzip2. The driver below runs the real Get-PinnedGit
against the real pinned archive with a PATH that carries no bzip2 (or
anything else to fall back on) and demands a working git.exe plus the
bundled bash contract pm/shell.py relies on.
"""
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts" / "install.ps1"


@pytest.mark.platforms("windows")
def test_pinned_git_extracts_without_a_bzip2_capable_tar(tmp_path):
    driver = tmp_path / "driver.ps1"
    driver.write_text(
        textwrap.dedent(f"""
            . '{INSTALLER}' -HermesHome '{tmp_path / "home"}'
            $env:HERMES_RUNTIME_DIR = '{tmp_path / "tools"}'
            # This machine has no bzip2 (and nothing else to fall back on).
            $env:PATH = "$env:SystemRoot\\System32;$env:SystemRoot"
            $git = Get-PinnedGit
            if (-not $git) {{ exit 9 }}
            & $git --version 2>$null | Out-Null
            if ($LASTEXITCODE) {{ exit 9 }}
            $entry = Split-Path (Split-Path $git)
            if (-not (Test-Path (Join-Path $entry 'usr\\bin\\bash.exe'))) {{ exit 9 }}
        """),
        encoding="ascii",
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(driver)],
        capture_output=True, timeout=900,
        check=False,
    )
    out = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert result.returncode == 0, out
