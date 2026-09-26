"""Pinned git staging must not need a bzip2-capable tar (#122512).

The reported Windows 10 box has no bzip2.exe on PATH, and its System32
tar.exe cannot run the bzip2 filter
("tar.exe: Error opening archive: Can't initialize filter; unable to run
program \"bzip2 -d\""), so extracting the pinned .tar.bz2 fails and
bootstrap dies at stage=prerequisites with "failed to extract pinned git
archive".

The driver below reproduces that machine deterministically: PATH carries
no bzip2, and the installer's own Invoke-Native seam resolves $inboxTar
to a stub System32 tar.exe that only prints the reported error. Then it
runs the real Get-PinnedGit against the real pinned archive and demands
a working git.exe plus the bundled bash contract pm/shell.py relies on.
"""
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts" / "install.ps1"
CSC = (
    Path(os.environ.get("SystemRoot", r"C:\Windows"))
    / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe"
)


@pytest.mark.platforms("windows")
def test_pinned_git_extracts_without_a_bzip2_capable_tar(tmp_path):
    assert CSC.is_file(), f"missing C# compiler: {CSC}"

    # The reporter's System32 tar.exe: it exists, but has no bzip2 filter.
    stub_src = tmp_path / "bzip2less_tar.cs"
    stub_src.write_text(
        "using System; class Stub { static int Main(string[] a) { "
        'System.Console.Error.WriteLine("tar.exe: Error opening archive: '
        "Can't initialize filter; unable to run program \\\"bzip2 -d\\\"\"); "
        "return 1; } }",
        encoding="ascii",
    )
    fake_root = tmp_path / "sysroot"
    stub_tar = fake_root / "System32" / "tar.exe"
    stub_tar.parent.mkdir(parents=True)
    compiled = subprocess.run(
        [str(CSC), "/nologo", f"/out:{stub_tar}", str(stub_src)],
        capture_output=True, text=True, timeout=120,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    driver = tmp_path / "driver.ps1"
    driver.write_text(
        textwrap.dedent(f"""
            . '{INSTALLER}' -HermesHome '{tmp_path / "home"}'
            $env:HERMES_RUNTIME_DIR = '{tmp_path / "tools"}'
            # This machine has no bzip2 (and nothing else to fall back on).
            $env:PATH = "$env:SystemRoot\\System32;$env:SystemRoot"
            function Invoke-Native {{
                param([scriptblock]$Command)
                # Dynamic scope: bind the installer's $inboxTar to the
                # bzip2-less stub before running its extraction command.
                $inboxTar = '{stub_tar}'
                & $Command
            }}
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
    )
    out = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert result.returncode == 0, out
