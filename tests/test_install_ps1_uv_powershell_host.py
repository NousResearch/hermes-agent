"""Native checks for the PowerShell host used by the uv installer."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


@pytest.mark.windows_only
def test_powershell_host_resolution_is_path_independent(tmp_path):
    executable = shutil.which("powershell") or shutil.which("pwsh")
    assert executable, "native PowerShell is required for this Windows-only test"
    harness = tmp_path / "host-harness.ps1"
    harness.write_text(
        f". '{_INSTALL_PS1}'\n"
        "$env:Path = ''\n"
        "$resolved = Get-PowerShellHostExe\n"
        "$result = @{\n"
        "  exists = Test-Path -LiteralPath $resolved\n"
        "  leaf = Split-Path -Leaf $resolved\n"
        "}\n"
        "Write-Output ($result | ConvertTo-Json -Compress)\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.setdefault("SystemDrive", Path(executable).drive or "C:")
    completed = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout.strip())

    assert result["exists"] is True
    assert result["leaf"].lower() in {"powershell.exe", "pwsh.exe"}
