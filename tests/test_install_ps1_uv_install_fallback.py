"""Native Windows coverage for the verified uv installer fallbacks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"
_EXPECTED_URLS = [
    "https://astral.sh/uv/0.11.6/install.ps1",
    "https://github.com/astral-sh/uv/releases/download/0.11.6/uv-installer.ps1",
]


def _powershell() -> str:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    assert executable, "native PowerShell is required for this Windows-only test"
    return executable


def _run_install_uv_harness(tmp_path: Path, *, digest_matches: bool) -> dict:
    hermes_home = tmp_path / "hermes-home"
    user_profile = tmp_path / "user-profile"
    sentinel = tmp_path / "payload-executed.txt"
    payload = tmp_path / "payload.ps1"
    payload.write_text("# fake downloaded installer\n", encoding="utf-8")

    reported_hash = (
        "46da9313591884d09aa4f06f7f78f74154ea01a8012d425ed090163d4799295c"
        if digest_matches
        else "0" * 64
    )
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        "param([string]$Installer)\n"
        f"$HarnessHermesHome = '{hermes_home}'\n"
        "$env:HERMES_HOME = $HarnessHermesHome\n"
        f"$env:USERPROFILE = '{user_profile}'\n"
        f". '{_INSTALL_PS1}' -HermesHome $HarnessHermesHome -InstallDir (Join-Path $HarnessHermesHome 'hermes-agent')\n"
        "$HermesHome = $HarnessHermesHome\n"
        "$script:AttemptedUrls = @()\n"
        "function global:Invoke-WebRequest {\n"
        "  param([switch]$UseBasicParsing, $Uri, $OutFile, $ErrorAction)\n"
        "  $script:AttemptedUrls += [string]$Uri\n"
        "  Copy-Item -LiteralPath $Installer -Destination $OutFile -Force\n"
        "}\n"
        "function global:Get-FakeFileHash {\n"
        "  param($Algorithm, $Path)\n"
        f"  [pscustomobject]@{{ Hash = '{reported_hash}' }}\n"
        "}\n"
        "Set-Alias -Name Get-FileHash -Value Get-FakeFileHash -Scope Global\n"
        "function global:Get-Command { param($Name) if ($Name -eq 'uv') { return $null } Microsoft.PowerShell.Core\\Get-Command @PSBoundParameters }\n"
        "function global:Get-PowerShellHostExe { return 'Invoke-FakePowerShell' }\n"
        "function global:Invoke-FakePowerShell {\n"
        f"  Add-Content -LiteralPath '{sentinel}' -Value executed\n"
        "  $target = Join-Path $env:UV_INSTALL_DIR 'uv.exe'\n"
        "  Set-Content -LiteralPath $target -Value fake-uv\n"
        "}\n"
        "$ok = Install-Uv\n"
        "$result = @{\n"
        "  attempted_urls = @($script:AttemptedUrls)\n"
        f"  payload_executed = Test-Path -LiteralPath '{sentinel}'\n"
        f"  execution_count = @((Get-Content -LiteralPath '{sentinel}' -ErrorAction SilentlyContinue)).Count\n"
        "  install_succeeded = [bool]$ok\n"
        "}\n"
        "Write-Output ('HARNESS:' + ($result | ConvertTo-Json -Compress))\n",
        encoding="utf-8",
    )

    executable = _powershell()
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
            "-Installer",
            str(payload),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    marker = next(
        line.removeprefix("HARNESS:")
        for line in completed.stdout.splitlines()
        if line.startswith("HARNESS:")
    )
    return json.loads(marker)


@pytest.mark.windows_only
def test_checksum_mismatch_attempts_both_sources_without_execution(tmp_path):
    result = _run_install_uv_harness(tmp_path, digest_matches=False)

    assert result["attempted_urls"] == _EXPECTED_URLS
    assert result["payload_executed"] is False
    assert result["execution_count"] == 0
    assert result["install_succeeded"] is False


@pytest.mark.windows_only
def test_matching_checksum_executes_only_downloaded_installer(tmp_path):
    result = _run_install_uv_harness(tmp_path, digest_matches=True)

    assert result["payload_executed"] is True
    assert result["attempted_urls"] == _EXPECTED_URLS[:1]
    assert result["execution_count"] == 1
    assert result["install_succeeded"] is False
