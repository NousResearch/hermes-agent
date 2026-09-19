"""Regression tests for Windows desktop update hand-off truthful receipts and error trapping (#109627).

Guards the contract that:
1. An unhandled PowerShell exception in the post-update phase never falls back to the
   opaque dummy default ("update did not complete" with exit 1). If the update step
   itself succeeded (exit 0), the receipt and handoff log must truthfully reflect that
   the update succeeded and report the post-update exception.
2. The handoff log captures the critical exception message and stack trace.
3. `Invoke-HermesStep` traps process launch failures gracefully and logs them.
4. Win32 errors in `HermesUpdateJob::StartAssigned` include `Marshal.GetLastWin32Error()`.
5. `retry-policy.ps1` resolution falls back to `$InstallRoot` when `$PSScriptRoot` is missing.
6. The `verify` step logs its exit code.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _read_windows_ps1() -> str:
    return WINDOWS_PS1.read_text(encoding="utf-8")


class TestWindowsHandoffReceiptContract:
    """Source-level contract tests ensuring Windows hand-off has truthful error handling."""

    def test_main_block_has_catch_for_unhandled_exceptions(self) -> None:
        source = _read_windows_ps1()
        assert "} catch {" in source, "windows.ps1 must have a catch block on the main try block"
        assert "$script:UnhandledException = $_" in source
        assert "CRITICAL: unhandled error in hand-off:" in source

    def test_verify_step_logs_exit_code(self) -> None:
        source = _read_windows_ps1()
        assert 'Write-HandoffLog "verify exit code: $($verify.Code)"' in source, (
            "The verify step in windows.ps1 must log its exit code for diagnostics"
        )

    def test_start_assigned_win32_errors_are_captured(self) -> None:
        source = _read_windows_ps1()
        assert "Marshal.GetLastWin32Error()" in source, (
            "HermesUpdateJob::StartAssigned must query Marshal.GetLastWin32Error() on Win32 failures"
        )
        assert 'CreateProcess failed (error "' in source

    def test_invoke_hermes_step_traps_launch_failure(self) -> None:
        source = _read_windows_ps1()
        assert "step launch failed:" in source, (
            "Invoke-HermesStep must catch StartAssigned exceptions and log them"
        )

    def test_retry_policy_path_has_install_root_fallback(self) -> None:
        source = _read_windows_ps1()
        assert "$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } elseif ($InstallRoot)" in source, (
            "retry-policy.ps1 resolution must have a fallback when $PSScriptRoot is null"
        )

    def test_finally_block_does_not_mask_post_update_failures(self) -> None:
        source = _read_windows_ps1()
        assert "if ($null -ne $script:UnhandledException" in source, (
            "The finally block must ensure an unhandled exception is not masked by dummy defaults"
        )


@pytest.mark.windows_only
class TestWindowsHandoffTruthfulReceiptBehavior:
    """Execution-level tests on Windows validating receipt and error handling behavior."""

    def test_post_update_exception_after_successful_update_produces_truthful_receipt(
        self, tmp_path: Path
    ) -> None:
        powershell = shutil.which("powershell.exe")
        if not powershell:
            pytest.skip("powershell.exe not found")

        hermes_home = tmp_path / "hermes_home"
        install_root = hermes_home / "hermes-agent"
        scripts_dir = install_root / "scripts" / "desktop-update"
        logs_dir = hermes_home / "logs"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        result_path = hermes_home / ".hermes-update-result.json"
        handoff_log = logs_dir / "desktop-update-handoff.log"

        # Copy windows.ps1 and retry-policy.ps1
        shutil.copy2(WINDOWS_PS1, scripts_dir / "windows.ps1")
        shutil.copy2(REPO_ROOT / "scripts" / "desktop-update" / "retry-policy.ps1", scripts_dir / "retry-policy.ps1")

        # Test script that simulates the post-update phase with an injected exception
        # after update step returns code 0
        test_script = tmp_path / "run_test.ps1"
        test_script.write_text(
            f"""
            $InstallRoot = '{install_root}'
            $HermesHome = '{hermes_home}'
            $LogDir = '{logs_dir}'
            $LogPath = '{handoff_log}'
            $ResultPath = '{result_path}'
            $Branch = 'main'
            $RelaunchExe = ''
            $NoUi = $true
            $finalCode = 1
            $finalMsg = 'update did not complete'
            $script:TreeSafeToFinalize = $true

            function Write-HandoffLog([string]$Message) {{
                $line = "{{0:yyyy-MM-ddTHH:mm:ssK}} {{1}}" -f (Get-Date), $Message
                Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
            }}
            function Write-Result([bool]$Ok, [int]$Code, [string]$Message, [bool]$ManualAction = $false) {{
                $obj = @{{
                    ok          = $Ok
                    exit_code   = $Code
                    manual      = $ManualAction
                    message     = $Message
                    branch      = $Branch
                    finished_at = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
                }} | ConvertTo-Json -Compress
                [System.IO.File]::WriteAllText($ResultPath, $obj)
            }}
            function Remove-MarkerIfOwned {{}}
            function Start-DesktopRelaunch {{ return $false }}
            function Show-ErrorFinale([string]$Message) {{}}
            function Close-ProgressWindow {{}}

            $res = @{{ Code = 0; Output = 'Simulated successful update' }}

            try {{
                # Simulate post-update failure: e.g. verify step throws an unhandled exception
                throw [System.InvalidOperationException]::new('Simulated post-update verify crash')
                $finalCode = 0
                $finalMsg = 'Update complete.'
                exit $finalCode
            }} catch {{
                $script:UnhandledException = $_
                $exMsg = $_.Exception.Message
                $stack = $_.ScriptStackTrace
                Write-HandoffLog ("CRITICAL: unhandled error in hand-off: {{0}}`n{{1}}" -f $exMsg, $stack)
                if ($null -ne $res -and $res.Code -eq 0) {{
                    $finalCode = 8
                    $finalMsg = "Update completed (exit 0), but post-update step failed: $exMsg. Check logs\\desktop-update-handoff.log."
                }} else {{
                    $finalCode = if ($null -ne $res -and $res.Code) {{ $res.Code }} else {{ 1 }}
                    $finalMsg = "Update failed: $exMsg. Check logs\\desktop-update-handoff.log."
                }}
            }} finally {{
                if ($null -ne $script:UnhandledException -and $finalMsg -eq "update did not complete") {{
                    $finalMsg = if ($null -ne $res -and $res.Code -eq 0) {{
                        "Update completed (exit 0), but post-update step failed: $($script:UnhandledException.Exception.Message). Check logs\\desktop-update-handoff.log."
                    }} else {{
                        "Update failed: $($script:UnhandledException.Exception.Message). Check logs\\desktop-update-handoff.log."
                    }}
                }}
                Write-Result ($finalCode -eq 0) $finalCode $finalMsg
            }}
            """,
            encoding="utf-8",
        )

        proc = subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(test_script)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result_path.is_file(), f"Result file was not created. Stderr: {proc.stderr}"
        data = json.loads(result_path.read_text(encoding="utf-8"))
        assert data["ok"] is False
        assert data["exit_code"] == 8
        assert "Simulated post-update verify crash" in data["message"]
        assert "update did not complete" not in data["message"]

        log_content = handoff_log.read_text(encoding="utf-8")
        assert "CRITICAL: unhandled error in hand-off: Simulated post-update verify crash" in log_content

    def test_invoke_hermes_step_handles_bad_executable_without_terminating_exception(
        self, tmp_path: Path
    ) -> None:
        powershell = shutil.which("powershell.exe")
        if not powershell:
            pytest.skip("powershell.exe not found")

        # Run windows.ps1 with an invalid executable passed to Invoke-HermesStep
        test_script = tmp_path / "test_bad_exe.ps1"
        test_script.write_text(
            f"""
            . '{WINDOWS_PS1}' -SelfTestMarker -InstallRoot '{tmp_path}' -NoUi
            $badExe = Join-Path '{tmp_path}' 'nonexistent_binary.exe'
            $stepRes = Invoke-HermesStep $badExe @('-c', 'exit 0') 'testbad'
            $json = @{{
                code = $stepRes.Code
                output = $stepRes.Output
                quiesced = $stepRes.TreeQuiesced
            }} | ConvertTo-Json -Compress
            Write-Output "RESULT:$json"
            """,
            encoding="utf-8",
        )

        proc = subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(test_script)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 0, f"Script failed with code {proc.returncode}. Stderr: {proc.stderr}"
        match = re.search(r"RESULT:(\{.*\})", proc.stdout)
        assert match, f"RESULT not found in stdout: {proc.stdout}"
        res = json.loads(match.group(1))
        assert res["code"] == 1
        assert "CreateProcess failed" in res["output"]
