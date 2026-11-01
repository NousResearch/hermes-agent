# Native-Windows evidence harness for the base-interpreter update-lock hypothesis.
#
# This is intentionally a real subprocess / real DLL experiment:
#   1. create a real venv;
#   2. place a real CPython native extension in venv\Lib\site-packages;
#   3. start the BASE interpreter with that site-packages directory inherited
#      through PYTHONPATH (the behavior proposed by PR #74559);
#   4. prove which .pyd was imported;
#   5. try to delete it while mapped, then after interpreter exit.

$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') {
    Write-Host 'SKIP: native-extension file locking is Windows-specific'
    return
}

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$basePython = (Get-Command python -ErrorAction Stop).Source
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-venv-lock-proof-" + [Guid]::NewGuid().ToString('N'))
$venvRoot = Join-Path $tempRoot 'venv'
$sitePackages = Join-Path $venvRoot 'Lib\site-packages'
$childScript = Join-Path $tempRoot 'hold_native_extension.py'
$readyFile = Join-Path $tempRoot 'loaded-extension.txt'
$child = $null
$savedPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH')

try {
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

    & $basePython -m venv --without-pip $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create proof venv (exit $LASTEXITCODE)"
    }

    $sourceNative = (& $basePython -c "import _sqlite3; print(_sqlite3.__file__)" | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $sourceNative -PathType Leaf)) {
        throw "Could not locate base Python's _sqlite3 native extension: $sourceNative"
    }

    New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null
    $venvNative = Join-Path $sitePackages ([System.IO.Path]::GetFileName($sourceNative))
    Copy-Item -LiteralPath $sourceNative -Destination $venvNative -Force

    # The runner's _sqlite3.pyd depends on sqlite3.dll from the same DLLs
    # directory. Copy sibling DLLs into the probe directory so the child
    # process can import the venv-shadowed extension instead of exiting before
    # the lock assertion. Windows searches the extension's directory for DLL
    # dependencies, so this does not weaken the proof: the .pyd is still loaded
    # from venv\Lib\site-packages and remains the file we attempt to delete.
    Get-ChildItem -LiteralPath (Split-Path -Parent $sourceNative) -Filter '*.dll' -File -ErrorAction SilentlyContinue |
        Copy-Item -Destination $sitePackages -Force

    @'
import _sqlite3
import pathlib
import sys
import time

pathlib.Path(sys.argv[1]).write_text(str(pathlib.Path(_sqlite3.__file__).resolve()), encoding="utf-8")
time.sleep(120)
'@ | Set-Content -LiteralPath $childScript -Encoding UTF8

    # Mirror Desktop's ordering: checkout root, venv site-packages, inherited
    # PYTHONPATH. Crucially, FileName is the BASE interpreter, not venv Python.
    $entries = @($repoRoot, $sitePackages)
    if ($savedPythonPath) { $entries += $savedPythonPath }
    $env:PYTHONPATH = $entries -join ';'

    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $basePython
    $start.Arguments = ('"{0}" "{1}"' -f $childScript, $readyFile)
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $child = [System.Diagnostics.Process]::Start($start)

    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while (-not (Test-Path -LiteralPath $readyFile -PathType Leaf)) {
        if ($child.HasExited) {
            throw "Base-Python proof child exited before importing the native extension (exit $($child.ExitCode))"
        }
        if ([DateTime]::UtcNow -gt $deadline) {
            throw 'Timed out waiting for base-Python proof child to import _sqlite3'
        }
        Start-Sleep -Milliseconds 100
    }

    $importedNative = (Get-Content -LiteralPath $readyFile -Raw).Trim()
    $expectedNative = [System.IO.Path]::GetFullPath($venvNative)
    if (-not [String]::Equals($importedNative, $expectedNative, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Proof child imported the wrong extension. Expected $expectedNative; got $importedNative"
    }

    $deleteBlocked = $false
    $deleteError = ''
    try {
        Remove-Item -LiteralPath $venvNative -Force -ErrorAction Stop
    } catch {
        $deleteBlocked = $true
        $deleteError = $_.Exception.GetType().FullName + ': ' + $_.Exception.Message
    }

    Write-Host "OBSERVED base_python=$basePython"
    Write-Host "OBSERVED inherited_pythonpath=$($env:PYTHONPATH)"
    Write-Host "OBSERVED imported_native=$importedNative"
    Write-Host "OBSERVED delete_while_loaded=$(if ($deleteBlocked) { 'BLOCKED' } else { 'SUCCEEDED' })"
    if ($deleteError) { Write-Host "OBSERVED lock_error=$deleteError" }

    if (-not $deleteBlocked -or -not (Test-Path -LiteralPath $venvNative -PathType Leaf)) {
        throw 'Expected Windows to block deletion of the loaded venv .pyd, but deletion succeeded'
    }

    $child.Kill()
    if (-not $child.WaitForExit(10000)) {
        throw 'Proof child did not exit after Kill()'
    }
    $child.Dispose()
    $child = $null

    Remove-Item -LiteralPath $venvNative -Force -ErrorAction Stop
    if (Test-Path -LiteralPath $venvNative) {
        throw 'Loaded .pyd remained undeletable after base interpreter exited'
    }

    Write-Host 'OBSERVED delete_after_exit=SUCCEEDED'
    Write-Host 'PASS: base Python + venv site-packages on PYTHONPATH locks the loaded .pyd on Windows'
} finally {
    if ($child -and -not $child.HasExited) {
        try { $child.Kill() } catch {}
        try { $child.WaitForExit(5000) | Out-Null } catch {}
    }
    if ($child) { try { $child.Dispose() } catch {} }

    if ($null -eq $savedPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $savedPythonPath
    }

    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
