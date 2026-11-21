# Tests for install.ps1's UTF-8 console pin (Set-ConsoleUtf8).
#
# Run from a PowerShell prompt:
#
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-utf8-console.ps1
#
# Background: on a CJK-locale Windows (OEM codepage 936) the installer's
# piped stdout was written in GBK while the Hermes-Setup reader decodes
# UTF-8 first and falls back to CP1252 -- so bootstrap-installer.log, the
# log the failure dialog points users at, recorded '?'-runs instead of the
# localized system-error text needed to diagnose the failure (#132531).
# install.ps1 now pins [Console]::OutputEncoding (both pipe directions),
# $OutputEncoding, and PYTHONUTF8 / PYTHONIOENCODING for Python children,
# mirroring scripts/desktop-update/windows.ps1.
#
# HOW THIS RUNS THE CODE: no source reading. Case 1 dirties the live
# console/pipe encodings and invokes the dot-sourced Set-ConsoleUtf8
# directly, asserting it repairs them. Case 2 runs install.ps1 as a real
# subprocess (-ShowResolvedPaths, the side-effect-free early exit BELOW the
# pin) with a non-ASCII HERMES_HOME and a console pre-dirtied to Latin-1,
# then reads the child's stdout as RAW BYTES: if the pin is wired into the
# entry path, the bytes are UTF-8 and the JSON's hermes_home round-trips
# the non-ASCII path; without the pin PowerShell encodes it as '?' and the
# round-trip assertion fails. The pre-dirtied console is what gives the
# case red/green discrimination on UTF-8-default hosts (macOS/Linux CI)
# too, not just on real cp936 Windows.
#
# Portability: both cases run on any host with PowerShell 7. On Windows the
# same assertions also cover the OEM-codepage shape from the report.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts/install.ps1"

if (-not (Test-Path $installScript)) {
    throw "Could not locate install.ps1 at $installScript"
}

$failures = 0

function Assert-Equal {
    param($Expected, $Actual, [Parameter(Mandatory = $true)][string]$Label)
    if ("$Expected" -ne "$Actual") {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        Write-Host "  expected: $Expected"
        Write-Host "  actual:   $Actual"
        $script:failures++
    } else {
        Write-Host "PASS: $Label" -ForegroundColor Green
    }
}

function Assert-True {
    param([bool]$Condition, [Parameter(Mandatory = $true)][string]$Label)
    if ($Condition) { Write-Host "PASS: $Label" -ForegroundColor Green }
    else {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        $script:failures++
    }
}

# Tripwires exist before dot-sourcing, so a broken guard cannot run an install.
function Invoke-WebRequest { throw 'unexpected download' }
function Invoke-RestMethod { throw 'unexpected download' }
function git { throw 'unexpected git command' }
function uv { throw 'unexpected uv command' }
function node { throw 'unexpected node command' }
function npm { throw 'unexpected npm command' }

# --- Case 1: Set-ConsoleUtf8 repairs dirtied console/pipeline encodings -------

# Latin-1: built into .NET Core, cannot encode CJK text (the '?' fallback is
# exactly the corruption under test), and is settable on any host.
$dirty = [System.Text.Encoding]::GetEncoding('iso-8859-1')

$savedConsole = [Console]::OutputEncoding
$savedPipe = $global:OutputEncoding
$savedUtf8 = $env:PYTHONUTF8
$savedIoEncoding = $env:PYTHONIOENCODING
$canDirtyConsole = $true
try {
    [Console]::OutputEncoding = $dirty
} catch {
    # A host with no settable console (some CI service hosts) cannot run the
    # discrimination; say so rather than fail.
    $canDirtyConsole = $false
}

try {
    $dotSourceHome = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-utf8-dotsource-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
    . $installScript -HermesHome $dotSourceHome -InstallDir (Join-Path $dotSourceHome 'hermes-agent')

    if ($canDirtyConsole) {
        [Console]::OutputEncoding = $dirty
        $global:OutputEncoding = $dirty
        Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
        Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue

        Set-ConsoleUtf8

        Assert-Equal 'utf-8' ([Console]::OutputEncoding.WebName) 'Set-ConsoleUtf8 repairs the console output encoding'
        Assert-Equal 'utf-8' ($global:OutputEncoding.WebName) 'Set-ConsoleUtf8 repairs the pipeline output encoding'
        Assert-Equal '1' $env:PYTHONUTF8 'Set-ConsoleUtf8 exports PYTHONUTF8 for Python children'
        Assert-Equal 'utf-8' $env:PYTHONIOENCODING 'Set-ConsoleUtf8 exports PYTHONIOENCODING for Python children'
    } else {
        Write-Host "SKIP: Case 1 (console encoding not settable on this host)" -ForegroundColor Yellow
    }
} finally {
    if ($canDirtyConsole) {
        [Console]::OutputEncoding = $savedConsole
        $global:OutputEncoding = $savedPipe
    }
    if ($null -eq $savedUtf8) { Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue } else { $env:PYTHONUTF8 = $savedUtf8 }
    if ($null -eq $savedIoEncoding) { Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue } else { $env:PYTHONIOENCODING = $savedIoEncoding }
}

# --- Case 2: the entry path pins the console before any output line ----------

# A HERMES_HOME whose name a non-UTF-8 codepage cannot represent: if the
# installer's stdout leaves in the dirtied encoding, these characters arrive
# as '?' and the JSON round-trip below fails.
$nonAscii = "安装-instalál-probe"
$probeHome = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-utf8-pin-" + $nonAscii + "-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $probeHome | Out-Null

$pwshExe = (Get-Process -Id $PID).Path
$inner = "[Console]::OutputEncoding = [System.Text.Encoding]::GetEncoding('iso-8859-1'); & '$installScript' -ShowResolvedPaths -HermesHome '$probeHome'"

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $pwshExe
$psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -Command `"$inner`""
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true

$child = [System.Diagnostics.Process]::Start($psi)
$captured = [System.IO.MemoryStream]::new()
$child.StandardOutput.BaseStream.CopyTo($captured)
$stderrText = $child.StandardError.ReadToEnd()
$child.WaitForExit()
$stdoutBytes = $captured.ToArray()

try {
    Assert-Equal 0 $child.ExitCode "installer early-exit runs clean under a dirtied console (stderr: $(if ($stderrText) { $stderrText.Trim() } else { '<empty>' }))"

    # Strict UTF-8: throws on any byte sequence the pin failed to cover.
    $strictUtf8 = [System.Text.UTF8Encoding]::new($false, $true)
    $stdoutText = $null
    $decodedOk = $true
    try {
        $stdoutText = $strictUtf8.GetString($stdoutBytes)
    } catch {
        $decodedOk = $false
    }
    Assert-True $decodedOk 'installer stdout under a dirtied console decodes as strict UTF-8'

    if ($decodedOk) {
        $report = $stdoutText | ConvertFrom-Json
        Assert-True ($null -ne $report.hermes_home) 'path report JSON parses with a hermes_home field'
        if ($null -ne $report.hermes_home) {
            Assert-Equal $probeHome $report.hermes_home 'non-ASCII HERMES_HOME round-trips through the pinned stdout'
        }
    }
} finally {
    Remove-Item -Recurse -Force $probeHome -ErrorAction SilentlyContinue
}

# --- Summary ------------------------------------------------------------------

if ($failures -gt 0) {
    Write-Host ""
    Write-Host "$failures assertion(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "All assertions passed." -ForegroundColor Green
exit 0
