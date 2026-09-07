# nf-preflight - launch-time readiness probe + silent self-heal for North Forge.
#
# north-forge.cmd runs this BEFORE it starts the agent, on every launch. It
# replaces the old "if hermes.exe exists, we're good" assumption, which a real
# first-handoff failure proved wrong: a venv built on one machine is not portable
# to another (ERR-2026-09-07-006).
<#
  scripts\nf-preflight.ps1  -RepoRoot <path>  -VenvDir <path>  -DataDir <path>
                            [-LogFile <path>]  [-NoRebuild]  [-ProbeTimeoutSec 30]

  WHAT IT DOES
    1. Runs the four readiness checks (see scripts\lib\nf-readiness.ps1):
         python_exec          - the venv's python.exe actually executes
         import_hermes_cli    - `import hermes_cli` works in it
         module_in_checkout   - that hermes_cli lives under THIS checkout
         marker_repo_matches  - .nf-bootstrapped's repo= is THIS checkout's path
    2. If every check passes -> exit 0, the launcher starts the agent.
    3. If ANY check fails -> silently rebuild the venv (bootstrap-north-forge.ps1;
       -Force when a venv is already there). The data folder is NEVER touched.
       No dialog, no admin prompt, no user decision - the "self-healing"
       principle already used for the drive-letter fixes. Re-probe afterwards.
    4. Always append exactly one line to the launcher log (default:
       <parent>\<checkout-name>-launcher.log, a sibling of the checkout so it
       survives a venv rebuild): timestamp, computer name, drive + repo path,
       every check's PASS/FAIL, and the recovery action taken. This write does
       NOT depend on Python working - a broken interpreter can't stop the line
       from being written.

  -NoRebuild : run the probe and write the log, but never invoke bootstrap.
               (Used by tests, and available for a read-only diagnosis.)

  EXIT  0 = ready to launch (was ready, or rebuild/bootstrap fixed it)
        1 = not ready and the rebuild/bootstrap did not fix it (or errored)
        2 = not ready, -NoRebuild given (no repair attempted)
  REQUIRES  PowerShell 5.1+. Python/uv only matter when a (re)build is needed.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$VenvDir,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$LogFile,
    [switch]$NoRebuild,
    [int]$ProbeTimeoutSec = 30
)

$ErrorActionPreference = 'Stop'

# Resolve the checkout path up front (the probe and the log both compare against
# it). Everything else is best-effort and must not throw past the finally below.
try { $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path } catch { }
$RepoRoot = $RepoRoot.TrimEnd([char]'\', [char]'/')

# Normalized venv path for the log line only (north-forge.cmd can hand us
# "D:\\north-forge-agent-venv" when the checkout sits one level below a drive
# root); every real comparison in the probe canonicalizes on its own.
$venvDisplay = $VenvDir
try { $venvDisplay = [System.IO.Path]::GetFullPath($VenvDir).TrimEnd([char]'\', [char]'/') } catch { }

if (-not $LogFile) {
    $parent  = Split-Path -Parent $RepoRoot
    $leaf    = Split-Path -Leaf   $RepoRoot
    if (-not $parent) { $parent = $RepoRoot }   # RepoRoot is a drive root - degenerate, but don't crash
    $LogFile = Join-Path $parent ($leaf + '-launcher.log')
}

$driveRoot = ''
try { $driveRoot = [System.IO.Path]::GetPathRoot($RepoRoot) } catch { }
$computer = $env:COMPUTERNAME
if (-not $computer) { try { $computer = [System.Net.Dns]::GetHostName() } catch { $computer = 'unknown' } }

$action    = 'none'          # none | rebuild-venv | bootstrap
$result    = 'unknown'       # ready | rebuild-ok | rebuild-failed | bootstrap-ok | bootstrap-failed | not-ready-norebuild | preflight-error:<msg>
$probe     = $null
$reprobe   = $null
$exitCode  = 1

function Write-NfLauncherLine {
    # Self-contained: no dependency on the readiness lib, so it still writes even
    # if dot-sourcing that lib is what failed. Single rotation at ~1 MB.
    param([string]$Path, [string]$Line)
    try {
        $dir = Split-Path -Parent $Path
        if ($dir -and -not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
        if ((Test-Path -LiteralPath $Path -PathType Leaf) -and
            ((Get-Item -LiteralPath $Path).Length -gt 1048576)) {
            $rot = "$Path.1"
            if (Test-Path -LiteralPath $rot) { Remove-Item -LiteralPath $rot -Force -ErrorAction SilentlyContinue }
            Move-Item -LiteralPath $Path -Destination $rot -Force -ErrorAction SilentlyContinue
        }
        [System.IO.File]::AppendAllText($Path, $Line + "`r`n", (New-Object System.Text.UTF8Encoding($false)))
    } catch {
        # Last resort: the launcher log is diagnostic, never load-bearing. A
        # locked or unwritable log must not stop the agent from starting.
        Write-Warning "nf-preflight: could not write launcher log '$Path': $($_.Exception.Message)"
    }
}

try {
    . (Join-Path $PSScriptRoot 'lib\nf-readiness.ps1')

    $probe = Test-NfVenvReady -RepoRoot $RepoRoot -VenvDir $VenvDir -ProbeTimeoutSec $ProbeTimeoutSec

    if ($probe.Ready) {
        $action = 'none'; $result = 'ready'; $exitCode = 0
    }
    elseif ($NoRebuild) {
        $action = 'none'; $result = 'not-ready-norebuild'; $exitCode = 2
    }
    else {
        # --- silent self-heal: rebuild the venv, never the data folder --------
        $pyExe     = Join-Path $VenvDir 'Scripts\python.exe'
        $bootstrap = Join-Path $RepoRoot 'scripts\bootstrap-north-forge.ps1'
        $venvThere = Test-Path -LiteralPath $pyExe -PathType Leaf
        $action    = if ($venvThere) { 'rebuild-venv' } else { 'bootstrap' }

        Write-Host "[nf-preflight] environment not ready ($($probe.Summary)) - $action ..." -ForegroundColor Yellow
        foreach ($k in $probe.Details.Keys) {
            if (-not $probe.Checks[$k]) { Write-Host "[nf-preflight]   $k : $($probe.Details[$k])" }
        }

        if (-not (Test-Path -LiteralPath $bootstrap -PathType Leaf)) {
            throw "cannot self-heal: '$bootstrap' is missing"
        }

        if ($venvThere) {
            & $bootstrap -Force -RepoRoot $RepoRoot -VenvDir $VenvDir -DataDir $DataDir
        } else {
            & $bootstrap -RepoRoot $RepoRoot -VenvDir $VenvDir -DataDir $DataDir
        }
        $bootExit = $LASTEXITCODE

        $reprobe = Test-NfVenvReady -RepoRoot $RepoRoot -VenvDir $VenvDir -ProbeTimeoutSec $ProbeTimeoutSec
        if ($reprobe.Ready) {
            $result   = if ($action -eq 'bootstrap') { 'bootstrap-ok' } else { 'rebuild-ok' }
            $exitCode = 0
        } else {
            $result   = if ($action -eq 'bootstrap') { 'bootstrap-failed' } else { 'rebuild-failed' }
            $exitCode = 1
            Write-Host "[nf-preflight] rebuild finished (bootstrap exit $bootExit) but the venv still fails the readiness probe: $($reprobe.Summary)" -ForegroundColor Red
        }
    }
}
catch {
    $result   = 'preflight-error: ' + ($_.Exception.Message -replace '\s+', ' ')
    $exitCode = 1
}
finally {
    $ts = (Get-Date).ToString('yyyy-MM-ddTHH:mm:sszzz')

    $checkStr = if ($probe) { $probe.Summary } else { '(probe did not run)' }
    if ($reprobe) { $checkStr = "$checkStr  -> after ${action}: $($reprobe.Summary)" }

    $line = ('{0} | host={1} | drive={2} | repo={3} | venv={4} | checks: {5} | action={6} | result={7}' -f `
        $ts, $computer, $driveRoot, $RepoRoot, $venvDisplay, $checkStr, $action, $result)

    Write-NfLauncherLine -Path $LogFile -Line $line
    Write-Host "[nf-preflight] $line"
}

exit $exitCode
