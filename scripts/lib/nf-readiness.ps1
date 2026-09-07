# scripts/lib/nf-readiness.ps1 - dot-source me; not a standalone script.
#
# ONE canonical answer to "can this venv actually run North Forge from THIS
# checkout, right now, on THIS machine?" - shared by:
#   * scripts/nf-preflight.ps1        (the launch path: north-forge.cmd calls it)
#   * scripts/bootstrap-north-forge.ps1 (its "already bootstrapped?" early-return)
#
# WHY (ERR-2026-09-07-006, reclassified CRITICAL after a real first-handoff
# failure): a Python venv built on one machine is NOT portable to another. Copy
# the drive to a new computer - or just rename / re-letter the checkout - and
# hermes.exe + .nf-bootstrapped are still sitting there, so every "does the file
# exist?" check passes, yet the interpreter cannot start or imports stale code
# from a path that no longer exists. "The file exists" was never "the environment
# works". This runs a real probe instead.
#
# The probe is four checks. ANY failure means the venv is not trustworthy and the
# caller must rebuild it (never the data folder - data is untouched, always):
#   1. python_exec          - the venv's python.exe actually executes
#   2. import_hermes_cli    - `import hermes_cli` succeeds in that interpreter
#   3. module_in_checkout   - the resolved hermes_cli lives under THIS checkout,
#                             not a different/old one
#   4. marker_repo_matches  - .nf-bootstrapped's recorded repo= path is THIS
#                             checkout's actual path
#
# No output, no dialogs, no exit calls here - this file only reports. The caller
# decides what to do and does it silently (the "self-healing" principle already
# used for the drive-letter fixes). Dot-sourced: it defines functions and must
# not change the caller's execution preferences.

function Get-NfCanonicalDir {
    <# Canonical form of a directory path for comparison: absolute, forward/back
       slashes normalized, trailing separators trimmed, a bare drive root kept as
       "D:\". Returns '' for a null/blank/unparseable path (never throws). #>
    param([Parameter(Mandatory = $true)][AllowEmptyString()][AllowNull()][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return '' }
    try { $full = [System.IO.Path]::GetFullPath($Path.Trim()) } catch { return '' }
    $trimmed = $full.TrimEnd([char]'\', [char]'/')
    if ($trimmed -match '^[A-Za-z]:$') { $trimmed += '\' }   # keep "D:\", not "D:"
    return $trimmed
}

function Test-NfPathsEqual {
    <# True when A and B resolve to the same directory (ordinal, case-insensitive
       - Windows paths). '' never equals anything. #>
    param([AllowEmptyString()][AllowNull()][string]$A, [AllowEmptyString()][AllowNull()][string]$B)
    $ca = Get-NfCanonicalDir $A
    $cb = Get-NfCanonicalDir $B
    if (-not $ca -or -not $cb) { return $false }
    return [string]::Equals($ca, $cb, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-NfMarkerRepo {
    <# The repo= value recorded in a .nf-bootstrapped marker, or $null if the
       file is missing or has no repo= line. #>
    param([Parameter(Mandatory = $true)][string]$MarkerPath)
    if (-not (Test-Path -LiteralPath $MarkerPath -PathType Leaf)) { return $null }
    try { $lines = [System.IO.File]::ReadAllLines($MarkerPath) } catch { return $null }
    foreach ($line in $lines) {
        if ($line -match '^\s*repo\s*=\s*(.+?)\s*$') { return $Matches[1] }
    }
    return $null
}

function Invoke-NfPythonProbe {
    <# Run `python.exe -I -c <Code>` with stdout/stderr captured and a hard
       timeout. -I isolates the probe: no PYTHONPATH, no user-site, cwd is not on
       sys.path - so a broken venv can't be rescued by ambient env, and a stray
       hermes_cli/ dir in the working directory can't fake check 3. .pth files in
       the venv's own site-packages (editable install) ARE still honored (-I does
       not imply -S). Returns @{ ExitCode; StdOut; StdErr; TimedOut }. #>
    param(
        [Parameter(Mandatory = $true)][string]$PyExe,
        [Parameter(Mandatory = $true)][string]$Code,
        [int]$TimeoutSec = 30
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $PyExe
    $psi.Arguments              = '-I -c "' + $Code + '"'   # $Code: single-quoted Python literals only, never a double quote
    $psi.UseShellExecute        = $false
    $psi.CreateNoWindow         = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    try {
        $psi.WorkingDirectory = [System.IO.Path]::GetTempPath()   # never probe with cwd inside the checkout
    } catch { }

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    try {
        [void]$p.Start()
    } catch {
        return @{ ExitCode = -1; StdOut = ''; StdErr = "could not start '$PyExe': $($_.Exception.Message)"; TimedOut = $false }
    }
    # Async reads so a full pipe buffer can't deadlock against WaitForExit.
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill() } catch { }
        return @{ ExitCode = -1; StdOut = ''; StdErr = "timed out after ${TimeoutSec}s"; TimedOut = $true }
    }
    $p.WaitForExit()   # let the async readers drain
    return @{
        ExitCode = $p.ExitCode
        StdOut   = [string]$outTask.Result
        StdErr   = [string]$errTask.Result
        TimedOut = $false
    }
}

function Test-NfVenvReady {
    <# Run the four readiness checks. Returns a [pscustomobject]:
         .Ready      [bool]   - every check passed
         .Checks     [ordered]- name -> [bool]
         .Details    [ordered]- name -> short human string (why it passed/failed)
         .Summary    [string] - "python_exec=PASS import_hermes_cli=PASS ..."
         .MarkerRepo [string] - repo= from the marker, or $null
         .RepoRoot   [string] - canonical checkout path used for the comparison
         .VenvDir    [string] - canonical venv path
       Never throws; never writes output; never exits. #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$VenvDir,
        [int]$ProbeTimeoutSec = 30
    )

    $repoCanon = Get-NfCanonicalDir $RepoRoot
    $pyExe     = Join-Path $VenvDir 'Scripts\python.exe'
    $marker    = Join-Path $VenvDir '.nf-bootstrapped'

    $checks  = [ordered]@{
        python_exec         = $false
        import_hermes_cli   = $false
        module_in_checkout  = $false
        marker_repo_matches = $false
    }
    $details = [ordered]@{}

    # -- check 4: marker repo= matches this checkout -----------------------------
    # Pure PowerShell - needs no working interpreter, so it still runs (and still
    # logs a verdict) when Python itself is the thing that's broken.
    $markerRepo = Get-NfMarkerRepo -MarkerPath $marker
    if ($null -eq $markerRepo) {
        $details['marker_repo_matches'] = 'no .nf-bootstrapped marker (or no repo= line)'
    } elseif (Test-NfPathsEqual $markerRepo $RepoRoot) {
        $checks['marker_repo_matches'] = $true
        $details['marker_repo_matches'] = "marker repo=$markerRepo"
    } else {
        $details['marker_repo_matches'] = "marker repo=$markerRepo != checkout $repoCanon (foreign / moved venv)"
    }

    # -- check 1: the venv's python.exe actually executes ----------------------
    if (-not (Test-Path -LiteralPath $pyExe -PathType Leaf)) {
        $details['python_exec'] = "no python.exe at $pyExe"
    } else {
        $r = Invoke-NfPythonProbe -PyExe $pyExe -Code "import sys; sys.stdout.write('NF-OK')" -TimeoutSec $ProbeTimeoutSec
        if ($r.ExitCode -eq 0 -and ([string]$r.StdOut).Trim() -eq 'NF-OK') {
            $checks['python_exec'] = $true
            $details['python_exec'] = 'python.exe runs'
        } elseif ($r.TimedOut) {
            $details['python_exec'] = "python.exe $($r.StdErr)"
        } else {
            $tail = (([string]$r.StdErr).Trim() -split "`n" | Select-Object -Last 1).Trim()
            $details['python_exec'] = "python.exe exit=$($r.ExitCode) $tail".Trim()
        }
    }

    # -- checks 2 + 3: import hermes_cli, and it resolves under this checkout ---
    if ($checks['python_exec']) {
        # print(<repo root two levels above hermes_cli/__init__.py>)
        $code = 'import os, hermes_cli; print(os.path.dirname(os.path.dirname(os.path.abspath(hermes_cli.__file__))))'
        $r = Invoke-NfPythonProbe -PyExe $pyExe -Code $code -TimeoutSec $ProbeTimeoutSec
        if ($r.ExitCode -eq 0) {
            $checks['import_hermes_cli'] = $true
            $details['import_hermes_cli'] = 'import hermes_cli ok'
            $modRoot = ([string]$r.StdOut).Trim()
            if (Test-NfPathsEqual $modRoot $RepoRoot) {
                $checks['module_in_checkout'] = $true
                $details['module_in_checkout'] = "hermes_cli resolves under $modRoot"
            } else {
                $details['module_in_checkout'] = "hermes_cli resolves under '$modRoot', not this checkout '$repoCanon'"
            }
        } else {
            $tail = (([string]$r.StdErr).Trim() -split "`n" | Select-Object -Last 1).Trim()
            if ($r.TimedOut) { $tail = [string]$r.StdErr }
            $details['import_hermes_cli'] = "import hermes_cli failed: $tail".Trim()
            $details['module_in_checkout'] = 'skipped - import hermes_cli failed'
        }
    } else {
        $details['import_hermes_cli']  = 'skipped - python.exe not runnable'
        $details['module_in_checkout'] = 'skipped - python.exe not runnable'
    }

    $ready = $true
    foreach ($v in $checks.Values) { if (-not $v) { $ready = $false } }

    $summary = (@($checks.Keys | ForEach-Object {
        '{0}={1}' -f $_, $(if ($checks[$_]) { 'PASS' } else { 'FAIL' })
    }) -join ' ')

    return [pscustomobject]@{
        Ready      = $ready
        Checks     = $checks
        Details    = $details
        Summary    = $summary
        MarkerRepo = $markerRepo
        RepoRoot   = $repoCanon
        VenvDir    = Get-NfCanonicalDir $VenvDir
    }
}
