param([string]$Python, [string]$Receipt, [int]$ExpectedProgressCode = 7)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$temp = Join-Path ([IO.Path]::GetTempPath()) ('update-watchdog-' + [guid]::NewGuid())
New-Item -ItemType Directory $temp | Out-Null
$env:HOME = $temp
$env:USERPROFILE = $temp
$env:HERMES_HOME = $temp
$LogDir = Join-Path $temp 'logs'
New-Item -ItemType Directory $LogDir | Out-Null
$script:Ui = $null
$script:TreeSafeToFinalize = $true
function Write-HandoffLog([string]$Message) {
    Add-Content -LiteralPath (Join-Path $temp 'handoff.log') -Value $Message -Encoding UTF8
}
# Execute the maintained process/job/watchdog boundary, not a reimplementation.
# Deliberately omit all actual update, service, marker and Desktop launch phases.
$source = [IO.File]::ReadAllText((Join-Path $repo 'scripts/desktop-update/windows.ps1'))
$start = $source.IndexOf('$script:StepDrainGraceSeconds = 20')
$end = $source.IndexOf('$finalCode = 1', $start)
if ($start -lt 0 -or $end -lt $start) { throw 'SETUP FAIL: handoff boundary not found' }
. ([scriptblock]::Create($source.Substring($start, $end - $start)))
$script:StepIdleTimeoutSeconds = 3
$script:StepDrainGraceSeconds = 2
$rows = @()
try {
    foreach ($mode in @('progress', 'silent')) {
        Remove-Item -LiteralPath $script:StepProgressLogPath -ErrorAction SilentlyContinue
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $result = Invoke-HermesStep $Python @((Join-Path $PSScriptRoot 'live_output.py'), '--step', $mode) $mode
        $rows += @{ mode=$mode; code=$result.Code; seconds=$clock.Elapsed.TotalSeconds; tree_quiesced=$result.TreeQuiesced; job_assigned=$result.StartedAfterJobAssignment; output=$result.Output }
    }
    @{ platform='native Windows'; python=$Python; rows=$rows } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Receipt -Encoding UTF8
    if ($rows[0].code -ne $ExpectedProgressCode) { throw "progress code $($rows[0].code), expected $ExpectedProgressCode" }
    if ($rows[1].code -ne 124) { throw "silent child escaped watchdog: $($rows[1].code)" }
    foreach ($row in $rows) {
        if (-not $row.tree_quiesced -or -not $row.job_assigned) { throw 'unsafe child lifecycle' }
        if ($row.output.Length -ne 0) { throw 'build output leaked to screen pipe' }
    }
    Write-Host ('VERDICT: progress={0}, silent=124; private Windows job quiesced' -f $rows[0].code)
} finally {
    Copy-Item -LiteralPath (Join-Path $temp 'handoff.log') -Destination ($Receipt + '.log') -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $temp -Recurse -Force
}
