param([string]$WorkRoot = $env:TEMP)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$caseRoot = Join-Path $WorkRoot ('installer-native-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($caseRoot) | Out-Null
$ownedPids = [Collections.Generic.List[int]]::new()
$failures = [Collections.Generic.List[string]]::new()
$hostExe = (Get-Process -Id $PID).Path
function Encoded([string]$Code) { [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Code)) }
function Check([bool]$Condition, [string]$Message) {
    if (-not $Condition) { $script:failures.Add($Message); Write-Host "FAIL: $Message" }
    else { Write-Host "PASS: $Message" }
}
try {
    . (Join-Path $repoRoot 'scripts/install.ps1') -HermesHome (Join-Path $caseRoot 'home') -InstallDir (Join-Path $caseRoot 'repo')
    $stdout = Join-Path $caseRoot 'stdout.log'
    $stderr = Join-Path $caseRoot 'stderr.log'
    $result = Invoke-ProcessWithWallClockTimeout -FilePath $hostExe -ArgumentList @(
        '-NoProfile', '-NonInteractive', '-EncodedCommand',
        (Encoded '[Console]::Out.WriteLine("output-before-exit"); [Console]::Error.WriteLine("error-before-exit"); exit 7')
    ) -TimeoutSec 10 -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Check ($result.ExitCode -eq 7 -and -not $result.TimedOut) 'native exit status survives redirected output'
    Check ([IO.File]::ReadAllText($stdout).Contains('output-before-exit')) 'stdout is retained'
    Check ([IO.File]::ReadAllText($stderr).Contains('error-before-exit')) 'stderr is retained'

    $result = Invoke-ProcessWithWallClockTimeout -FilePath $hostExe -ArgumentList @(
        '-NoProfile', '-NonInteractive', '-EncodedCommand', (Encoded 'Start-Sleep -Seconds 3; exit 0')
    ) -TimeoutSec 10 -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Check ($result.ExitCode -eq 0 -and -not $result.TimedOut) 'healthy quiet work is allowed its full wall-clock budget'

    $elapsed = [Diagnostics.Stopwatch]::StartNew()
    $result = Invoke-ProcessWithWallClockTimeout -FilePath $hostExe -ArgumentList @(
        '-NoProfile', '-NonInteractive', '-EncodedCommand',
        (Encoded '$chunk = "x" * 8192; while ($true) { [Console]::Out.Write($chunk); [Threading.Thread]::Sleep(1) }')
    ) -TimeoutSec 8 -RedirectStandardOutput $stdout -RedirectStandardError $stderr 6>$null
    $elapsed.Stop()
    Check ($result.TimedOut -and $result.ExitCode -eq 124) 'continuous native output cannot starve the deadline'
    Check ($result.Output.Length -gt 8192) 'chatty child produced sustained output before termination'
    Check ($elapsed.Elapsed.TotalSeconds -lt 25) 'chatty child teardown and final output drain are bounded'

    $pidFile = Join-Path $caseRoot 'descendant.pid'
    $identityFile = Join-Path $caseRoot 'descendant.start'
    $writeFile = Join-Path $caseRoot 'descendant-writes.log'
    $child = "[IO.File]::WriteAllText('$($identityFile.Replace("'", "''"))', [string](Get-Process -Id `$PID).StartTime.ToUniversalTime().Ticks); [IO.File]::WriteAllText('$($pidFile.Replace("'", "''"))', [string]`$PID); while (`$true) { [IO.File]::AppendAllText('$($writeFile.Replace("'", "''"))', 'x'); Start-Sleep -Milliseconds 100 }"
    $childArgs = '-NoProfile -NonInteractive -EncodedCommand ' + (Encoded $child)
    $parent = "Start-Process -FilePath '$($hostExe.Replace("'", "''"))' -ArgumentList '$childArgs' -WindowStyle Hidden; while (-not (Test-Path -LiteralPath '$($pidFile.Replace("'", "''"))')) { Start-Sleep -Milliseconds 50 }; exit 0"
    $elapsed = [Diagnostics.Stopwatch]::StartNew()
    $result = Invoke-ProcessWithWallClockTimeout -FilePath $hostExe -ArgumentList @(
        '-NoProfile', '-NonInteractive', '-EncodedCommand', (Encoded $parent)
    ) -TimeoutSec 8 -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $elapsed.Stop()
    if (Test-Path -LiteralPath $pidFile) { $ownedPids.Add([int][IO.File]::ReadAllText($pidFile)) }
    Check ($ownedPids.Count -eq 1) 'actual native descendant started before its launcher exited'
    Check $result.TimedOut 'launcher exit does not release an active descendant from the deadline'
    Check ($elapsed.Elapsed.TotalSeconds -lt 20) 'deadline and teardown return within a bounded interval'
    foreach ($childPid in $ownedPids) {
        # A terminated Windows process can remain enumerable while another
        # handle references it. The execution contract is that it has exited.
        $descendant = Get-Process -Id $childPid -ErrorAction SilentlyContinue
        $sameProcess = $descendant -and -not $descendant.HasExited -and $descendant.StartTime.ToUniversalTime().Ticks -eq [long][IO.File]::ReadAllText($identityFile)
        Check (-not $sameProcess) 'owned native descendant has exited before return'
    }
    $before = if (Test-Path -LiteralPath $writeFile) { (Get-Item -LiteralPath $writeFile).Length } else { 0 }
    Start-Sleep -Seconds 2
    $after = if (Test-Path -LiteralPath $writeFile) { (Get-Item -LiteralPath $writeFile).Length } else { 0 }
    Check ($after -eq $before) 'no old writer survives to collide with a retry'
} finally {
    if (Get-Variable pidFile -ErrorAction SilentlyContinue) {
        if (Test-Path -LiteralPath $pidFile) {
            $childPid = [int][IO.File]::ReadAllText($pidFile)
            $owned = Get-Process -Id $childPid -ErrorAction SilentlyContinue
            if ($owned -and (Test-Path -LiteralPath $identityFile) -and
                $owned.StartTime.ToUniversalTime().Ticks -eq [long][IO.File]::ReadAllText($identityFile)) {
                Stop-Process -InputObject $owned -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
if ($failures.Count) { throw ($failures -join '; ') }
Write-Host 'PASS: native command ownership contracts'
