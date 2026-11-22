# Journey-owned mock and ordinary window close, shared by source and MSIX drivers.
# Callers supply a checkout-provisioned Node; never resolve tooling from OLD.
function Start-DesktopJourneyMock([string]$Node, [string]$Assets, [string]$Work, [string]$HermesHome, [string]$Out) {
    New-Item -ItemType Directory -Path $Out -Force | Out-Null
    $urlFile = Join-Path $Work 'chat-mock-url'
    # An earlier step's mock died with that step; the detached gateway did not, and it
    # kept the provider URL it read at boot. Rebind that URL's port (see mock-provider.mjs).
    $priorUrl = if (Test-Path -LiteralPath $urlFile) { (Get-Content -LiteralPath $urlFile -Raw).Trim() } else { '' }
    Remove-Item -LiteralPath $urlFile -Force -ErrorAction SilentlyContinue
    $mockArgs = @(('"' + (Join-Path $Assets 'mock-provider.mjs') + '"'), ('"' + $urlFile + '"'))
    if ($priorUrl) { $mockArgs += ('"' + $priorUrl + '"') }
    $mock = Start-Process -FilePath $Node -ArgumentList $mockArgs `
        -PassThru -RedirectStandardOutput (Join-Path $Out 'chat-mock.log') -RedirectStandardError (Join-Path $Out 'chat-mock-error.log')
    try {
        $deadline = (Get-Date).AddSeconds(30)
        while (-not (Test-Path -LiteralPath $urlFile)) {
            if ($mock.HasExited -or (Get-Date) -ge $deadline) { throw 'Desktop mock did not become ready' }
            Start-Sleep -Milliseconds 200
        }
        $env:HERMES_E2E_MOCK_URL = (Get-Content -LiteralPath $urlFile -Raw).Trim()
        Invoke-RestMethod -Uri "$env:HERMES_E2E_MOCK_URL/__e2e__/prompts" -TimeoutSec 5 | Out-Null
        $configWriter = [IO.Path]::GetFullPath((Join-Path $Assets '..\..\..\tests-js\scripts\mock-provider-config.ts'))
        & $Node $configWriter $HermesHome $env:HERMES_E2E_MOCK_URL
        if ($LASTEXITCODE -ne 0) { throw 'Desktop mock configuration failed' }
        return $mock
    } catch {
        if (-not $mock.HasExited) { Stop-Process -Id $mock.Id -ErrorAction SilentlyContinue }
        throw
    }
}

# The Actions runner holds every step process in a job object that refuses
# CREATE_BREAKAWAY_FROM_JOB (limit flags 0), and so does Task Scheduler for a task's
# process; Hermes deliberately refuses to start a gateway it cannot detach
# (gateway_runtime_start.py::spawn_unmanaged_gateway), so an app launched from either
# reports "Gateway inaccessible (windows_breakaway_unavailable)". A user's
# Explorer/Start-menu launch is in no job, and neither is a Win32_Process.Create child,
# which WMI starts outside its caller's job (probed on hosted runners). A desktop
# checkpoint therefore runs from such a child, with this step's environment, cwd and
# arguments replayed from a spec file. Returns the exit code.
$script:OutsideJobWrapper = @'
param([string]$SpecPath)
$ErrorActionPreference = 'Continue'
$code = 99
$spec = Get-Content -LiteralPath $SpecPath -Raw | ConvertFrom-Json
# Node writes UTF-8 into a pipe; decode it as such, not as the OEM code page. Cosmetic,
# and a session without a console rejects it, so a failure here must not fail the run.
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }
try {
    foreach ($entry in $spec.env.PSObject.Properties) { [Environment]::SetEnvironmentVariable($entry.Name, [string]$entry.Value, 'Process') }
    Set-Location -LiteralPath $spec.cwd
    & $spec.exe @($spec.args) 2>&1 | ForEach-Object { "$_" } | Set-Content -LiteralPath $spec.log -Encoding UTF8
    $code = $LASTEXITCODE
} catch {
    "outside-job wrapper failed: $_" | Add-Content -LiteralPath $spec.log -Encoding UTF8
} finally {
    # Write then rename: the parent stops waiting the moment this file exists, and a rename
    # within one directory is atomic, so it can never read a half-written exit code.
    Set-Content -LiteralPath "$($spec.exit).tmp" -Value $code -Encoding ascii
    Move-Item -LiteralPath "$($spec.exit).tmp" -Destination $spec.exit -Force
}
'@

function Invoke-OutsideRunnerJob([string]$Name, [string]$Work, [string]$Exe, [string[]]$Arguments, [int]$TimeoutMinutes = 45) {
    $dir = Join-Path $Work "outside-job-$Name"
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    $spec = Join-Path $dir 'spec.json'
    $log = Join-Path $dir 'output.log'
    $exitFile = Join-Path $dir 'exit-code'
    $wrapper = Join-Path $dir 'run.ps1'
    Remove-Item -LiteralPath $log, $exitFile -Force -ErrorAction SilentlyContinue
    $environment = [ordered]@{}
    Get-ChildItem Env: | ForEach-Object { $environment[$_.Name] = $_.Value }
    $cwd = (Get-Location).ProviderPath
    # -InputObject keeps a one-element argument list a JSON array.
    ConvertTo-Json -Depth 4 -InputObject ([ordered]@{ exe = $Exe; args = [object[]]@($Arguments); cwd = $cwd; env = $environment; log = $log; exit = $exitFile }) |
        Set-Content -LiteralPath $spec -Encoding UTF8
    $script:OutsideJobWrapper | Set-Content -LiteralPath $wrapper -Encoding UTF8
    # The same PowerShell as this step, so native argument passing matches a direct call.
    $shell = (Get-Process -Id $PID).Path
    $commandLine = "`"$shell`" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$wrapper`" -SpecPath `"$spec`""
    $wrapperPid = 0
    try {
        $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $commandLine; CurrentDirectory = $cwd }
        if ($created.ReturnValue -ne 0) { throw "Outside-job run '$Name' could not start (Win32_Process.Create returned $($created.ReturnValue))" }
        $wrapperPid = [int]$created.ProcessId
        $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
        while (-not (Test-Path -LiteralPath $exitFile)) {
            if ((Get-Date) -ge $deadline) { throw "Outside-job run '$Name' did not finish within $TimeoutMinutes minutes" }
            if (-not (Get-Process -Id $wrapperPid -ErrorAction SilentlyContinue)) {
                # The wrapper writes the exit file last; give a racing write a moment.
                Start-Sleep -Seconds 2
                if (-not (Test-Path -LiteralPath $exitFile)) { throw "Outside-job run '$Name' exited without an exit code" }
                break
            }
            Start-Sleep -Seconds 2
        }
    } finally {
        # Outside the runner's job, nothing reaps an unfinished run's tree. A finished run's
        # detached gateway is meant to outlive it, so only a run without an exit code is killed.
        if ($wrapperPid -and -not (Test-Path -LiteralPath $exitFile) -and (Get-Process -Id $wrapperPid -ErrorAction SilentlyContinue)) { & taskkill.exe /PID $wrapperPid /T /F | Out-Null }
        Write-Host "--- outside-job '$Name' output ($log) ---"
        if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Encoding UTF8 | ForEach-Object { Write-Host $_ } }
        Write-Host "--- end outside-job '$Name' output ---"
        # The spec replays this step's environment, runner tokens included.
        Remove-Item -LiteralPath $spec -Force -ErrorAction SilentlyContinue
    }
    $text = (Get-Content -LiteralPath $exitFile -Raw).Trim()
    if ($text -notmatch '^-?\d+$') { throw "Outside-job run '$Name' wrote no exit code ('$text')" }
    return [int]$text
}

function Get-VerifiedDesktopWindows([string]$Exe) {
    # The detached updater canonicalizes the WorkRoot path while the driver can
    # retain ``..`` segments. Compare executable identity, not path spelling.
    $expectedExe = [IO.Path]::GetFullPath($Exe)
    $rows = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -and [IO.Path]::GetFullPath($_.ExecutablePath) -ieq $expectedExe
    })
    return @($rows | ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue } | Where-Object { $_.MainWindowHandle -ne 0 })
}

function Close-VerifiedDesktop([string]$Exe, [int]$ProcessId = 0) {
    $windows = @(Get-VerifiedDesktopWindows $Exe)
    if ($ProcessId) { $windows = @($windows | Where-Object { $_.Id -eq $ProcessId }) }
    if ($windows.Count -ne 1) { throw 'Cannot identify exactly one verified desktop window to close normally' }
    $window = $windows[0]
    if (-not $window.CloseMainWindow()) { throw 'Verified desktop refused a normal window close' }
    if (-not $window.WaitForExit(30000)) { throw 'Verified desktop did not exit normally; no force-kill or smoke relaunch attempted' }
    $deadline = (Get-Date).AddSeconds(30)
    do {
        $remaining = @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -ieq $Exe })
        if (-not $remaining.Count) { return }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    throw 'Verified desktop children did not exit normally; no smoke relaunch attempted'
}

function Main-Processes([string]$Exe) {
    # Prevent the function output pipeline from unrolling zero/one process arrays.
    return ,@(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -ieq $Exe -and $_.CommandLine -notmatch '(?:^|\s)--type(?:=|\s)' })
}
