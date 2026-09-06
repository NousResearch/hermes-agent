param([string]$WorkRoot = $env:TEMP, [string]$InstallerPath)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$caseRoot = Join-Path $WorkRoot ('desktop-deadlines space-' + [char]0x00e9 + '-' + [guid]::NewGuid().ToString('N'))
$bin = Join-Path $caseRoot 'bin'
$install = Join-Path $caseRoot 'repo'
$desktop = Join-Path $install 'apps/desktop'
$electron = Join-Path $install 'node_modules/electron'
foreach ($dir in @($bin, $desktop, $electron)) { [IO.Directory]::CreateDirectory($dir) | Out-Null }
$utf8 = [Text.UTF8Encoding]::new($true)
[IO.File]::WriteAllText((Join-Path $desktop 'package.json'), '{}')
[IO.File]::WriteAllText((Join-Path $electron 'install.js'), '// fixture')
[IO.File]::WriteAllText((Join-Path $electron 'package.json'), '{}')
$env:DEADLINE_CASE_ROOT = $caseRoot
$env:DEADLINE_INSTALL = $install
$env:GITHUB_SHA = '0000000000000000000000000000000000000001'
$env:GITHUB_REF_NAME = 'fixture'
$env:ELECTRON_MIRROR = $null
$env:PATH = $bin + ';' + $PSHOME
$env:TEMP = $caseRoot
$env:TMP = $caseRoot
[IO.File]::WriteAllText((Join-Path $bin 'npm-impl.ps1'), @'
$ErrorActionPreference = 'Stop'
$call = $args -join ' '
[IO.File]::AppendAllText((Join-Path $env:DEADLINE_CASE_ROOT 'calls.log'), $call + "`n")
[Console]::Out.WriteLine('native npm progress: ' + $call)
if (($call -eq 'ci' -and $env:DEADLINE_MODE -eq 'ci-hang') -or ($call -eq 'run pack' -and $env:DEADLINE_MODE -eq 'pack-hang')) {
    [IO.File]::AppendAllText((Join-Path $env:DEADLINE_CASE_ROOT 'owned.identities'), "$PID|$((Get-Process -Id $PID).StartTime.ToUniversalTime().Ticks)`n")
    [IO.File]::AppendAllText((Join-Path $env:DEADLINE_CASE_ROOT 'owned.pids'), [string]$PID + "`n")
    Start-Sleep -Seconds 12
    [IO.File]::WriteAllText((Join-Path $env:DEADLINE_CASE_ROOT 'late-completion'), 'expired command was allowed to continue')
    exit 99
}
if ($call -eq 'install') {
    foreach ($oldPid in [IO.File]::ReadAllLines((Join-Path $env:DEADLINE_CASE_ROOT 'owned.pids'))) {
        $previous = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue
        if ($previous -and -not $previous.HasExited) { throw 'previous npm attempt is still alive' }
    }
}
if ($call -eq 'run pack') {
    $out = Join-Path $env:DEADLINE_INSTALL 'apps/desktop/release/win-unpacked'
    [IO.Directory]::CreateDirectory($out) | Out-Null
    [IO.File]::WriteAllText((Join-Path $out 'Hermes.exe'), 'fixture artifact')
}
exit 0
'@, $utf8)
[IO.File]::WriteAllText((Join-Path $bin 'node-impl.ps1'), @'
$ErrorActionPreference = 'Stop'
[Console]::Out.WriteLine('native Electron progress')
[IO.File]::AppendAllText((Join-Path $env:DEADLINE_CASE_ROOT 'electron.calls'), [string]$env:ELECTRON_MIRROR + "`n")
if (-not $env:ELECTRON_MIRROR) {
    [IO.File]::AppendAllText((Join-Path $env:DEADLINE_CASE_ROOT 'owned.identities'), "$PID|$((Get-Process -Id $PID).StartTime.ToUniversalTime().Ticks)`n")
    [IO.File]::WriteAllText((Join-Path $env:DEADLINE_CASE_ROOT 'electron.pid'), [string]$PID)
    Start-Sleep -Seconds 60
    exit 99
}
$oldPid = [int][IO.File]::ReadAllText((Join-Path $env:DEADLINE_CASE_ROOT 'electron.pid'))
$previous = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
if ($previous -and -not $previous.HasExited) { throw 'old Electron writer survives into mirror retry' }
$dist = Join-Path $env:DEADLINE_INSTALL 'node_modules/electron/dist'
[IO.Directory]::CreateDirectory($dist) | Out-Null
[IO.File]::WriteAllText((Join-Path $dist 'electron.exe'), 'fixture electron')
exit 0
'@, $utf8)
$hostExe = (Get-Process -Id $PID).Path
foreach ($name in @('npm', 'node')) {
    $shim = '@"' + $hostExe + '" -NoProfile -ExecutionPolicy Bypass -File "%~dp0' + $name + '-impl.ps1" %*'
    [IO.File]::WriteAllText((Join-Path $bin "$name.cmd"), $shim + "`r`n", [Text.Encoding]::ASCII)
}

try {
    if (-not $InstallerPath) { $InstallerPath = Join-Path $repoRoot 'scripts/install.ps1' }
    . $InstallerPath -HermesHome (Join-Path $caseRoot 'home') -InstallDir $install
    # These are unrelated host discovery/publication edges, never the process,
    # deadline, retry, output or filesystem seams under test.
    function Test-Node { return $true }
    function New-DesktopShortcuts { }
    function icacls { $global:LASTEXITCODE = 0 }
    function Clear-ElectronBuildCache { @() }
    # Native npm.cmd adds a second cold script host; allow it to start before
    # the intentional 12-second stall, including under concurrent CI load.
    $script:InstallerCommandTimeouts = @{ Desktop = 8; Electron = 8 }
    $env:DEADLINE_MODE = 'ci-hang'
    Install-Desktop
    if (Test-Path -LiteralPath (Join-Path $caseRoot 'late-completion')) { throw 'Desktop command outlived its deadline and wrote after expiration' }
    $calls = [IO.File]::ReadAllLines((Join-Path $caseRoot 'calls.log'))
    if (($calls -join ',') -ne 'ci,install,run pack') { throw "Wrong npm retry sequence: $calls" }
    Write-Host 'PASS: actual desktop npm timeout tears down before fallback and packs successfully'

    $env:DEADLINE_MODE = 'pack-hang'
    [IO.Directory]::CreateDirectory((Join-Path $electron 'dist')) | Out-Null
    [IO.File]::WriteAllText((Join-Path $electron 'dist/electron.exe'), 'existing fixture Electron')
    $env:ELECTRON_MIRROR = 'fixture-mirror-already-set'
    $failed = $false
    try { Install-Desktop } catch { $failed = $_.Exception.Message -match 'build failed \(exit 124\)' }
    if (-not $failed) { throw 'Timed-out packaging did not fail the actual desktop worker' }
    Write-Host 'PASS: actual packaging timeout reports failure despite a pre-existing product executable'

    $env:ELECTRON_MIRROR = $null
    Remove-Item -LiteralPath (Join-Path $electron 'dist/electron.exe') -Force
    if (-not (Try-RestoreElectronDist -InstallDir $install)) { throw 'Electron mirror recovery failed' }
    if ($env:ELECTRON_MIRROR) { throw 'Electron mirror environment leaked into the caller' }
    $electronCalls = [IO.File]::ReadAllLines((Join-Path $caseRoot 'electron.calls'))
    if ($electronCalls.Count -ne 2 -or $electronCalls[1] -ne $script:DesktopElectronFallbackMirror) { throw 'Wrong Electron mirror sequence' }
    Write-Host 'PASS: Electron mirror starts after the expired writer exits and restores environment'
} finally {
    foreach ($file in @('owned.identities')) {
        $path = Join-Path $caseRoot $file
        if (Test-Path -LiteralPath $path) {
            foreach ($identity in [IO.File]::ReadAllLines($path)) {
                $parts = $identity.Split('|')
                $owned = Get-Process -Id ([int]$parts[0]) -ErrorAction SilentlyContinue
                if ($owned -and $owned.StartTime.ToUniversalTime().Ticks -eq [long]$parts[1]) {
                    Stop-Process -InputObject $owned -Force -ErrorAction SilentlyContinue
                }
            }
        }
    }
}
