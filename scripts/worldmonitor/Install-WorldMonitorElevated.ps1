#Requires -Version 5.1
<#
.SYNOPSIS
  World Monitor デスクトップを管理者権限（UAC）でインストールし、sidecar を起動して Hermes と接続する。

.DESCRIPTION
  1) 未管理者なら自身を RunAs で再実行（UAC プロンプト）
  2) MSI をダウンロードしてサイレントインストール
  3) インストール先を検出し sidecar を起動（ポート 46123）
  4) ユーザー権限で hermes worldmonitor-osint setup-auth --mode sidecar

.PARAMETER Version
  World Monitor リリースタグ（既定: v2.5.23）

.PARAMETER SkipMsi
  MSI インストールをスキップ（既存のポータブル展開のみ使う）

.PARAMETER SkipHermes
  Hermes setup-auth をスキップ

.EXAMPLE
  .\Install-WorldMonitorElevated.ps1
#>
[CmdletBinding()]
param(
    [string] $Version = 'v2.5.23',
    [switch] $SkipMsi,
    [switch] $SkipHermes,
    [string] $HermesRepoRoot = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if (-not $HermesRepoRoot) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $HermesRepoRoot = (Resolve-Path (Join-Path $scriptDir '..\..')).Path
}

$CacheDir = Join-Path $env:LOCALAPPDATA 'WorldMonitorInstall'
$LogFile = Join-Path $CacheDir 'install-elevated.log'
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

function Write-Log([string] $Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
    Write-Host $line
}

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Find-WorldMonitorRoot {
    $pf86 = $env:ProgramFilesx86
    if (-not $pf86) { $pf86 = ${env:ProgramFiles(x86)} }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\World Monitor'),
        (Join-Path $env:ProgramFiles 'World Monitor'),
        (Join-Path $pf86 'World Monitor')
    )
    foreach ($dir in $candidates) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        $sidecar = Join-Path $dir 'sidecar\local-api-server.mjs'
        if (Test-Path -LiteralPath $sidecar) {
            return $dir
        }
    }
    $uninstallRoots = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*'
    )
    foreach ($pattern in $uninstallRoots) {
        $items = Get-ItemProperty $pattern -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -match 'World\s*Monitor' }
        foreach ($item in $items) {
            $loc = $item.InstallLocation
            if (-not $loc) { continue }
            if (-not (Test-Path -LiteralPath $loc)) { continue }
            $sidecar = Join-Path $loc 'sidecar\local-api-server.mjs'
            if (Test-Path -LiteralPath $sidecar) { return $loc }
        }
    }
    return $null
}

function Get-NodeExe([string] $WmRoot) {
    $bundled = Join-Path $WmRoot 'sidecar\node\node.exe'
    if (Test-Path -LiteralPath $bundled) { return $bundled }
    $cmd = Get-Command node -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    throw "node.exe not found (bundled or PATH)"
}

function Start-WorldMonitorSidecar([string] $WmRoot, [int] $Port = 46123) {
    $existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Log "Sidecar already listening on port $Port"
        return $true
    }

    $node = Get-NodeExe -WmRoot $WmRoot
    $mjs = Join-Path $WmRoot 'sidecar\local-api-server.mjs'
    if (-not (Test-Path -LiteralPath $mjs)) {
        throw "Sidecar script missing: $mjs"
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $node
    $psi.Arguments = "`"$mjs`""
    $psi.WorkingDirectory = $WmRoot
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.Environment['LOCAL_API_RESOURCE_DIR'] = $WmRoot
    $psi.Environment['LOCAL_API_PORT'] = [string] $Port
    $null = [System.Diagnostics.Process]::Start($psi)
    Write-Log "Started sidecar (node=$node, root=$WmRoot, port=$Port)"

    $deadline = (Get-Date).AddSeconds(45)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        $probeUrl = "http://127.0.0.1:$Port/api/news/v1/list-feed-digest?variant=full&lang=en"
        try {
            $resp = Invoke-WebRequest -Uri $probeUrl -UseBasicParsing -TimeoutSec 10
            if ($resp.StatusCode -eq 200) {
                Write-Log "Sidecar HTTP probe OK on port $Port"
                return $true
            }
        }
        catch {
            $code = [int]$_.Exception.Response.StatusCode
            if ($code -in 200, 401, 403) {
                Write-Log "Sidecar reachable (HTTP $code) on port $Port"
                return $true
            }
        }
    }
    throw "Sidecar did not become ready on port $Port within 45s"
}

function Install-WorldMonitorMsi([string] $Tag) {
    $ver = $Tag.TrimStart('v')
    $msiName = "World.Monitor_${ver}_x64_en-US.msi"
    $msiUrl = "https://github.com/koala73/worldmonitor/releases/download/$Tag/$msiName"
    $msiPath = Join-Path $CacheDir $msiName

    if (-not (Test-Path -LiteralPath $msiPath)) {
        Write-Log "Downloading $msiUrl"
        Invoke-WebRequest -Uri $msiUrl -OutFile $msiPath -UseBasicParsing
    }
    else {
        Write-Log "Using cached MSI: $msiPath"
    }

    Write-Log "Running msiexec /i (silent, elevated)"
    $args = @('/i', "`"$msiPath`"", '/qn', '/norestart', '/L*v', "`"$(Join-Path $CacheDir 'msi-install.log')`"")
    $proc = Start-Process -FilePath 'msiexec.exe' -ArgumentList $args -Wait -PassThru -NoNewWindow
    if ($proc.ExitCode -ne 0) {
        throw "msiexec failed with exit code $($proc.ExitCode). See $(Join-Path $CacheDir 'msi-install.log')"
    }
    Write-Log "MSI install completed (exit 0)"
}

function Invoke-HermesSidecarSetup([string] $RepoRoot) {
    Push-Location $RepoRoot
    try {
        Write-Log 'Running: py -3 -m hermes_cli.main worldmonitor-osint setup-auth --mode sidecar'
        & py -3 -m hermes_cli.main worldmonitor-osint setup-auth --mode sidecar
        if ($LASTEXITCODE -ne 0) { throw "hermes setup-auth exit $LASTEXITCODE" }
        Write-Log 'Running: py -3 -m hermes_cli.main worldmonitor-osint status'
        & py -3 -m hermes_cli.main worldmonitor-osint status
    }
    finally {
        Pop-Location
    }
}

# --- メイン ---
Write-Log "=== Install-WorldMonitorElevated start (admin=$(Test-IsAdmin)) ==="

if (-not $SkipMsi -and -not (Test-IsAdmin)) {
    Write-Log 'Relaunching with RunAs (UAC prompt)...'
    $argList = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', "`"$PSCommandPath`"",
        '-Version', $Version
    )
    if ($SkipHermes) { $argList += '-SkipHermes' }
    $argList += '-HermesRepoRoot', "`"$HermesRepoRoot`""
    Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList $argList -Wait
    Write-Log 'Elevated child finished; continuing as user for Hermes + sidecar check'

    $wmRoot = Find-WorldMonitorRoot
    if (-not $wmRoot) {
        Write-Log 'WARN: install dir not found after elevation; trying cached portable path'
        $wmRoot = Join-Path $env:LOCALAPPDATA 'Programs\World Monitor'
    }
    if (Test-Path -LiteralPath (Join-Path $wmRoot 'sidecar\local-api-server.mjs')) {
        Start-WorldMonitorSidecar -WmRoot $wmRoot | Out-Null
    }
    if (-not $SkipHermes) {
        Invoke-HermesSidecarSetup -RepoRoot $HermesRepoRoot
    }
    Write-Log '=== Done (orchestrator after UAC child) ==='
    exit 0
}

if (-not $SkipMsi) {
    if (-not (Test-IsAdmin)) {
        throw 'MSI install requires administrator. Re-run without -SkipMsi to trigger UAC.'
    }
    Install-WorldMonitorMsi -Tag $Version
}

$wmRoot = Find-WorldMonitorRoot
if (-not $wmRoot) {
    throw 'World Monitor install directory not found after MSI. Check msi-install.log'
}
Write-Log "World Monitor root: $wmRoot"

Start-WorldMonitorSidecar -WmRoot $wmRoot | Out-Null

if (-not $SkipHermes) {
    # 昇格セッションから Hermes を叩くと HERMES_HOME がずれることがあるので、ユーザーへ委譲
    if (Test-IsAdmin) {
        Write-Log 'Skipping Hermes in elevated session (run orchestrator pass as normal user)'
    }
    else {
        Invoke-HermesSidecarSetup -RepoRoot $HermesRepoRoot
    }
}

Write-Log '=== Install-WorldMonitorElevated complete ==='
