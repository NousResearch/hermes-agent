param(
    [string]$HermesRoot = "",
    [string]$Cwd = "",
    [string]$HermesHome = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
. (Join-Path $ScriptDir "Resolve-CanonicalHermesHome.ps1")

if (-not $HermesRoot) {
    $HermesRoot = $RepoRoot
}
if (-not $Cwd) {
    $Cwd = $HermesRoot
}
$HermesHome = Resolve-CanonicalHermesHome -Preferred $HermesHome -RepoRoot $RepoRoot

$PythonExe = Join-Path $HermesRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = Join-Path $HermesRoot "venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

$env:HERMES_HOME = $HermesHome
$env:HERMES_DESKTOP_HERMES_ROOT = $HermesRoot
$env:HERMES_DESKTOP_CWD = $Cwd
$WebDist = Join-Path $HermesRoot "hermes_cli\web_dist"
if (Test-Path -LiteralPath (Join-Path $WebDist "index.html")) {
    $env:HERMES_DESKTOP_DASHBOARD_WEB_DIST = $WebDist
}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$staleDesktopPattern = [regex]::Escape("AppData\Local\hermes\hermes-agent\apps\desktop\release\win-unpacked\Hermes.exe")
$staleDesktopProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and $_.CommandLine -match $staleDesktopPattern
}
foreach ($proc in $staleDesktopProcesses) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}

$sourceElectronPattern = [regex]::Escape((Join-Path $HermesRoot "node_modules\electron\dist\electron.exe"))
$sourceDesktopRunning = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and $_.CommandLine -match $sourceElectronPattern
} | Select-Object -First 1
if ($sourceDesktopRunning) {
    exit 0
}

Set-Location -LiteralPath $HermesRoot
& $PythonExe -m hermes_cli.main desktop --source --skip-build --hermes-root $HermesRoot --cwd $Cwd
exit $LASTEXITCODE
