param(
    [string]$InstallDir = "$env:USERPROFILE\.local\bin"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$shimTarget = Join-Path $repoRoot "scripts\xurl_windows.py"
if (-not (Test-Path -LiteralPath $shimTarget)) {
    throw "Missing shim target: $shimTarget"
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pyCommand = Get-Command py -ErrorAction SilentlyContinue
if (-not $pythonCommand -and -not $pyCommand) {
    throw "Python was not found on PATH. Install Python or add it to PATH first."
}

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
$cmdPath = Join-Path $InstallDir "xurl.cmd"

$pythonLine = if ($pythonCommand) {
    'python "%HERMES_XURL_WINDOWS_SHIM%" %*'
} else {
    'py -3 "%HERMES_XURL_WINDOWS_SHIM%" %*'
}

$content = @"
@echo off
set "HERMES_XURL_WINDOWS_SHIM=$shimTarget"
$pythonLine
"@

Set-Content -LiteralPath $cmdPath -Value $content -Encoding ASCII

$pathEntries = [Environment]::GetEnvironmentVariable("Path", "User") -split ';' | Where-Object { $_ }
if ($pathEntries -notcontains $InstallDir) {
    [Environment]::SetEnvironmentVariable("Path", (($pathEntries + $InstallDir) -join ';'), "User")
    Write-Output "Installed xurl shim to $cmdPath"
    Write-Output "Added $InstallDir to the user PATH. Open a new terminal if this shell does not see xurl yet."
} else {
    Write-Output "Installed xurl shim to $cmdPath"
}
