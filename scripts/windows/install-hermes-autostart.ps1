# One-command Hermes logon autostart + optional desktop shortcut refresh.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/install-hermes-autostart.ps1
#   powershell ... -File scripts/windows/install-hermes-autostart.ps1 -RefreshDesktopShortcuts
#   powershell ... -File scripts/windows/install-hermes-autostart.ps1 -Unregister

[CmdletBinding()]
param(
    [switch]$Unregister,
    [switch]$RefreshDesktopShortcuts,
    [switch]$IncludeLlama,
    [switch]$GatewayOnly,
    [switch]$IncludeLegacyStack
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$RegisterScript = Join-Path $ScriptDir "register-hermes-autostart.ps1"
$ShortcutsScript = Join-Path $RepoRoot "scripts\create-hermes-desktop-shortcuts.ps1"

if (-not (Test-Path -LiteralPath $RegisterScript)) {
    throw "Missing register script: $RegisterScript"
}

$registerArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RegisterScript)
if ($Unregister) { $registerArgs += "-Unregister" }
if ($IncludeLlama) { $registerArgs += "-IncludeLlama" }
if ($GatewayOnly) { $registerArgs += "-GatewayOnly" }
if ($IncludeLegacyStack) { $registerArgs += "-IncludeLegacyStack" }

& powershell.exe @registerArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($RefreshDesktopShortcuts -and -not $Unregister) {
    if (-not (Test-Path -LiteralPath $ShortcutsScript)) {
        throw "Missing shortcuts script: $ShortcutsScript"
    }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $ShortcutsScript
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "install-hermes-autostart.ps1 finished." -ForegroundColor Green
