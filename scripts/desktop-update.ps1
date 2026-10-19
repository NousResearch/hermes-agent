# COMPAT FORWARDER — do not add logic here.
#
# The hand-off moved to scripts/desktop-update/windows.ps1. This forwarder
# exists for exactly one consumer: an already-installed Desktop whose asar
# is one update behind and still spawns scripts/desktop-update.ps1 (see
# resolveUpdateScriptHandoff in apps/desktop/electron/updater-process.ts).
# Without it, that Desktop would silently fall back to the frozen staged
# Tauri binary for one update cycle — the exact rot this script family
# exists to escape.
$target = Join-Path $PSScriptRoot "desktop-update\windows.ps1"
if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    Write-Error "The maintained Desktop updater is missing. Repair the Hermes installation before updating."
    exit 3
}
& $target @args
exit $LASTEXITCODE
