# COMPAT FORWARDER — do not add logic here.
#
# The hand-off moved to scripts/desktop-update/windows.ps1. This forwarder
# exists for exactly one consumer: an already-installed Desktop whose asar
# is one update behind and still spawns scripts/desktop-update.ps1 (see
# resolveUpdateScriptHandoff in apps/desktop/electron/updater-process.ts).
# Without it, that Desktop would silently fall back to the frozen staged
# Tauri binary for one update cycle — the exact rot this script family
# exists to escape.
& (Join-Path $PSScriptRoot "desktop-update\windows.ps1") @args
exit $LASTEXITCODE
