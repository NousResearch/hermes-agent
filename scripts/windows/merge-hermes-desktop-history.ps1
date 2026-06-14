# Merge Hermes desktop history (legacy %LOCALAPPDATA%\hermes -> canonical ~/.hermes).
# Stop gateway/desktop before running to avoid state.db lock contention.
param(
    [switch]$DryRun,
    [string]$CanonicalHome = "",
    [string]$LegacyHome = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

$argsList = @("$RepoRoot\scripts\merge_hermes_desktop_history.py")
if ($DryRun) { $argsList += "--dry-run" }
if ($CanonicalHome) { $argsList += @("--canonical-home", $CanonicalHome) }
if ($LegacyHome) { $argsList += @("--legacy-home", $LegacyHome) }

Write-Host "Merging desktop history into canonical HERMES_HOME..."
py -3 @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $DryRun) {
    $target = if ($CanonicalHome) { $CanonicalHome } else { Join-Path $env:USERPROFILE ".hermes" }
    [Environment]::SetEnvironmentVariable("HERMES_HOME", $target, "User")
    Write-Host "Set user HERMES_HOME=$target (restart desktop apps to pick up)."
}
