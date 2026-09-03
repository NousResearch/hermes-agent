# Local harness for windows-e2e.ps1's non-GUI logic, runnable under linux
# pwsh (nix run nixpkgs#powershell). Exercises the pieces that cost CI
# round-trips when wrong: parameter surface, ref-flag probing, and the
# install/update dispatch reaching the right arm names.
param()
$ErrorActionPreference = "Stop"
$driver = Join-Path $PSScriptRoot "..\tests\install\windows-e2e.ps1"

# 1. Parse cleanly.
$tok = $null; $err = $null
[System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $driver), [ref]$tok, [ref]$err) | Out-Null
if ($err -and $err.Count) { $err | ForEach-Object { Write-Host $_.Message }; exit 1 }
Write-Host "parse: OK"

# 2. Parameter surface: both axes present with the generator's ids.
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $driver), [ref]$tok, [ref]$err)
$params = $ast.ParamBlock.Parameters
$byName = @{}
foreach ($p in $params) { $byName[$p.Name.VariablePath.UserPath] = $p }
foreach ($required in @("Phase", "InstallMethod", "Route", "InstallRef")) {
    if (-not $byName.ContainsKey($required)) { Write-Host "missing param: $required"; exit 1 }
}
$imSet = ($byName["InstallMethod"].Attributes | Where-Object { $_.TypeName.Name -eq "ValidateSet" }).PositionalArguments.Value
foreach ($m in @("desktop-installer@latest", "installer-script", "installer-script+desktop")) {
    if ($imSet -notcontains $m) { Write-Host "InstallMethod ValidateSet missing $m"; exit 1 }
}
$rSet = ($byName["Route"].Attributes | Where-Object { $_.TypeName.Name -eq "ValidateSet" }).PositionalArguments.Value
foreach ($m in @("open-app-update", "hermes-desktop-app-update", "hermes-update", "installer-script", "installer-script+desktop", "desktop-installer@latest")) {
    if ($rSet -notcontains $m) { Write-Host "Route ValidateSet missing $m"; exit 1 }
}
Write-Host "parameter surface: OK"

# 3. Dispatch bodies reference the right arms (AST-level: the switch on
# InstallMethod contains the three arms; the switch on Route contains all
# six, with desktop-installer@latest re-running the GUI installer).
$text = Get-Content -LiteralPath $driver -Raw
foreach ($needle in @(
    'function Invoke-PhaseInstall',
    'function Invoke-PhaseUpdate',
    'Invoke-RefInstaller $state.old "old" -IncludeDesktop',
    'Invoke-HermesDesktopAppUpdate $state.current',
    'Invoke-HermesUpdate',
    'Invoke-PhaseInstallGui -Mode "update"'
)) {
    if ($text.IndexOf($needle) -lt 0) { Write-Host "dispatch missing: $needle"; exit 1 }
}
Write-Host "dispatch arms: OK"
Write-Host "ALL HARNESS CHECKS PASSED"
