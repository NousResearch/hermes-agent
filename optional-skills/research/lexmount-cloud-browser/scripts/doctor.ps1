$ErrorActionPreference = "Stop"
$architecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($architecture -ne "AMD64") {
  Write-Output '{"ok":false,"error":"unsupported_platform","message":"This Skill supports Windows x64 through scripts/doctor.ps1, and macOS arm64 and Linux x86_64 through scripts/doctor.sh."}'
  exit 2
}
$skillBinary = Join-Path (Split-Path -Parent $PSScriptRoot) "bin\browser-cli.exe"
if (Test-Path $skillBinary) { & $skillBinary doctor; exit $LASTEXITCODE }
Write-Output '{"ok":false,"error":"command_not_found","message":"Skill-local browser-cli.exe is missing. Run scripts/bootstrap.ps1 first."}'
exit 1
