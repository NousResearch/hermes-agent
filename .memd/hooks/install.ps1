param(
  [string]$Prefix = $(if ($env:MEMD_HOOK_PREFIX) { $env:MEMD_HOOK_PREFIX } else { Join-Path $HOME "bin" }),
  [string]$MemdBin = $(if ($env:MEMD_BIN) { $env:MEMD_BIN } else { "memd" })
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $Prefix | Out-Null

Copy-Item -Force (Join-Path $scriptDir "memd-context.ps1") (Join-Path $Prefix "memd-context.ps1")
Copy-Item -Force (Join-Path $scriptDir "memd-capture.ps1") (Join-Path $Prefix "memd-capture.ps1")
Copy-Item -Force (Join-Path $scriptDir "memd-spill.ps1") (Join-Path $Prefix "memd-spill.ps1")
Copy-Item -Force (Join-Path $scriptDir "memd-stop-save.ps1") (Join-Path $Prefix "memd-stop-save.ps1")
Copy-Item -Force (Join-Path $scriptDir "memd-precompact-save.ps1") (Join-Path $Prefix "memd-precompact-save.ps1")

$contextShim = @"
param([Parameter(ValueFromRemainingArguments = `$true)][string[]]`$Args)
& (Join-Path "$Prefix" "memd-context.ps1") @Args
"@

$spillShim = @"
param([Parameter(ValueFromRemainingArguments = `$true)][string[]]`$Args)
& "$MemdBin" hook spill @Args
"@

$captureShim = @"
param([Parameter(ValueFromRemainingArguments = `$true)][string[]]`$Args)
& (Join-Path "$Prefix" "memd-capture.ps1") @Args
"@

$stopSaveShim = @"
param([Parameter(ValueFromRemainingArguments = `$true)][string[]]`$Args)
& (Join-Path "$Prefix" "memd-stop-save.ps1") @Args
"@

$precompactSaveShim = @"
param([Parameter(ValueFromRemainingArguments = `$true)][string[]]`$Args)
& (Join-Path "$Prefix" "memd-precompact-save.ps1") @Args
"@

Set-Content -Path (Join-Path $Prefix "memd-hook-context.ps1") -Value $contextShim -Encoding UTF8
Set-Content -Path (Join-Path $Prefix "memd-hook-spill.ps1") -Value $spillShim -Encoding UTF8
Set-Content -Path (Join-Path $Prefix "memd-hook-capture.ps1") -Value $captureShim -Encoding UTF8
Set-Content -Path (Join-Path $Prefix "memd-hook-stop-save.ps1") -Value $stopSaveShim -Encoding UTF8
Set-Content -Path (Join-Path $Prefix "memd-hook-precompact-save.ps1") -Value $precompactSaveShim -Encoding UTF8

Write-Host "Installed memd hooks to $Prefix"
Write-Host "Add $Prefix to PATH if needed."
Write-Host "Set MEMD_BIN if the memd CLI is not already on PATH."
