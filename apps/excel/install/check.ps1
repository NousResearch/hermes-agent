[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$failures = New-Object System.Collections.Generic.List[string]

Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.ps1' -File | ForEach-Object {
  $tokens = $null
  $errors = $null
  [void][System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$tokens, [ref]$errors)
  foreach ($error in $errors) {
    $failures.Add("$($_.Name): $($error.Message)")
  }
}

$manifest = [xml](Get-Content -LiteralPath (Join-Path $root 'manifest.xml') -Raw)
if ($manifest.OfficeApp.Id -ne '4fd4d435-7f9a-4d6d-9251-32f154f83a1f') {
  $failures.Add('manifest.xml has an unexpected or missing add-in ID.')
}

foreach ($relative in @(
  'broker\server.mjs',
  'taskpane.html',
  'taskpane.css',
  'taskpane.js',
  'assets\icon-16.png',
  'assets\icon-32.png',
  'assets\icon-64.png',
  'assets\icon-80.png',
  'service\bridge-service.cmd',
  'install\run-bridge.cmd.template',
  'install\run-bridge.vbs.template'
)) {
  if (-not (Test-Path -LiteralPath (Join-Path $root $relative))) {
    $failures.Add("Required payload file is missing: $relative")
  }
}

$taskScript = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'register-task.ps1') -Raw
if ($taskScript -notmatch 'shell\.Run\s+"""\$ServiceCmd""",\s*0,\s*True') {
  $failures.Add('Scheduled Task shim must wait for the supervisor (bWaitOnReturn=True).')
}

$installer = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'addin-install.ps1') -Raw
if ($installer -notmatch "service\s+-eq\s+'hermes-excel-bridge'") {
  $failures.Add('Installer health check must validate the bridge service identity.')
}

$pane = Get-Content -LiteralPath (Join-Path $root 'taskpane.html') -Raw
if ($pane -notmatch '<input[^>]+id="reviewToggle"[^>]+checked') {
  $failures.Add('Review-before-apply must be visibly enabled in the initial HTML.')
}

if ($failures.Count) {
  $failures | ForEach-Object { Write-Error $_ }
  exit 1
}

Write-Host "Excel add-in install/package checks passed."
