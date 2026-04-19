$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$homeTarget = Join-Path $repoRoot ".hermes-home"
$homeLink = Join-Path $HOME ".hermes"
$hermesExe = Join-Path $repoRoot ".venv\Scripts\hermes.exe"
$nodeExe = "C:\Program Files\nodejs\node.exe"
$uiEntry = "C:\Users\user\tools\hermes-web-ui\node_modules\hermes-web-ui\bin\hermes-web-ui.mjs"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HERMES_BIN = $hermesExe
$env:Path = "C:\Program Files\Git\cmd;C:\Program Files\nodejs;$env:Path"

if (-not (Test-Path -LiteralPath $hermesExe)) {
  throw "Hermes executable not found: $hermesExe"
}

if (-not (Test-Path -LiteralPath $nodeExe)) {
  throw "Node.js executable not found: $nodeExe"
}

if (-not (Test-Path -LiteralPath $uiEntry)) {
  throw "hermes-web-ui entrypoint not found: $uiEntry"
}

if (-not (Test-Path -LiteralPath $homeLink)) {
  New-Item -ItemType Junction -Path $homeLink -Target $homeTarget | Out-Null
}

Set-Location $repoRoot
& $nodeExe $uiEntry start
