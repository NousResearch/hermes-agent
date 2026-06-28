param(
    [switch]$SkipRestart,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = Join-Path $RepoRoot "venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = (Get-Command py -ErrorAction Stop).Source
    $PythonArgs = @("-3")
} else {
    $PythonArgs = @()
}

function Invoke-HermesPython {
    param([string[]]$ScriptArgs)
    if ($PythonArgs.Count -gt 0) {
        & $PythonExe @PythonArgs @ScriptArgs
    } else {
        & $PythonExe @ScriptArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($ScriptArgs -join ' ')"
    }
}

Write-Host "[1/4] Applying operator stack config (Codex main + Grok Build sub)..."
$applyArgs = @("$RepoRoot\scripts\apply_operator_stack.py")
if ($DryRun) { $applyArgs += "--dry-run" }
Invoke-HermesPython -ScriptArgs $applyArgs

Write-Host "[2/4] Syncing social traces into Ebbinghaus memory (+ Obsidian when vault is available)..."
$socialArgs = @("$RepoRoot\sync_memory.py")
if ($DryRun) { $socialArgs += "--dry-run" }
Invoke-HermesPython -ScriptArgs $socialArgs

Write-Host "[3/4] Syncing git memory vault (encrypted Ebbinghaus + brain docs)..."
$vaultArgs = @("$RepoRoot\scripts\memory\memory_vault_sync.py")
if ($DryRun) { $vaultArgs += "--dry-run" }
Invoke-HermesPython -ScriptArgs $vaultArgs

if (-not $SkipRestart) {
    Write-Host "[4/4] Restarting Hermes stack via UAC script..."
    & (Join-Path $ScriptDir "restart-hermes-autostart-admin.ps1")
} else {
    Write-Host "[4/4] Skipped restart (-SkipRestart)."
}

Write-Host "Hakua operator stack applied."
