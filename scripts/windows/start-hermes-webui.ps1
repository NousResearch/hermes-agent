param(
    [string]$WebUiRoot = "",
    [string]$AgentRoot = "",
    [int]$Port = 8787
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $AgentRoot) {
    $AgentRoot = $env:HERMES_WEBUI_AGENT_DIR
    if (-not $AgentRoot) { $AgentRoot = $RepoRoot }
}
if (-not $WebUiRoot) {
    $WebUiRoot = $env:HERMES_WEBUI_ROOT
    if (-not $WebUiRoot) {
        $WebUiRoot = Join-Path $env:USERPROFILE "Desktop\hermes-webui"
    }
}

$ExampleEnv = Join-Path $AgentRoot "config\hermes-webui.env.example"
$TargetEnv = Join-Path $WebUiRoot ".env"

if (-not (Test-Path -LiteralPath (Join-Path $WebUiRoot "bootstrap.py"))) {
    Write-Error "Hermes WebUI not found at $WebUiRoot"
}

if (-not (Test-Path -LiteralPath (Join-Path $AgentRoot "run_agent.py"))) {
    Write-Error "Hermes agent checkout not found at $AgentRoot"
}

if (-not (Test-Path -LiteralPath $TargetEnv)) {
    if (-not (Test-Path -LiteralPath $ExampleEnv)) {
        Write-Error "Missing template: $ExampleEnv"
    }
    Copy-Item -LiteralPath $ExampleEnv -Destination $TargetEnv
    Write-Host "Created $TargetEnv from upstream-sync template."
}

$env:HERMES_WEBUI_AGENT_DIR = $AgentRoot
$env:HERMES_WEBUI_PORT = "$Port"

$StartScript = Join-Path $WebUiRoot "start.ps1"
if (-not (Test-Path -LiteralPath $StartScript)) {
    Write-Error "Missing start.ps1 in $WebUiRoot"
}

Write-Host "Starting Hermes WebUI on http://127.0.0.1:$Port (agent: $AgentRoot)"
& $StartScript @args
