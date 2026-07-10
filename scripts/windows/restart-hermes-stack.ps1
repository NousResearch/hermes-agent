# Idempotent Hermes stack restart: gateway/harness/webui/dashboard.
# Pass -StartLlama only for rollback/recovery checks that need the local GGUF server.
param(
    [switch]$SkipTunnels,
    [switch]$StartLlama,
    [int]$WaitModelsSeconds = 300
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
}
$HermesHome = Join-Path $env:USERPROFILE ".hermes"
$DesiredModel = "yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M"
$TailscaleScript = Join-Path $env:LOCALAPPDATA "HermesWebUI\Update-HermesTailscaleServe.ps1"
$RepoTailscaleScript = Join-Path $PSScriptRoot "Update-HermesTailscaleServe.ps1"
$LlamaNgrokScript = Join-Path $env:LOCALAPPDATA "HermesWebUI\Start-HermesLlamaNgrok.ps1"
$LineNgrokScript = Join-Path $env:LOCALAPPDATA "HermesWebUI\Start-HermesLineNgrok.ps1"
$WebUiScript = Join-Path $env:LOCALAPPDATA "HermesWebUI\Start-HermesWebUI.ps1"
$MemoryGraphScript = Join-Path $PSScriptRoot "start-obsidian-memory-graph-server.ps1"

function Write-Step([string]$Message) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message)
}

function Stop-PortListener {
    param([int]$Port, [string]$NamePattern = ".*")
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $conn) { return }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)" -ErrorAction SilentlyContinue
    if (-not $proc) { return }
    if ($proc.CommandLine -and $proc.CommandLine -match $NamePattern) {
        Write-Step "Stopping $Port pid=$($proc.ProcessId) name=$($proc.Name)"
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}

Set-Location -LiteralPath $ProjectRoot
$env:HERMES_HOME = $HermesHome
$env:HF_HUB_CACHE = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } else { "H:\elt_data\hf-cache" }

Write-Step "Stopping Hermes services (gateway/harness/webui/dashboard)"
try { & $PythonExe -m hermes_cli.main gateway stop --all 2>$null } catch {}
try { & $PythonExe -m hermes_cli.main harness stop 2>$null } catch {}
Stop-PortListener -Port 8787 -NamePattern "server\.py|hermes"
Stop-PortListener -Port 9120 -NamePattern "hermes_cli\.main dashboard|dashboard"
if ($StartLlama) {
    Stop-PortListener -Port 8080 -NamePattern "llama-server"
} else {
    Write-Step "Skipping llama restart; pass -StartLlama only for rollback/recovery checks"
}

if ($StartLlama) {
    Write-Step "Starting llama secretary on :8080 (H: HF cache)"
    & (Join-Path $PSScriptRoot "start-llama-secretary.ps1") -WaitSeconds $WaitModelsSeconds

    $modelsOk = $false
    $deadline = (Get-Date).AddSeconds($WaitModelsSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $models = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/models" -TimeoutSec 8
            $ids = @($models.data | ForEach-Object { $_.id })
            Write-Step ("8080 models: {0}" -f ($ids -join ", "))
            if ($ids -contains $DesiredModel) {
                $modelsOk = $true
                break
            }
            if ($ids.Count -gt 0) {
                Write-Warning "Desired model not listed yet; continuing to wait"
            }
        } catch {
            Write-Step ("Waiting for llama /v1/models: {0}" -f $_.Exception.Message)
        }
        Start-Sleep -Seconds 5
    }
    if (-not $modelsOk) {
        throw "llama /v1/models did not expose $DesiredModel within ${WaitModelsSeconds}s"
    }
}

if (-not $SkipTunnels) {
    if (Test-Path -LiteralPath $MemoryGraphScript) {
        Write-Step "Ensuring Obsidian memory-graph server (:8765)"
        & $MemoryGraphScript
    }

    if (Test-Path -LiteralPath $RepoTailscaleScript) {
        Copy-Item -LiteralPath $RepoTailscaleScript -Destination $TailscaleScript -Force
    }

    if (Test-Path -LiteralPath $TailscaleScript) {
        Write-Step "Updating Tailscale serve (/ /line /v1 /memory-graph)"
        if ($StartLlama) {
            & $TailscaleScript -LlamaPort 8080
        } else {
            & $TailscaleScript
        }
    } else {
        Write-Warning "Missing Tailscale script: $TailscaleScript"
    }

    if (Test-Path -LiteralPath $LineNgrokScript) {
        Write-Step "Ensuring LINE ngrok (:8646)"
        & $LineNgrokScript
    }
    if ($StartLlama -and (Test-Path -LiteralPath $LlamaNgrokScript)) {
        Write-Step "Ensuring llama ngrok (:8080)"
        & $LlamaNgrokScript -LlamaPort 8080
    }
}

Write-Step "Starting gateway"
& (Join-Path $PSScriptRoot "start-hermes-gateway.ps1") -StartLlama:$StartLlama

Write-Step "Starting harness"
Start-Process -FilePath $PythonExe -ArgumentList @("-m", "hermes_cli.main", "harness", "start") -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null
Start-Sleep -Seconds 4

if (Test-Path -LiteralPath $WebUiScript) {
    Write-Step "Starting WebUI"
    & $WebUiScript
}
& (Join-Path $PSScriptRoot "start-hermes-dashboard.ps1") -HermesRoot $ProjectRoot -HermesHome $HermesHome

Write-Step "Health checks"
& $PythonExe -m hermes_cli.main gateway status
& $PythonExe -m hermes_cli.main harness status
Invoke-RestMethod http://127.0.0.1:8787/health -TimeoutSec 10 | Out-Null
(Invoke-WebRequest http://127.0.0.1:9120/ -UseBasicParsing -TimeoutSec 10).StatusCode | Out-Null
Write-Step "Hermes stack restart complete"
