param(
    [ValidateSet("rtx5060ti", "rtx3080", "rtx3060")]
    [string]$GpuProfile = "rtx5060ti",
    [string]$ServerExe = "",
    [string]$ModelPath = "",
    [int]$Port = 8080,
    [int]$ContextSize = 0,
    [ValidateSet("f16v_turbo4", "f16v_q4_0", "turbo4", "q4_0")]
    [string]$KvProfile = "f16v_turbo4",
    [ValidateSet("ngram-mod", "mtp", "none")]
    [string]$SpecType = "ngram-mod",
    [int]$SpecNgramMatch = 24,
    [int]$SpecNgramMin = 48,
    [int]$SpecNgramMax = 64,
    [int]$SpecDraftNMax = 3,
    [double]$SpecDraftPMin = 0.75,
    [int]$WaitSeconds = 180
)

$ErrorActionPreference = "Stop"

function Resolve-LlamaFallbackDefaults {
    param(
        [string]$ServerExe,
        [string]$ModelPath
    )

    if (-not $ServerExe) {
        if ($env:HERMES_LLAMA_SERVER_EXE) {
            $ServerExe = $env:HERMES_LLAMA_SERVER_EXE
        } else {
            $ServerExe = Join-Path $env:LOCALAPPDATA "Programs\llama-turboquant\bin\llama-server.exe"
        }
    }

    if (-not $ModelPath) {
        $ModelPath = $env:HERMES_LLAMA_GGUF_PATH
    }

    if (-not $ModelPath) {
        $ModelPath = $env:HERMES_LLAMA_MODEL_PATH
    }

    return @{
        ServerExe = $ServerExe
        ModelPath = $ModelPath
    }
}

function Resolve-DefaultContextSize {
    param([string]$Profile)

    switch ($Profile) {
        "rtx5060ti" { return 65536 }
        "rtx3080" { return 49152 }
        "rtx3060" { return 65536 }
        default { return 65536 }
    }
}

function Resolve-KvProfile {
    param([string]$Profile)

    switch ($Profile) {
        "f16v_turbo4" { return @{ K = "f16"; V = "turbo4" } }
        "f16v_q4_0" { return @{ K = "f16"; V = "q4_0" } }
        "turbo4" { return @{ K = "turbo4"; V = "turbo4" } }
        "q4_0" { return @{ K = "q4_0"; V = "q4_0" } }
    }
}

if ($ContextSize -le 0) {
    $ContextSize = Resolve-DefaultContextSize -Profile $GpuProfile
}

function Import-HermesLlamaEnv {
    $dotEnv = Join-Path $env:USERPROFILE ".hermes\.env"
    if (-not (Test-Path -LiteralPath $dotEnv)) { return }
    Get-Content -LiteralPath $dotEnv | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith('#')) { return }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { return }
        $key = $line.Substring(0, $eq).Trim()
        if ($key -notin @('HERMES_LLAMA_MODEL_PATH', 'HERMES_LLAMA_GGUF_PATH', 'HERMES_LLAMA_SERVER_EXE')) { return }
        if (-not [string]::IsNullOrWhiteSpace((Get-Item -Path "Env:$key" -ErrorAction SilentlyContinue).Value)) { return }
        $value = $line.Substring($eq + 1).Trim().Trim('"').Trim("'")
        if ($value) { Set-Item -Path "Env:$key" -Value $value }
    }
}

Import-HermesLlamaEnv
$resolved = Resolve-LlamaFallbackDefaults -ServerExe $ServerExe -ModelPath $ModelPath
$ServerExe = $resolved.ServerExe
$ModelPath = $resolved.ModelPath
if (-not (Test-Path -LiteralPath $ServerExe)) {
    throw "llama-server not found: $ServerExe"
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "fallback model not found: $ModelPath"
}

$existing = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" } |
    Select-Object -First 1

if ($existing) {
    Write-Output "llama.cpp fallback already listening on port $Port (pid=$($existing.OwningProcess))."
    exit 0
}

$kv = Resolve-KvProfile -Profile $KvProfile
$logDir = Join-Path $env:USERPROFILE ".hermes\logs\llama-fallback"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $logDir "llama-fallback-$stamp.out.log"
$stderrPath = Join-Path $logDir "llama-fallback-$stamp.err.log"

$serverArgs = @(
    "--model", $ModelPath,
    "--host", "127.0.0.1",
    "--port", [string]$Port,
    "--ctx-size", [string]$ContextSize,
    "--n-gpu-layers", "all",
    "--flash-attn", "on",
    "--cache-type-k", $kv.K,
    "--cache-type-v", $kv.V,
    "--parallel", "1",
    "--batch-size", "2048",
    "--ubatch-size", "512",
    "--reasoning", "off",
    "--reasoning-budget", "0",
    "--jinja",
    "--cont-batching"
)

if ($SpecType -eq "ngram-mod") {
    $serverArgs += @(
        "--spec-type", "ngram-mod",
        "--spec-ngram-mod-n-match", [string]$SpecNgramMatch,
        "--spec-ngram-mod-n-min", [string]$SpecNgramMin,
        "--spec-ngram-mod-n-max", [string]$SpecNgramMax
    )
}
elseif ($SpecType -eq "mtp") {
    $serverArgs += @(
        "--spec-type", "mtp",
        "--spec-draft-n-max", [string]$SpecDraftNMax,
        "--spec-draft-p-min", [string]$SpecDraftPMin
    )
}

$process = Start-Process `
    -FilePath $ServerExe `
    -ArgumentList $serverArgs `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$modelsUrl = "http://127.0.0.1:$Port/v1/models"

while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        $stderrTail = ""
        if (Test-Path -LiteralPath $stderrPath) {
            $stderrTail = (Get-Content -LiteralPath $stderrPath -Tail 80) -join "`n"
        }
        throw "llama-server exited during startup (exit=$($process.ExitCode)). stderr tail:`n$stderrTail"
    }

    try {
        $models = Invoke-RestMethod -Uri $modelsUrl -TimeoutSec 3
        Write-Output "llama.cpp fallback ready on $modelsUrl"
        Write-Output "pid=$($process.Id)"
        Write-Output "gpu_profile=$GpuProfile"
        Write-Output "model=$ModelPath"
        Write-Output "kv_profile=$KvProfile cache_type_k=$($kv.K) cache_type_v=$($kv.V)"
        Write-Output "spec_type=$SpecType"
        Write-Output "stdout=$stdoutPath"
        Write-Output "stderr=$stderrPath"
        $models | ConvertTo-Json -Depth 8
        exit 0
    } catch {
        Start-Sleep -Seconds 2
    }
}

throw "llama-server did not become ready within $WaitSeconds seconds. stdout=$stdoutPath stderr=$stderrPath"
