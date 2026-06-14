# Local secretary runtime — llama.cpp primary launcher (RTX 3060 profile)
# Starts Qwen3.5-9B via HuggingFace repo id with --jinja for tool calling.

param(
    [switch]$SkipFallbackOnFailure,
    [int]$WaitSeconds = 240
)

$ErrorActionPreference = "Stop"

function Import-HermesDotEnvKeys {
    $dotEnv = Join-Path $env:USERPROFILE ".hermes\.env"
    if (-not (Test-Path -LiteralPath $dotEnv)) { return }
    Get-Content -LiteralPath $dotEnv | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith('#')) { return }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { return }
        $key = $line.Substring(0, $eq).Trim().Trim([char]0xFEFF)
        if ($key -notlike 'HERMES_LLAMA_*') { return }
        if (-not [string]::IsNullOrWhiteSpace((Get-Item -Path "Env:$key" -ErrorAction SilentlyContinue).Value)) { return }
        $value = $line.Substring($eq + 1).Trim().Trim('"').Trim("'")
        if ($value) { Set-Item -Path "Env:$key" -Value $value }
    }
}

function Resolve-Default {
    param([string]$Name, [string]$Default)
    $fromEnv = [Environment]::GetEnvironmentVariable($Name)
    if (-not [string]::IsNullOrWhiteSpace($fromEnv)) { return $fromEnv }
    return $Default
}

function Get-LlamaHelpText {
    param([string]$ServerExe)
    $output = & $ServerExe --help 2>&1 | Out-String
    return $output
}

function Test-HelpFlag {
    param([string]$HelpText, [string]$Pattern)
    return ($HelpText -match $Pattern)
}

Import-HermesDotEnvKeys

$ServerExe = Resolve-Default "HERMES_LLAMA_SERVER_EXE" (Join-Path $env:LOCALAPPDATA "Programs\llama-turboquant\bin\llama-server.exe")
$ModelRepo = Resolve-Default "HERMES_LLAMA_MODEL" "unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL"
$Alias = Resolve-Default "HERMES_LLAMA_ALIAS" "qwen35-9b-secretary"
$HostName = Resolve-Default "HERMES_LLAMA_HOST" "127.0.0.1"
$Port = [int](Resolve-Default "HERMES_LLAMA_PORT" "8080")
$Ctx = [int](Resolve-Default "HERMES_LLAMA_CTX" "65536")
$CacheK = Resolve-Default "HERMES_LLAMA_CACHE_TYPE_K" "q4_0"
$CacheV = Resolve-Default "HERMES_LLAMA_CACHE_TYPE_V" "q4_0"
$SpecType = Resolve-Default "HERMES_LLAMA_SPEC_TYPE" "ngram-mod"
$SpecDraftNMax = [int](Resolve-Default "HERMES_LLAMA_SPEC_DRAFT_N_MAX" "64")
$Profile = Resolve-Default "HERMES_LLAMA_PROFILE" "rtx3060"

if ($Ctx -lt 64000) {
    throw "HERMES_LLAMA_CTX=$Ctx is below the local-secretary minimum of 64000. Set HERMES_LLAMA_CTX=65536."
}

if (-not (Test-Path -LiteralPath $ServerExe)) {
    throw "llama-server not found: $ServerExe (set HERMES_LLAMA_SERVER_EXE)"
}

$existing = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" } |
    Select-Object -First 1
if ($existing) {
    Write-Output "llama.cpp secretary already listening on port $Port (pid=$($existing.OwningProcess))."
    exit 0
}

$helpText = Get-LlamaHelpText -ServerExe $ServerExe
$supportsCacheK = Test-HelpFlag $helpText '--cache-type-k'
$supportsSpecType = Test-HelpFlag $helpText '--spec-type'
$supportsHfRepoLong = Test-HelpFlag $helpText '--hf-repo'
$supportsHfRepoShort = Test-HelpFlag $helpText '(^|[\s,])-hf([\s,]|$)'
$supportsHfRepo = $supportsHfRepoLong -or $supportsHfRepoShort

$logDir = Join-Path $env:USERPROFILE ".hermes\logs\llama-secretary"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $logDir "llama-secretary-$stamp.out.log"
$stderrPath = Join-Path $logDir "llama-secretary-$stamp.err.log"

$gpuLayerSteps = @(
    [int](Resolve-Default "HERMES_LLAMA_GPU_LAYERS" "99"),
    32, 24, 16, 8
) | Select-Object -Unique

function Build-ServerArgs {
    param(
        [int]$GpuLayers,
        [bool]$IncludeSpec,
        [bool]$IncludeCache
    )
    $args = @()
    if ($supportsHfRepo) {
        $hfFlag = if ($supportsHfRepoLong) { "--hf-repo" } else { "-hf" }
        $args += @($hfFlag, $ModelRepo)
    } else {
        throw "This llama-server build lacks -hf/--hf-repo; cannot load $ModelRepo"
    }
    $args += @(
        "--alias", $Alias,
        "--host", $HostName,
        "--port", [string]$Port,
        "--jinja",
        "-fa", "on",
        "-c", [string]$Ctx,
        "-ngl", [string]$GpuLayers,
        "-np", "1"
    )
    if ($IncludeCache -and $supportsCacheK) {
        $args += @("--cache-type-k", $CacheK, "--cache-type-v", $CacheV)
    }
    if ($IncludeSpec -and $supportsSpecType -and $SpecType -and $SpecType -ne "none") {
        $args += @("--spec-type", $SpecType)
        if ($SpecType -eq "ngram-mod") {
            if (Test-HelpFlag $helpText '--spec-ngram-mod-n-max') {
                $args += @("--spec-ngram-mod-n-max", [string]$SpecDraftNMax)
            }
        } elseif (Test-HelpFlag $helpText '--spec-draft-n-max') {
            $args += @("--spec-draft-n-max", [string]$SpecDraftNMax)
        }
    }
    return $args
}

function Test-OomInStderr {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $tail = (Get-Content -LiteralPath $Path -Tail 120 -ErrorAction SilentlyContinue) -join "`n"
    return ($tail -match '(?i)(out of memory|cuda error|OOM|failed to allocate|insufficient memory)')
}

function Start-LlamaAttempt {
    param(
        [int]$GpuLayers,
        [bool]$IncludeSpec,
        [bool]$IncludeCache
    )
    $attemptArgs = Build-ServerArgs -GpuLayers $GpuLayers -IncludeSpec $IncludeSpec -IncludeCache $IncludeCache
    $proc = Start-Process `
        -FilePath $ServerExe `
        -ArgumentList $attemptArgs `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru
    return @{ Process = $proc; Args = $attemptArgs }
}

$modelsUrl = "http://${HostName}:${Port}/v1/models"
$attemptPlans = @(
    @{ IncludeSpec = $true; IncludeCache = $true },
    @{ IncludeSpec = $true; IncludeCache = $false },
    @{ IncludeSpec = $false; IncludeCache = $false }
)

$started = $false
foreach ($plan in $attemptPlans) {
    if (-not $plan.IncludeCache -and $supportsCacheK) {
        Write-Warning "cache-type-k/v unsupported or disabled; falling back to default f16 KV cache."
    }
    if (-not $plan.IncludeSpec -and $supportsSpecType -and $SpecType -ne "none") {
        Write-Warning "spec-type '$SpecType' unsupported on this build; starting without speculative decoding."
    }
    foreach ($layers in $gpuLayerSteps) {
        if ($started) { break }
        $attempt = Start-LlamaAttempt -GpuLayers $layers -IncludeSpec $plan.IncludeSpec -IncludeCache $plan.IncludeCache
        $deadline = (Get-Date).AddSeconds($WaitSeconds)
        while ((Get-Date) -lt $deadline) {
            if ($attempt.Process.HasExited) {
                if (Test-OomInStderr -Path $stderrPath) {
                    Write-Warning "CUDA OOM at ngl=$layers; retrying with fewer GPU layers."
                    break
                }
                $stderrTail = (Get-Content -LiteralPath $stderrPath -Tail 80 -ErrorAction SilentlyContinue) -join "`n"
                throw "llama-server exited during startup (exit=$($attempt.Process.ExitCode)). stderr:`n$stderrTail"
            }
            try {
                $null = Invoke-RestMethod -Uri $modelsUrl -TimeoutSec 3
                $started = $true
                Write-Output "llama.cpp secretary ready on $modelsUrl"
                Write-Output "pid=$($attempt.Process.Id)"
                Write-Output "profile=$Profile model=$ModelRepo alias=$Alias ctx=$Ctx ngl=$layers"
                Write-Output "stdout=$stdoutPath"
                Write-Output "stderr=$stderrPath"
                exit 0
            } catch {
                Start-Sleep -Seconds 2
            }
        }
        if (-not $started -and -not $attempt.Process.HasExited) {
            Stop-Process -Id $attempt.Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not $SkipFallbackOnFailure) {
    Write-Warning "Primary secretary model failed to start; invoking Hermes-3 fallback launcher."
    $fallbackScript = Join-Path $PSScriptRoot "start-llama-secretary-fallback.ps1"
    if (Test-Path -LiteralPath $fallbackScript) {
        & $fallbackScript -WaitSeconds $WaitSeconds
        exit $LASTEXITCODE
    }
}

throw "Failed to start llama.cpp secretary on port $Port. See $stderrPath"
