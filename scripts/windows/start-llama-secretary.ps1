# Local secretary runtime — llama.cpp primary launcher (RTX 5060 Ti 16GB profile)
# Loads the configured model GGUF via local path or llama-server -hf/--hf-repo with --jinja for tool calling.
# RTX 5060 Ti 16GB optimized: ngl=99 (all layers on GPU), fa, q8_0 KV cache, 6 threads (physical cores),
# parallel=2 (Hermes + subagent concurrent requests), batch=2048, ubatch=512, ctx=131072.
# Hardware: RTX 5060 Ti 16GB (換装: 2026-07-22, 旧: RTX 3060 12GB)

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
        if ($key -notlike 'HERMES_LLAMA_*' -and $key -notin @('HF_HUB_CACHE', 'HF_HOME')) { return }
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

if (-not $env:HF_HUB_CACHE) {
    $env:HF_HUB_CACHE = "H:\elt_data\hf-cache"
}
New-Item -ItemType Directory -Path $env:HF_HUB_CACHE -Force | Out-Null

$ServerExe = Resolve-Default "HERMES_LLAMA_SERVER_EXE" (Join-Path $env:LOCALAPPDATA "Programs\llama-turboquant\bin\llama-server.exe")
$ModelRepo = Resolve-Default "HERMES_LLAMA_MODEL" "yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M"
$ModelPath = Resolve-Default "HERMES_LLAMA_GGUF_PATH" ""
$Alias = Resolve-Default "HERMES_LLAMA_ALIAS" "yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M"
$HostName = Resolve-Default "HERMES_LLAMA_HOST" "127.0.0.1"
$Port = [int](Resolve-Default "HERMES_LLAMA_PORT" "8080")
$Ctx = [int](Resolve-Default "HERMES_LLAMA_CTX" "131072")
$CacheK = Resolve-Default "HERMES_LLAMA_CACHE_TYPE_K" "q8_0"
# turbo3 = zapabob llama-turboquant custom KV type; falls back to q8_0 on standard builds via plan iteration
$CacheV = Resolve-Default "HERMES_LLAMA_CACHE_TYPE_V" "turbo3"
$SpecType = Resolve-Default "HERMES_LLAMA_SPEC_TYPE" "ngram-mod"
$SpecNgramMatch = [int](Resolve-Default "HERMES_LLAMA_SPEC_NGRAM_MATCH" "24")
$SpecNgramMin = [int](Resolve-Default "HERMES_LLAMA_SPEC_NGRAM_MIN" "48")
$SpecNgramMax = [int](Resolve-Default "HERMES_LLAMA_SPEC_NGRAM_MAX" "64")
$SpecDraftNMax = [int](Resolve-Default "HERMES_LLAMA_SPEC_DRAFT_N_MAX" "64")
$BatchSize = [int](Resolve-Default "HERMES_LLAMA_BATCH_SIZE" "2048")
$UbatchSize = [int](Resolve-Default "HERMES_LLAMA_UBATCH_SIZE" "512")
# Threads: physical core count (avoid HT contention when GPU handles inference)
$Threads = [int](Resolve-Default "HERMES_LLAMA_THREADS" "8")
# Parallel slots: 2 allows Hermes + subagent concurrent requests without queuing
$Parallel = [int](Resolve-Default "HERMES_LLAMA_PARALLEL" "2")
$Profile = Resolve-Default "HERMES_LLAMA_PROFILE" "rtx5060ti"

if ($Ctx -lt 64000) {
    throw "HERMES_LLAMA_CTX=$Ctx is below the Hermes Agent minimum of 64000. Set HERMES_LLAMA_CTX=65536."
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
$supportsSo8Triality = Test-HelpFlag $helpText '--so8-triality-k'
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
    32, 28, 24, 20, 16, 12, 8
) | Select-Object -Unique

function Build-ServerArgs {
    param(
        [int]$GpuLayers,
        [bool]$IncludeSpec,
        [bool]$IncludeCache,
        [bool]$IncludeSo8
    )
    $args = @()
    if ($ModelPath -and (Test-Path -LiteralPath $ModelPath)) {
        $args += @("-m", $ModelPath)
    } elseif ($supportsHfRepo) {
        $hfFlag = if ($supportsHfRepoLong) { "--hf-repo" } else { "-hf" }
        $args += @($hfFlag, $ModelRepo)
    } else {
        throw "Set HERMES_LLAMA_GGUF_PATH to a local .gguf or use a llama-server build with -hf/--hf-repo (model=$ModelRepo)"
    }
    $args += @(
        "--alias", $Alias,
        "--host", $HostName,
        "--port", [string]$Port,
        "--jinja",
        "-fa", "on",
        "-c", [string]$Ctx,
        "-ngl", [string]$GpuLayers,
        "-t", [string]$Threads,
        "-np", [string]$Parallel
    )
    if (Test-HelpFlag $helpText '--cont-batching') {
        $args += @("--cont-batching")
    }
    if (Test-HelpFlag $helpText '--batch-size') {
        $args += @("--batch-size", [string]$BatchSize)
    }
    if (Test-HelpFlag $helpText '--ubatch-size') {
        $args += @("--ubatch-size", [string]$UbatchSize)
    }
    if ($IncludeCache -and $supportsCacheK) {
        $args += @("--cache-type-k", $CacheK, "--cache-type-v", $CacheV)
        if ($IncludeSo8 -and $supportsSo8Triality) {
            $args += @("--so8-triality-k")
        }
    }
    if ($IncludeSpec -and $supportsSpecType -and $SpecType -and $SpecType -ne "none") {
        $args += @("--spec-type", $SpecType)
        if ($SpecType -eq "ngram-mod") {
            if (Test-HelpFlag $helpText '--spec-ngram-mod-n-match') {
                $args += @("--spec-ngram-mod-n-match", [string]$SpecNgramMatch)
            }
            if (Test-HelpFlag $helpText '--spec-ngram-mod-n-min') {
                $args += @("--spec-ngram-mod-n-min", [string]$SpecNgramMin)
            }
            if (Test-HelpFlag $helpText '--spec-ngram-mod-n-max') {
                $args += @("--spec-ngram-mod-n-max", [string]$SpecNgramMax)
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
        [bool]$IncludeCache,
        [bool]$IncludeSo8
    )
    $attemptArgs = Build-ServerArgs -GpuLayers $GpuLayers -IncludeSpec $IncludeSpec -IncludeCache $IncludeCache -IncludeSo8 $IncludeSo8
    $env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { $env:HF_HUB_CACHE }
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
    @{ IncludeSpec = $true; IncludeCache = $true; IncludeSo8 = $true },
    @{ IncludeSpec = $true; IncludeCache = $true; IncludeSo8 = $false },
    @{ IncludeSpec = $true; IncludeCache = $false; IncludeSo8 = $false },
    @{ IncludeSpec = $false; IncludeCache = $false; IncludeSo8 = $false }
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
        $attempt = Start-LlamaAttempt -GpuLayers $layers -IncludeSpec $plan.IncludeSpec -IncludeCache $plan.IncludeCache -IncludeSo8 $plan.IncludeSo8
        $deadline = (Get-Date).AddSeconds($WaitSeconds)
        while ((Get-Date) -lt $deadline) {
            if ($attempt.Process.HasExited) {
                if (Test-OomInStderr -Path $stderrPath) {
                    Write-Warning "CUDA OOM at ngl=$layers; retrying with fewer GPU layers."
                    break
                }
                $stderrTail = (Get-Content -LiteralPath $stderrPath -Tail 30 -ErrorAction SilentlyContinue) -join "`n"
                Write-Warning "llama-server exited with exit code $($attempt.Process.ExitCode) for plan (Spec:$($plan.IncludeSpec), Cache:$($plan.IncludeCache), SO8:$($plan.IncludeSo8), ngl:$layers). stderr:`n$stderrTail"
                break
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
