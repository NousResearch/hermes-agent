# Fallback llama.cpp launcher — Hermes-3 8B Q4_K_M on port 8081 by default.

param(
    [int]$WaitSeconds = 240
)

$ErrorActionPreference = "Stop"

function Resolve-Default {
    param([string]$Name, [string]$Default)
    $fromEnv = [Environment]::GetEnvironmentVariable($Name)
    if (-not [string]::IsNullOrWhiteSpace($fromEnv)) { return $fromEnv }
    return $Default
}

$ServerExe = Resolve-Default "HERMES_LLAMA_SERVER_EXE" (Join-Path $env:LOCALAPPDATA "Programs\llama-turboquant\bin\llama-server.exe")
$ModelRepo = Resolve-Default "HERMES_LLAMA_FALLBACK_MODEL" "NousResearch/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M"
$Alias = Resolve-Default "HERMES_LLAMA_FALLBACK_ALIAS" "hermes3-8b-fallback"
$HostName = Resolve-Default "HERMES_LLAMA_FALLBACK_HOST" "127.0.0.1"
$Port = [int](Resolve-Default "HERMES_LLAMA_FALLBACK_PORT" "8081")
$Ctx = [int](Resolve-Default "HERMES_LLAMA_FALLBACK_CTX" "65536")
$GpuLayers = [int](Resolve-Default "HERMES_LLAMA_FALLBACK_GPU_LAYERS" "99")

if ($Ctx -lt 64000) {
    throw "HERMES_LLAMA_FALLBACK_CTX=$Ctx is below minimum 64000."
}
if (-not (Test-Path -LiteralPath $ServerExe)) {
    throw "llama-server not found: $ServerExe"
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

$existing = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
    Where-Object { $_.State -eq "Listen" } |
    Select-Object -First 1
if ($existing) {
    Write-Output "llama.cpp fallback already listening on port $Port (pid=$($existing.OwningProcess))."
    exit 0
}

$logDir = Join-Path $env:USERPROFILE ".hermes\logs\llama-secretary-fallback"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $logDir "fallback-$stamp.out.log"
$stderrPath = Join-Path $logDir "fallback-$stamp.err.log"
$helpText = Get-LlamaHelpText -ServerExe $ServerExe
$supportsHfRepoLong = Test-HelpFlag $helpText '--hf-repo'
$supportsHfRepoShort = Test-HelpFlag $helpText '(^|[\s,])-hf([\s,]|$)'
$supportsHfRepo = $supportsHfRepoLong -or $supportsHfRepoShort
if (-not $supportsHfRepo) {
    throw "This llama-server build lacks -hf/--hf-repo; cannot load $ModelRepo"
}
$hfFlag = if ($supportsHfRepoLong) { "--hf-repo" } else { "-hf" }

$serverArgs = @(
    $hfFlag, $ModelRepo,
    "--alias", $Alias,
    "--host", $HostName,
    "--port", [string]$Port,
    "--jinja",
    "-fa", "on",
    "-c", [string]$Ctx,
    "-ngl", [string]$GpuLayers,
    "-np", "1"
)

$process = Start-Process `
    -FilePath $ServerExe `
    -ArgumentList $serverArgs `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

$modelsUrl = "http://${HostName}:${Port}/v1/models"
$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        $stderrTail = (Get-Content -LiteralPath $stderrPath -Tail 80 -ErrorAction SilentlyContinue) -join "`n"
        throw "fallback llama-server exited (exit=$($process.ExitCode)). stderr:`n$stderrTail"
    }
    try {
        $models = Invoke-RestMethod -Uri $modelsUrl -TimeoutSec 3
        Write-Output "llama.cpp fallback ready on $modelsUrl"
        Write-Output "pid=$($process.Id) model=$ModelRepo alias=$Alias ctx=$Ctx"
        $models | ConvertTo-Json -Depth 6
        exit 0
    } catch {
        Start-Sleep -Seconds 2
    }
}

throw "Fallback llama-server did not become ready within $WaitSeconds seconds. stderr=$stderrPath"
