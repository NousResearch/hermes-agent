param(
    [string]$RepoDir = "",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8088,
    [int]$StartupTimeoutSeconds = 90,
    [string]$HfCacheRoot = "",
    [string]$ModelDevice = "auto",
    [string]$CodecDevice = "auto",
    [string]$BackendExtra = "cu128"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoDir)) {
    $hermesRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $RepoDir = Join-Path (Split-Path -Parent $hermesRoot) "irodori-tts-server"
}

if (-not (Test-Path -LiteralPath $RepoDir)) {
    throw "Irodori-TTS-Server repo was not found: $RepoDir"
}

$logDir = Join-Path $env:LOCALAPPDATA "hermes\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if ([string]::IsNullOrWhiteSpace($HfCacheRoot)) {
    if (Test-Path -LiteralPath "D:\") {
        $HfCacheRoot = "D:\llama-cpp-cache\huggingface"
    } else {
        $HfCacheRoot = Join-Path $env:LOCALAPPDATA "hermes\huggingface"
    }
}
$hfHubCache = Join-Path $HfCacheRoot "hub"
New-Item -ItemType Directory -Force -Path $hfHubCache | Out-Null
$torchCacheRoot = Join-Path $env:LOCALAPPDATA "hermes\torch-cache"
$torchInductorCache = Join-Path $torchCacheRoot "inductor"
New-Item -ItemType Directory -Force -Path $torchInductorCache | Out-Null
if ([string]::IsNullOrWhiteSpace($env:USER) -and -not [string]::IsNullOrWhiteSpace($env:USERNAME)) {
    $env:USER = $env:USERNAME
}
$env:HF_HOME = $HfCacheRoot
$env:HF_HUB_CACHE = $hfHubCache
$env:HUGGINGFACE_HUB_CACHE = $hfHubCache
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TORCH_HOME = $torchCacheRoot
$env:TORCHINDUCTOR_CACHE_DIR = $torchInductorCache
if (-not [string]::IsNullOrWhiteSpace($ModelDevice)) {
    $env:IRODORI_MODEL_DEVICE = $ModelDevice
}
if (-not [string]::IsNullOrWhiteSpace($CodecDevice)) {
    $env:IRODORI_CODEC_DEVICE = $CodecDevice
}

$healthUrl = "http://${HostName}:${Port}/health"
try {
    $existing = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3
    if ($existing.status -eq "ok") {
        Write-Output "Irodori-TTS server already running at $healthUrl"
        exit 0
    }
} catch {
}

$stdout = Join-Path $logDir "irodori-tts-stdout.log"
$stderr = Join-Path $logDir "irodori-tts-stderr.log"
$arguments = @("run")
if (-not [string]::IsNullOrWhiteSpace($BackendExtra)) {
    $arguments += @("--extra", $BackendExtra)
}
$arguments += @(
    "python",
    "-m",
    "irodori_openai_tts",
    "--host",
    $HostName,
    "--port",
    [string]$Port
)

$process = Start-Process `
    -FilePath "uv" `
    -ArgumentList $arguments `
    -WorkingDirectory $RepoDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
do {
    Start-Sleep -Seconds 1
    if ($process.HasExited) {
        $err = if (Test-Path -LiteralPath $stderr) {
            Get-Content -LiteralPath $stderr -Tail 50 -ErrorAction SilentlyContinue | Out-String
        } else {
            ""
        }
        throw "Irodori-TTS server exited during startup. $err"
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3
        if ($health.status -eq "ok") {
            Write-Output "Irodori-TTS server started at $healthUrl (PID $($process.Id))"
            exit 0
        }
    } catch {
    }
} while ((Get-Date) -lt $deadline)

throw "Irodori-TTS server did not answer $healthUrl within $StartupTimeoutSeconds seconds. Logs: $stdout ; $stderr"
