# Start Irodori-TTS-Server as an OpenAI-compatible /v1/audio/speech endpoint.

param(
    [int]$StartupTimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"

function Resolve-Default {
    param([string]$Name, [string]$Default)
    $fromEnv = [Environment]::GetEnvironmentVariable($Name)
    if (-not [string]::IsNullOrWhiteSpace($fromEnv)) { return $fromEnv }
    return $Default
}

$RepoDir = Resolve-Default "IRODORI_TTS_DIR" ""
if ([string]::IsNullOrWhiteSpace($RepoDir)) {
    $hermesRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $RepoDir = Join-Path (Split-Path -Parent $hermesRoot) "irodori-tts-server"
}

$HostName = Resolve-Default "IRODORI_TTS_HOST" "127.0.0.1"
$Port = [int](Resolve-Default "IRODORI_TTS_PORT" "8088")
$Backend = Resolve-Default "IRODORI_TTS_BACKEND" "cuda"
$DefaultVoice = Resolve-Default "IRODORI_TTS_DEFAULT_VOICE" "none"
$OutputDir = Resolve-Default "IRODORI_TTS_OUTPUT_DIR" (Join-Path $env:USERPROFILE ".hermes\audio\irodori")

if (-not (Test-Path -LiteralPath $RepoDir)) {
    throw "Irodori-TTS-Server repo not found: $RepoDir (set IRODORI_TTS_DIR)"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$env:IRODORI_TTS_BACKEND = $Backend
$env:IRODORI_TTS_DEFAULT_VOICE = $DefaultVoice
$env:IRODORI_TTS_OUTPUT_DIR = $OutputDir

$healthUrl = "http://${HostName}:${Port}/health"
try {
    $existing = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3
    if ($existing.status -eq "ok") {
        Write-Output "Irodori-TTS already running at $healthUrl"
        exit 0
    }
} catch {
}

$legacyScript = Join-Path $PSScriptRoot "start-irodori-tts.ps1"
if (Test-Path -LiteralPath $legacyScript) {
    & $legacyScript -RepoDir $RepoDir -HostName $HostName -Port $Port -StartupTimeoutSeconds $StartupTimeoutSeconds
    exit $LASTEXITCODE
}

throw "start-irodori-tts.ps1 not found beside this script"
