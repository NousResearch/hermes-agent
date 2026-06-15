param(
    [Parameter(Mandatory = $false)]
    [string]$LiveId = "",
    [string]$ApiKey = "",
    [string]$ApiKeyEnv = "AITUBER_ONAIR_YOUTUBE_API_KEY",
    [int]$PollSeconds = 2,
    [switch]$NoPlay,
    [switch]$SkipExisting,
    [switch]$Force,
    [switch]$Detach
)

$ErrorActionPreference = "Stop"

function Resolve-Default {
    param([string]$Name, [string]$Default)
    $fromEnv = [Environment]::GetEnvironmentVariable($Name)
    if (-not [string]::IsNullOrWhiteSpace($fromEnv)) {
        return $fromEnv
    }
    return $Default
}

if ([string]::IsNullOrWhiteSpace($LiveId)) {
    throw "YouTube live-id or live URL is required. Set it with -LiveId."
}

$candidate = Resolve-Default $ApiKeyEnv ""
if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
    $candidate = $ApiKey
}

if (-not [string]::IsNullOrWhiteSpace($candidate)) {
    [Environment]::SetEnvironmentVariable($ApiKeyEnv, $candidate)
    if (([Environment]::GetEnvironmentVariable($ApiKeyEnv) -ne $candidate)) {
        throw "Failed to set API key in this process environment."
    }
}

if ([string]::IsNullOrWhiteSpace($candidate)) {
    $secretPrompt = Read-Host "YouTube Data API key" -AsSecureString
    $secretBytes = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secretPrompt)
    try {
        $candidate = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($secretBytes)
    } finally {
        [void][System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($secretBytes)
    }
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        throw "API key was not provided."
    }
    [Environment]::SetEnvironmentVariable($ApiKeyEnv, $candidate)
}

$hermesRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $hermesRoot)) {
    throw "Could not resolve hermes-agent root from script path: $PSScriptRoot"
}

$args = @(
    "run", "hermes", "aituber-onair", "start-comments",
    "--live-id", $LiveId,
    "--api-key-env", $ApiKeyEnv,
    "--poll-seconds", $PollSeconds.ToString(),
    "--force"
)

if ($NoPlay) {
    $args += "--no-play"
}
if ($SkipExisting) {
    $args += "--skip-existing"
}

if ($Detach) {
    $p = Start-Process `
        -FilePath "uv" `
        -ArgumentList $args `
        -WorkingDirectory $hermesRoot `
        -WindowStyle Hidden `
        -PassThru
    Write-Output "YouTube comment monitor started in background."
    Write-Output ("PID: {0}" -f $p.Id)
} else {
    & "uv" @args
}
