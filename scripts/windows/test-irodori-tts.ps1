# Quick Irodori-TTS health + speech generation smoke test.

param(
    [string]$BaseUrl = "http://127.0.0.1:8088",
    [string]$OutputPath = "",
    [string]$StartScriptPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $outDir = Join-Path $env:USERPROFILE ".hermes\audio\irodori"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $OutputPath = Join-Path $outDir ("smoke-{0}.wav" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$scriptPath = Join-Path $repoRoot "skills\audio\irodori-tts\scripts\irodori_tts.py"
$inputPath = [System.IO.Path]::GetTempFileName() + ".txt"
Set-Content -LiteralPath $inputPath -Value "Local secretary Irodori smoke test." -Encoding UTF8

try {
    $healthUrl = "$BaseUrl/health"
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3
        if ($health.status -ne "ok") { throw "unexpected health status" }
    } catch {
        if ([string]::IsNullOrWhiteSpace($StartScriptPath)) {
            $StartScriptPath = Join-Path $PSScriptRoot "start-irodori-tts-server.ps1"
        }
        & $StartScriptPath | Out-Null
    }

    $args = @(
        $scriptPath,
        "--text-file", $inputPath,
        "--output", $OutputPath,
        "--base-url", $BaseUrl,
        "--voice", "none",
        "--response-format", "wav",
        "--speed", "1.0",
        "--dry-run"
    )
    $json = py -3 @args
    Write-Output $json
    $parsed = $json | ConvertFrom-Json
    if (-not $parsed.success) { throw "irodori_tts dry-run failed" }
    if (-not (Test-Path -LiteralPath $OutputPath)) {
        throw "expected output file missing: $OutputPath"
    }
    Write-Output "Irodori-TTS smoke test ok: $OutputPath"
} finally {
    Remove-Item -LiteralPath $inputPath -Force -ErrorAction SilentlyContinue
}
