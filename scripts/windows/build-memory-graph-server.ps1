# Build memory-graph-server Go binary (Windows amd64).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-memory-graph-server.ps1

[CmdletBinding()]
param(
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$ModuleDir = Join-Path $RepoRoot "tools\memory-graph-server"
$BinDir = Join-Path $RepoRoot "bin"
$ExePath = Join-Path $BinDir "memory-graph-server.exe"

$go = Get-Command go -ErrorAction SilentlyContinue
if (-not $go) {
    throw "go not found on PATH — install Go 1.22+"
}

New-Item -ItemType Directory -Path $BinDir -Force | Out-Null

if ((Test-Path -LiteralPath $ExePath) -and -not $Force) {
    Write-Host "Already built: $ExePath (use -Force to rebuild)"
    exit 0
}

Push-Location $ModuleDir
try {
    Write-Host "Building memory-graph-server -> $ExePath"
    $env:GOMAXPROCS = "1"
    & go build -trimpath -ldflags "-s -w" -o $ExePath .
    if ($LASTEXITCODE -ne 0) {
        throw "go build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "OK: $ExePath"
