# Build hermes-watchdog.exe (Go + tsnet)
param(
    [switch]$SkipTest,
    [string]$OutputName = "hermes-watchdog.exe"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$GoDir = Join-Path $ScriptDir "watchdog-go"
$DistDir = Join-Path $GoDir "dist"
$OutPath = Join-Path $DistDir $OutputName

if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    throw "Go toolchain not found on PATH"
}

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
Push-Location -LiteralPath $GoDir
try {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Progress -Activity "Build Hermes Go Watchdog" -Status "go mod tidy" -PercentComplete 10
    Write-Host "[1/3] go mod tidy"
    go mod tidy
    if ($LASTEXITCODE -ne 0) { throw "go mod tidy failed" }

    if (-not $SkipTest) {
        Write-Progress -Activity "Build Hermes Go Watchdog" -Status "go test" -PercentComplete 45
        Write-Host "[2/3] go test ./..."
        go test ./... -count=1
        if ($LASTEXITCODE -ne 0) { throw "go test failed" }
    } else {
        Write-Host "[2/3] go test skipped"
    }

    Write-Progress -Activity "Build Hermes Go Watchdog" -Status "go build" -PercentComplete 80
    Write-Host "[3/3] go build"
    go build -buildvcs=false -trimpath -ldflags "-s -w" -o $OutPath .
    if ($LASTEXITCODE -ne 0) { throw "go build failed" }

    Write-Progress -Activity "Build Hermes Go Watchdog" -Completed -Status "done"
    $sw.Stop()
    Write-Host ("Built {0} in {1:n1}s" -f $OutPath, $sw.Elapsed.TotalSeconds)
    Get-Item -LiteralPath $OutPath | Format-List FullName, Length, LastWriteTime
}
finally {
    Pop-Location
}
