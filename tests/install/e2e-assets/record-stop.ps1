# Stop a recording started by record-start.ps1 and verify the file is real.
#
# Usage: powershell -File record-stop.ps1 -OutFile recording.mkv
#
# Drops the STOP marker the holder watches for (it writes 'q' to ffmpeg's
# live stdin), waits for the holder to exit, then asserts the output exists
# and has a decodable duration - a zero-frame recording is the classic
# silent failure.

#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)][string]$OutFile
)
$ErrorActionPreference = "Stop"

$stateFile = "$OutFile.state"
if (-not (Test-Path -LiteralPath $stateFile)) {
    Write-Host "record-stop: no state at $stateFile (was record-start run?)"
    exit 1
}
$state = (Get-Content -LiteralPath $stateFile -Raw).Trim() -split " ", 2
$holderPid = [int]$state[0]
$stopMarker = $state[1]

Set-Content -LiteralPath $stopMarker -Value "stop"
try {
    $holder = Get-Process -Id $holderPid -ErrorAction SilentlyContinue
    if ($holder) { $holder.WaitForExit(20000) | Out-Null }
} catch {}
# Belt and braces: if ffmpeg outlived the holder, kill it directly.
$ffpidFile = "$OutFile.ffpid"
if (Test-Path -LiteralPath $ffpidFile) {
    $ffpid = [int](Get-Content -LiteralPath $ffpidFile -Raw).Trim()
    try { Stop-Process -Id $ffpid -Force -ErrorAction SilentlyContinue } catch {}
}
Remove-Item -LiteralPath $stateFile, $stopMarker, $ffpidFile, "$OutFile.holder.ps1" -Force -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $OutFile) -or (Get-Item -LiteralPath $OutFile).Length -eq 0) {
    Write-Host "record-stop: $OutFile missing or empty"
    exit 1
}
if (Get-Command ffprobe -ErrorAction SilentlyContinue) {
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $dur = (& ffprobe -v error -show_entries format=duration -of csv=p=0 $OutFile 2>&1 | Out-String).Trim()
    $ErrorActionPreference = $prevEap
    if (-not $dur -or $dur -eq "0" -or $dur -like "0.0*") {
        Write-Host "record-stop: $OutFile has no duration (zero-frame recording)"
        exit 1
    }
    Write-Host "record-stop: $OutFile finalized (${dur}s)"
} else {
    Write-Host "record-stop: $OutFile finalized (ffprobe absent; size $((Get-Item -LiteralPath $OutFile).Length) bytes)"
}
