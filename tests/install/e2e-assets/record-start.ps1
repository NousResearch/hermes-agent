# Start a continuous ffmpeg screen recording in the background (windows).
#
# Usage: powershell -File record-start.ps1 -OutFile recording.mkv
#
# The graceful stop is the character 'q' on ffmpeg's LIVE stdin, which only
# System.Diagnostics.Process exposes (Start-Process -RedirectStandardInput
# hands ffmpeg a file handle already at EOF). This script therefore spawns a
# detached HOLDER powershell that owns the ffmpeg process and its stdin pipe,
# and stops it when a STOP marker file appears; record-stop.ps1 writes the
# marker. mkv on purpose: it stays playable even unfinalized. A missing
# ffmpeg is a HARD error - a graceful skip makes the missing tool invisible
# and the artifact silently loses its recording.

#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)][string]$OutFile
)
$ErrorActionPreference = "Stop"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "record-start: ffmpeg not on PATH (the workflow must install it)"
    exit 1
}

$outDir = Split-Path -Parent $OutFile
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}
$stopMarker = "$OutFile.stop"
$stateFile = "$OutFile.state"
Remove-Item -LiteralPath $stopMarker, $stateFile -Force -ErrorAction SilentlyContinue

$holder = "$OutFile.holder.ps1"
@'
param([string]$OutFile, [string]$StopMarker)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "ffmpeg"
$psi.Arguments = "-y -f gdigrab -framerate 15 -i desktop " +
    "-hide_banner -loglevel error " +
    "-c:v libx264 -preset ultrafast -pix_fmt yuv420p `"$OutFile`""
$psi.RedirectStandardInput = $true
$psi.UseShellExecute = $false
$proc = [System.Diagnostics.Process]::Start($psi)
Set-Content -LiteralPath "$OutFile.ffpid" -Value $proc.Id
while (-not $proc.HasExited) {
    if (Test-Path -LiteralPath $StopMarker) {
        try {
            $proc.StandardInput.Write("q")
            $proc.StandardInput.Close()
        } catch {}
        if (-not $proc.WaitForExit(15000)) { try { $proc.Kill() } catch {} }
        break
    }
    Start-Sleep -Milliseconds 500
}
'@ | Set-Content -LiteralPath $holder -Encoding UTF8

$holderProc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $holder, "-OutFile", $OutFile, "-StopMarker", $stopMarker `
    -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $stateFile -Value "$($holderProc.Id) $stopMarker"
Write-Host "record-start: recording to $OutFile (holder pid $($holderProc.Id))"
