# Leave Windows PowerShell's module analysis cache warm for the test processes.
#
# The first cmdlet a fresh powershell.exe autoloads makes it analyse every module on
# PSModulePath (Az, AWS, Graph on the hosted images: ~22s on the arm64 runners).
# PowerShell writes that analysis to the cache from a timer ~10s later, so a process
# that exits right after its first cmdlet, which is every PowerShell child a test
# spawns, never saves it and the next one pays the 22s again. When the analysis ran
# here, stay up until the cache file is rewritten. A warm cache returns at once.
Get-ChildItem . | Out-Null
$process = Get-Process -Id $PID
$analysis = ((Get-Date) - $process.StartTime).TotalSeconds
if ($analysis -gt 5) {
    $cache = $env:PSModuleAnalysisCachePath
    if (-not $cache) { $cache = Join-Path $env:LOCALAPPDATA 'Microsoft\Windows\PowerShell\ModuleAnalysisCache' }
    for ($i = 0; $i -lt 40; $i++) {
        if ((Test-Path $cache) -and (Get-Item $cache).LastWriteTime -gt $process.StartTime) { break }
        Start-Sleep -Milliseconds 500
    }
}
Write-Host ("module analysis cache ready; this launch analysed for {0:N1}s" -f $analysis)
