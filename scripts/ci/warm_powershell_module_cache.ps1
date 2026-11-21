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
$cache = $env:PSModuleAnalysisCachePath
if (-not $cache) { $cache = Join-Path $env:LOCALAPPDATA 'Microsoft\Windows\PowerShell\ModuleAnalysisCache' }
$saved = $analysis -le 5  # already warm: nothing new to save
for ($i = 0; -not $saved -and $i -lt 40; $i++) {
    $saved = (Test-Path $cache) -and (Get-Item $cache).LastWriteTime -gt $process.StartTime
    if (-not $saved) { Start-Sleep -Milliseconds 500 }
}
if ($saved) {
    Write-Host ("module analysis cache ready; this launch analysed for {0:N1}s" -f $analysis)
} else {
    # Not fatal: the tests still run, each PowerShell child paying the cold start.
    Write-Host ("::warning::PowerShell did not save its module analysis cache to {0} within 20s; Windows tests will start PowerShell cold" -f $cache)
}
