# Save what tells a failed Windows App Installer update apart: App Installer's own
# version, the installed package, App Installer processes and the AppX event logs
# since the leg began. It only reads. The job log gets each AppX warning or error
# that names this package, a bundle, a descriptor or the loopback feed.
param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [Parameter(Mandatory = $true)][datetime]$Since,
    [Parameter(Mandatory = $true)][string]$Identity
)
$ErrorActionPreference = 'Continue'
$ours = [regex]::Escape($Identity) + '|msixbundle|appinstaller|127\.0\.0\.1'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$packages = @(foreach ($name in 'Microsoft.DesktopAppInstaller', $Identity) {
    Get-AppxPackage -Name $name | Select-Object Name, Version, Architecture, Status, PackageFullName, InstallLocation
})
ConvertTo-Json -InputObject $packages -Depth 3 | Set-Content -LiteralPath (Join-Path $OutDir 'packages.json') -Encoding UTF8
$installers = @(Get-Process -Name AppInstaller -ErrorAction SilentlyContinue | Select-Object Id, StartTime, Path)
ConvertTo-Json -InputObject $installers | Set-Content -LiteralPath (Join-Path $OutDir 'appinstaller-processes.json') -Encoding UTF8
foreach ($package in $packages) { Write-Host "  [evidence] $($package.Name) $($package.Version) $($package.Architecture) $($package.Status)" }
$logs = @(
    'Microsoft-Windows-AppxPackaging/Operational',
    'Microsoft-Windows-AppXDeploymentServer/Operational',
    'Microsoft-Windows-AppXDeployment/Operational',
    'Microsoft-Windows-AppModel-Runtime/Admin'
)
foreach ($log in $logs) {
    $events = @(Get-WinEvent -FilterHashtable @{ LogName = $log; StartTime = $Since } -ErrorAction SilentlyContinue | Sort-Object TimeCreated)
    $lines = @($events | ForEach-Object {
        '{0:o} {1} {2} {3}' -f $_.TimeCreated.ToUniversalTime(), $_.Id, $_.LevelDisplayName, ($_.Message -replace '\s+', ' ')
    })
    Set-Content -LiteralPath (Join-Path $OutDir (($log -replace '[/\\]', '_') + '.txt')) -Value $lines -Encoding UTF8
    Write-Host "  [evidence] $($log): $($events.Count) events"
    foreach ($record in @($events | Where-Object { $_.Level -in 1, 2, 3 -and $_.Message -match $ours })) {
        $text = ($record.Message -replace '\s+', ' ')
        Write-Host "    $($record.TimeCreated.ToUniversalTime().ToString('o')) $($record.Id) $($record.LevelDisplayName): $($text.Substring(0, [Math]::Min(300, $text.Length)))"
    }
}
Write-Host "  [evidence] saved under $OutDir"
