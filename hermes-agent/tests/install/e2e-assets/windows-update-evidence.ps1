# Save what tells a failed Windows App Installer update apart: App Installer's own
# version, the installed package, App Installer processes, its diagnostic logs and the
# AppX event logs since the leg began. It only reads. The job log gets each AppX warning or error
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
# App Installer's own log names the HRESULT behind its dialogs. A bundle it cannot parse
# leaves that only there, not in the AppX logs. Keep the files it wrote since the leg began.
$diag = Join-Path $env:LOCALAPPDATA 'Packages\Microsoft.DesktopAppInstaller_8wekyb3d8bbwe\LocalState\DiagOutputDir'
if (Test-Path -LiteralPath $diag) {
    $diagFiles = @(Get-ChildItem -LiteralPath $diag -File | Where-Object { $_.LastWriteTime -ge $Since })
    $diagOut = New-Item -ItemType Directory -Force -Path (Join-Path $OutDir 'appinstaller-diagnostics')
    $diagFiles | Copy-Item -Destination $diagOut.FullName
    Write-Host "  [evidence] App Installer diagnostics: $($diagFiles.Count) files since the leg began"
    $errors = @($diagFiles | Where-Object Name -like 'AppInstaller-*' | Get-Content | Where-Object { $_ -match '\] ERROR: ' })
    foreach ($line in @($errors | Select-Object -First 20)) { Write-Host "    $line" }
} else {
    Write-Host "  [evidence] App Installer diagnostics: $diag does not exist"
}
$logs = @(
    'Microsoft-Windows-AppxPackaging/Operational',
    'Microsoft-Windows-AppXDeploymentServer/Operational',
    'Microsoft-Windows-AppXDeployment/Operational',
    'Microsoft-Windows-AppModel-Runtime/Admin'
)
foreach ($log in $logs) {
    # Get-WinEvent reports an empty window as an error too. Any other error (a missing channel
    # throws) is saved, so an unread log never looks empty and the remaining logs are still read.
    $events = @()
    $unread = $null
    try {
        $events = @(Get-WinEvent -FilterHashtable @{ LogName = $log; StartTime = $Since } -ErrorAction Stop | Sort-Object TimeCreated)
    } catch {
        if ($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound,*') { $unread = "could not read: $($_.Exception.Message)" }
    }
    $lines = @($events | ForEach-Object {
        '{0:o} {1} {2} {3}' -f $_.TimeCreated.ToUniversalTime(), $_.Id, $_.LevelDisplayName, ($_.Message -replace '\s+', ' ')
    })
    if ($unread) { $lines = @($unread) }
    Set-Content -LiteralPath (Join-Path $OutDir (($log -replace '[/\\]', '_') + '.txt')) -Value $lines -Encoding UTF8
    if ($unread) { Write-Host "  [evidence] $($log): $unread" } else { Write-Host "  [evidence] $($log): $($events.Count) events" }
    foreach ($record in @($events | Where-Object { $_.Level -in 1, 2, 3 -and $_.Message -match $ours })) {
        $text = ($record.Message -replace '\s+', ' ')
        Write-Host "    $($record.TimeCreated.ToUniversalTime().ToString('o')) $($record.Id) $($record.LevelDisplayName): $($text.Substring(0, [Math]::Min(300, $text.Length)))"
    }
}
Write-Host "  [evidence] saved under $OutDir"
