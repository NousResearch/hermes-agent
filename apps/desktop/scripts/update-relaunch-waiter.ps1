# Wait outside the package until the original app exits and Windows publishes its update.
param(
  [Parameter(Mandatory = $true)][int]$ProcessId,
  [Parameter(Mandatory = $true)][long]$ProcessStartTimeMs,
  [Parameter(Mandatory = $true)][string]$IdentityName,
  [Parameter(Mandatory = $true)][string]$ReadyFile,
  [int]$TimeoutSeconds = 900,
  [int]$PollMillis = 500
)

try {
  $ErrorActionPreference = 'Stop'
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  $package = Get-AppxPackage -Name $IdentityName | Select-Object -First 1
  if (-not $package) { exit 2 }
  $appId = (Get-AppxPackageManifest $package).Package.Applications.Application.Id | Select-Object -First 1
  if (-not $appId) { exit 2 }
  $fromVersion = $package.Version
  $familyName = $package.PackageFamilyName
  Set-Content -LiteralPath $ReadyFile -Value "$fromVersion|$familyName|$appId"

  while ($true) {
    $parent = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $parent) { break }
    # Compare absolute birth times. Process age changes on every poll.
    $birthMs = ([DateTimeOffset]$parent.StartTime).ToUnixTimeMilliseconds()
    if ([Math]::Abs($birthMs - $ProcessStartTimeMs) -gt 3000) { break }
    if ((Get-Date) -gt $deadline) { exit 3 }
    Start-Sleep -Milliseconds $PollMillis
  }

  while ((Get-Date) -le $deadline) {
    $current = Get-AppxPackage -Name $IdentityName | Select-Object -First 1
    if ($current -and $current.Version -ne $fromVersion -and $current.PackageFamilyName -eq $familyName) {
      Start-Process "shell:AppsFolder\$familyName!$appId"
      exit 0
    }
    Start-Sleep -Milliseconds $PollMillis
  }
  exit 4
} finally {
  Set-Location $env:TEMP
  Remove-Item -LiteralPath $ReadyFile -Force -ErrorAction SilentlyContinue
  if ([IO.Path]::GetFileName($PSScriptRoot).StartsWith('hermes-relaunch-')) {
    Remove-Item -LiteralPath $PSScriptRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}
