# Wait outside the package until the original app exits and Windows publishes its update.
param(
  [Parameter(Mandatory = $true)][int]$ProcessId,
  [Parameter(Mandatory = $true)][long]$ProcessStartTimeMs,
  [Parameter(Mandatory = $true)][string]$IdentityName,
  [Parameter(Mandatory = $true)][string]$ReadyFile,
  [int]$TimeoutSeconds = 900,
  [int]$PollMillis = 500
)

# The parent runs this detached with no stdio, so this file is the only record of why it gave up.
# It is shared across attempts: lines carry the stage dir name so the parent can pick out its own,
# and it restarts when it passes 64KB so it cannot grow without bound.
$LogFile = Join-Path $env:TEMP 'hermes-relaunch-waiter.log'
$Attempt = Split-Path -Leaf (Split-Path -Parent $ReadyFile)
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
try { if ((Test-Path -LiteralPath $LogFile) -and (Get-Item -LiteralPath $LogFile).Length -gt 65536) { Remove-Item -LiteralPath $LogFile -Force } } catch {}
function Write-WaiterLog([string]$Message) {
  # Tag every physical line: a multi-line error (message, location, details) must survive the parent's attempt filter.
  try {
    $stamp = Get-Date -Format o
    $lines = @($Message -split '\r?\n' | Where-Object { $_.Trim() })
    if (-not $lines) { return }
    $text = ($lines | ForEach-Object { '{0} [{1}] pid={2} {3}' -f $stamp, $Attempt, $PID, $_ }) -join "`n"
    [IO.File]::AppendAllText($LogFile, $text + "`n", $Utf8NoBom)
  } catch {}
}

try {
  $ErrorActionPreference = 'Stop'
  Write-WaiterLog "start: parent=$ProcessId identity=$IdentityName timeout=${TimeoutSeconds}s"
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  $package = Get-AppxPackage -Name $IdentityName | Select-Object -First 1
  if (-not $package) { Write-WaiterLog "no package named $IdentityName"; exit 2 }
  $appId = (Get-AppxPackageManifest $package).Package.Applications.Application.Id | Select-Object -First 1
  if (-not $appId) { Write-WaiterLog "package $($package.PackageFullName) has no application id"; exit 2 }
  $fromVersion = $package.Version
  $familyName = $package.PackageFamilyName
  Set-Content -LiteralPath $ReadyFile -Value "$fromVersion|$familyName|$appId"
  Write-WaiterLog "ready: from=$fromVersion family=$familyName app=$appId"

  while ($true) {
    $parent = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $parent) { break }
    # Compare absolute birth times. Process age changes on every poll.
    $birthMs = ([DateTimeOffset]$parent.StartTime).ToUnixTimeMilliseconds()
    if ([Math]::Abs($birthMs - $ProcessStartTimeMs) -gt 3000) { break }
    if ((Get-Date) -gt $deadline) { Write-WaiterLog 'timed out waiting for the parent to exit'; exit 3 }
    Start-Sleep -Milliseconds $PollMillis
  }

  while ((Get-Date) -le $deadline) {
    $current = Get-AppxPackage -Name $IdentityName | Select-Object -First 1
    if ($current -and $current.Version -ne $fromVersion -and $current.PackageFamilyName -eq $familyName) {
      Write-WaiterLog "package is now $($current.Version); launching"
      Start-Process "shell:AppsFolder\$familyName!$appId"
      Write-WaiterLog 'launched'
      exit 0
    }
    Start-Sleep -Milliseconds $PollMillis
  }
  Write-WaiterLog "timed out waiting for the package version to change from $fromVersion"
  exit 4
} catch {
  Write-WaiterLog ('failed: ' + ($_ | Out-String).Trim())
  exit 1
} finally {
  Set-Location $env:TEMP
  Remove-Item -LiteralPath $ReadyFile -Force -ErrorAction SilentlyContinue
  if ([IO.Path]::GetFileName($PSScriptRoot).StartsWith('hermes-relaunch-')) {
    Remove-Item -LiteralPath $PSScriptRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}
