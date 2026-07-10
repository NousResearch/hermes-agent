<#
.SYNOPSIS
  Register the Hermes for Excel add-in as a developer sideload for Excel desktop.

.DESCRIPTION
  requires elevation? NO. Writes only HKCU and per-user files.

  Excel desktop on Windows sideloads add-ins from a "shared folder catalog"
  declared under:
      HKCU\Software\Microsoft\Office\16.0\WEF\Developer
  Each value in that key is a folder path Excel scans for manifest .xml files.
  We point it at a per-user catalog folder and drop the manifest there.

  Catalog folder: %LOCALAPPDATA%\hermes\excel-addin\OfficeAddinManifests
  Manifest file : hermes-excel-addin.xml inside that folder.

  PORT NOTE: manifest.xml hardcodes http://localhost:8787 in every URL. If a
  box runs the bridge on a non-default port (collision boxes), we string-replace
  8787 -> the chosen port when copying the manifest into the catalog, so Excel
  loads a manifest whose URLs actually match the running bridge.

  MANUAL FALLBACK (if the registry catalog is not picked up, e.g. policy-locked
  or Excel build differs): in Excel use
      Home -> Add-ins -> More Add-ins -> My Add-ins ->
      (Manage My Add-ins) -> Upload My Add-in -> browse to the catalog manifest.
  The Upload-My-Add-in path always works even when the Developer catalog does not.

  Idempotent: re-running overwrites the catalog manifest and re-sets the value.

.PARAMETER InstallDir
  Absolute install root that contains manifest.xml. Default:
  %LOCALAPPDATA%\hermes\excel-addin

.PARAMETER Port
  Bridge port. If not 8787, the catalog manifest is port-substituted.

.PARAMETER OfficeVersion
  Office registry version segment. Default 16.0 (Office 2016/2019/2021/365).
  DEPLOYER: verify this matches the installed Excel build's registry hive.

.PARAMETER Unregister
  Remove the WEF Developer value and the catalog manifest.
#>
[CmdletBinding()]
param(
  [string] $InstallDir    = (Join-Path $env:LOCALAPPDATA 'hermes\excel-addin'),
  [int]    $Port          = 8787,
  [string] $OfficeVersion = '16.0',
  [switch] $Unregister
)

$ErrorActionPreference = 'Stop'

# ---- Config block (retarget here) ------------------------------------------
$CatalogDir   = Join-Path $InstallDir 'OfficeAddinManifests'
$CatalogName  = 'hermes-excel-addin.xml'
$CatalogPath  = Join-Path $CatalogDir $CatalogName
$SourceMani   = Join-Path $InstallDir 'manifest.xml'
$WefDevKey    = "HKCU:\Software\Microsoft\Office\$OfficeVersion\WEF\Developer"
$WefValueName = 'HermesExcelAddinCatalog'
$DefaultPort  = 8787
# ----------------------------------------------------------------------------

function Remove-Sideload {
  if (Test-Path -LiteralPath $WefDevKey) {
    $existing = (Get-ItemProperty -LiteralPath $WefDevKey -ErrorAction SilentlyContinue)
    if ($existing -and ($existing.PSObject.Properties.Name -contains $WefValueName)) {
      Remove-ItemProperty -LiteralPath $WefDevKey -Name $WefValueName -Force -ErrorAction SilentlyContinue
      Write-Host "[register-sideload] Removed WEF Developer value '$WefValueName'."
    } else {
      Write-Host "[register-sideload] WEF Developer value '$WefValueName' not present."
    }
  } else {
    Write-Host "[register-sideload] WEF Developer key not present; nothing to remove."
  }
  if (Test-Path -LiteralPath $CatalogPath) {
    Remove-Item -LiteralPath $CatalogPath -Force -ErrorAction SilentlyContinue
    Write-Host "[register-sideload] Removed catalog manifest '$CatalogPath'."
  }
}

if ($Unregister) {
  Remove-Sideload
  return
}

try {
  if (-not (Test-Path -LiteralPath $SourceMani)) {
    throw "Source manifest not found at '$SourceMani'. Run addin-install.ps1 first."
  }

  # Ensure catalog folder exists.
  if (-not (Test-Path -LiteralPath $CatalogDir)) {
    New-Item -ItemType Directory -Path $CatalogDir -Force | Out-Null
  }

  # Copy manifest into the catalog, port-substituting if needed.
  if ($Port -ne $DefaultPort) {
    Write-Host "[register-sideload] Port $Port != $DefaultPort; producing port-substituted manifest."
    $xml = Get-Content -LiteralPath $SourceMani -Raw
    # manifest only ever references localhost:8787; rewrite the port everywhere.
    $xml = $xml -replace 'localhost:8787', "localhost:$Port"
    Set-Content -LiteralPath $CatalogPath -Value $xml -Encoding UTF8 -Force
  } else {
    Copy-Item -LiteralPath $SourceMani -Destination $CatalogPath -Force
  }
  Write-Host "[register-sideload] Wrote catalog manifest '$CatalogPath'."

  # Ensure the WEF\Developer key exists, then point a value at the catalog FOLDER.
  if (-not (Test-Path -LiteralPath $WefDevKey)) {
    New-Item -Path $WefDevKey -Force | Out-Null
  }
  # The Developer catalog value's data is the FOLDER (Excel scans it for xml).
  New-ItemProperty -LiteralPath $WefDevKey -Name $WefValueName -Value $CatalogDir `
    -PropertyType String -Force | Out-Null

  Write-Host "[register-sideload] Set WEF Developer catalog -> '$CatalogDir' (key: $WefDevKey)."
  Write-Host "[register-sideload] Restart Excel, then open it from the Hermes ribbon group."
  Write-Host "[register-sideload] Manual fallback: Home -> Add-ins -> Upload My Add-in -> '$CatalogPath'."
}
catch {
  Write-Error "[register-sideload] FAILED: $($_.Exception.Message)"
  exit 1
}
