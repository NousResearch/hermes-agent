$ErrorActionPreference = "Stop"

function Invoke-Tls12Download {
  param(
    [Parameter(Mandatory = $true)][string]$Uri,
    [Parameter(Mandatory = $true)][string]$OutFile
  )
  try {
    Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $OutFile
  } catch {
    throw "Failed to download $Uri with TLS 1.2 enabled: $($_.Exception.Message)"
  }
}

$version = if ($env:LEXMOUNT_BROWSER_CLI_VERSION) { $env:LEXMOUNT_BROWSER_CLI_VERSION } else { "1.1.15" }
$downloadBaseUrl = if ($env:LEXMOUNT_BROWSER_CLI_DOWNLOAD_BASE_URL) { $env:LEXMOUNT_BROWSER_CLI_DOWNLOAD_BASE_URL.TrimEnd('/') } else { "https://cli-bin-1377899528.cos.ap-nanjing.myqcloud.com/releases/browser-cli" }
$architecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($architecture -ne "AMD64") { throw "Only Windows x64 is supported" }
$asset = "browser-cli-v$version-x86_64-pc-windows-msvc.exe"
$repo = "$downloadBaseUrl/v$version"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
  $previousSecurityProtocol = [Net.ServicePointManager]::SecurityProtocol
  try {
    [Net.ServicePointManager]::SecurityProtocol = $previousSecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
  } catch {
    throw "Failed to enable TLS 1.2 for COS downloads: $($_.Exception.Message)"
  }
  try {
    Invoke-Tls12Download "$repo/$asset" (Join-Path $tmp $asset)
    Invoke-Tls12Download "$repo/SHA256SUMS" (Join-Path $tmp "SHA256SUMS")
  } finally {
    [Net.ServicePointManager]::SecurityProtocol = $previousSecurityProtocol
  }
  # GNU sha256sum prefixes binary filenames with `*`; shasum uses plain whitespace.
  $line = Get-Content (Join-Path $tmp "SHA256SUMS") | Where-Object { $_ -match "\s+\*?$([regex]::Escape($asset))$" } | Select-Object -First 1
  if (-not $line) { throw "No checksum published for $asset" }
  $expected = ($line -split "\s+")[0].ToLowerInvariant()
  $actual = (Get-FileHash (Join-Path $tmp $asset) -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($expected -ne $actual) { throw "SHA-256 mismatch for $asset" }
  $skillDir = Split-Path -Parent $PSScriptRoot
  $installDir = if ($env:LEXMOUNT_BROWSER_CLI_INSTALL_DIR) { $env:LEXMOUNT_BROWSER_CLI_INSTALL_DIR } else { Join-Path $skillDir "bin" }
  New-Item -ItemType Directory -Path $installDir -Force | Out-Null
  Copy-Item (Join-Path $tmp $asset) (Join-Path $installDir "browser-cli.exe") -Force
  & (Join-Path $installDir "browser-cli.exe") version
  Write-Output "Installed browser-cli to $installDir\browser-cli.exe"
} finally { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
