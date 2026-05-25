param(
  [string]$BaseUrl = $(if ($env:MEMD_BASE_URL) { $env:MEMD_BASE_URL } else { "http://100.104.154.24:8787" }),
  [switch]$Apply,
  [switch]$SpillTransient
)

$bundleRoot = if ($env:MEMD_BUNDLE_ROOT) { $env:MEMD_BUNDLE_ROOT } else { ".memd" }
$backendEnvPath = Join-Path $bundleRoot "backend.env.ps1"
$envPath = Join-Path $bundleRoot "env.ps1"
if (Test-Path $backendEnvPath) {
  . $backendEnvPath
}
if (Test-Path $envPath) {
  . $envPath
}

$args = @("--base-url", $BaseUrl, "hook", "spill")
if ($Apply) { $args += "--apply" }
if ($SpillTransient) { $args += "--spill-transient" }
memd @args
