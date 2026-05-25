param(
  [string]$BaseUrl = $(if ($env:MEMD_BASE_URL) { $env:MEMD_BASE_URL } else { "http://100.104.154.24:8787" }),
  [Parameter(Mandatory = $true)][string]$Project = $(if ($env:MEMD_PROJECT) { $env:MEMD_PROJECT } else { throw "MEMD_PROJECT is required" }),
  [string]$Namespace = $(if ($env:MEMD_NAMESPACE) { $env:MEMD_NAMESPACE } else { "" }),
  [Parameter(Mandatory = $true)][string]$Agent = $(if ($env:MEMD_AGENT) { $env:MEMD_AGENT } else { throw "MEMD_AGENT is required" }),
  [string]$Route = $(if ($env:MEMD_ROUTE) { $env:MEMD_ROUTE } else { "auto" }),
  [string]$Intent = $(if ($env:MEMD_INTENT) { $env:MEMD_INTENT } else { "current_task" }),
  [int]$Limit = $(if ($env:MEMD_LIMIT) { [int]$env:MEMD_LIMIT } else { 8 }),
  [int]$RehydrationLimit = $(if ($env:MEMD_REHYDRATION_LIMIT) { [int]$env:MEMD_REHYDRATION_LIMIT } else { 4 }),
  [string]$Workspace = $(if ($env:MEMD_WORKSPACE) { $env:MEMD_WORKSPACE } else { "" }),
  [string]$Visibility = $(if ($env:MEMD_VISIBILITY) { $env:MEMD_VISIBILITY } else { "" })
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

$args = @(
  "--base-url", $BaseUrl,
  "wake",
  "--output", $bundleRoot,
  "--project", $Project,
  "--agent", $Agent,
  "--route", $Route,
  "--intent", $Intent,
  "--limit", $Limit,
  "--rehydration-limit", $RehydrationLimit,
  "--write"
)
if ($Namespace) {
  $args += @("--namespace", $Namespace)
}
if ($Workspace) {
  $args += @("--workspace", $Workspace)
}
if ($Visibility) {
  $args += @("--visibility", $Visibility)
}
& memd @args
