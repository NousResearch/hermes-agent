param(
  [string]$BaseUrl = $(if ($env:MEMD_BASE_URL) { $env:MEMD_BASE_URL } else { "http://100.104.154.24:8787" }),
  [string]$Project = $(if ($env:MEMD_PROJECT) { $env:MEMD_PROJECT } else { "" }),
  [string]$Namespace = $(if ($env:MEMD_NAMESPACE) { $env:MEMD_NAMESPACE } else { "" }),
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
  "hook",
  "capture",
  "--output", $bundleRoot,
  "--stdin",
  "--summary"
)
if ($Project) {
  $args += @("--project", $Project)
}
if ($Namespace) {
  $args += @("--namespace", $Namespace)
}
if ($Workspace) {
  $args += @("--workspace", $Workspace)
}
if ($Visibility) {
  $args += @("--visibility", $Visibility)
}

$stdin = [Console]::In.ReadToEnd()
if (-not [string]::IsNullOrWhiteSpace($stdin)) {
  $stdin | & memd @args
} else {
  & memd @args
}
