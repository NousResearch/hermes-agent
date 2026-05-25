$_memdDefaultRoot = "/home/aparcedodev/.hermes/hermes-agent/.memd"
function _Memd-ResolveRoot {
if ($env:MEMD_BUNDLE_ROOT -and (Test-Path $env:MEMD_BUNDLE_ROOT)) { return $env:MEMD_BUNDLE_ROOT }
$d = (Get-Location).Path
while ($d) {
$candidate = Join-Path $d '.memd'
if (Test-Path $candidate) { return $candidate }
$parent = Split-Path -Parent $d
if (-not $parent -or $parent -eq $d) { break }
$d = $parent
}
try {
$gcd = (git rev-parse --git-common-dir 2>$null) | Out-String
$gcd = $gcd.Trim()
if ($gcd) {
if (-not [System.IO.Path]::IsPathRooted($gcd)) { $gcd = Join-Path (Get-Location).Path $gcd }
$mainRoot = Split-Path -Parent $gcd
$candidate = Join-Path $mainRoot '.memd'
if (Test-Path $candidate) { return $candidate }
}
} catch { }
return $_memdDefaultRoot
}
$env:MEMD_BUNDLE_ROOT = (_Memd-ResolveRoot)
$bundleBackendEnv = Join-Path $env:MEMD_BUNDLE_ROOT "backend.env.ps1"
if (Test-Path $bundleBackendEnv) { . $bundleBackendEnv }
. (Join-Path $env:MEMD_BUNDLE_ROOT "env.ps1")
$args = @("rag", "sync")
if ($env:MEMD_PROJECT) { $args += @("--project", $env:MEMD_PROJECT) }
if ($env:MEMD_NAMESPACE) { $args += @("--namespace", $env:MEMD_NAMESPACE) }
memd @args @Args
