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
if (-not $env:MEMD_TAB_ID) {
  if ($env:WT_SESSION) {
    $env:MEMD_TAB_ID = "tab-{0}" -f $env:WT_SESSION.Substring(0, [Math]::Min(8, $env:WT_SESSION.Length))
  } elseif ($env:TERM_SESSION_ID) {
    $env:MEMD_TAB_ID = "tab-{0}" -f $env:TERM_SESSION_ID.Substring(0, [Math]::Min(8, $env:TERM_SESSION_ID.Length))
  } else {
    $env:MEMD_TAB_ID = "tab-{0}" -f $PID
  }
}
$env:MEMD_AGENT = "opencode"
$env:MEMD_WORKER_NAME = "Opencode"
try { memd wake --output $env:MEMD_BUNDLE_ROOT --route auto --intent current_task --write | Out-Null } catch { }
Start-Process -WindowStyle Hidden -FilePath memd -ArgumentList @('heartbeat','--output',$env:MEMD_BUNDLE_ROOT,'--watch','--interval-secs','30','--probe-base-url') -RedirectStandardOutput "$env:TEMP\memd-heartbeat.log" -RedirectStandardError "$env:TEMP\memd-heartbeat.err"
try { memd hive --output $env:MEMD_BUNDLE_ROOT --publish-heartbeat --summary | Out-Null } catch { }
memd wake --output $env:MEMD_BUNDLE_ROOT --route auto --intent current_task --write
