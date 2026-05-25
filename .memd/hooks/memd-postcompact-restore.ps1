param(
  [string]$StateDir = $(if ($env:MEMD_HOOK_STATE_DIR) { $env:MEMD_HOOK_STATE_DIR } else { Join-Path $HOME ".memd/hook_state" })
)

# PostCompact memd restore — Windows counterpart of memd-postcompact-restore.sh.
# NON-BLOCKING. See the bash script for design notes.

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$logPath = Join-Path $StateDir "hook.log"

$inputJson = [Console]::In.ReadToEnd()
$sessionId = "unknown"
try {
  $data = $inputJson | ConvertFrom-Json
  if ($data.session_id) { $sessionId = [string]$data.session_id }
} catch {
}

Add-Content -Path $logPath -Value "[$(Get-Date -Format 'HH:mm:ss')] POST-COMPACT session=$sessionId"

if ($env:MEMD_A4_LEDGER_SURVIVAL -ne "1") {
  Add-Content -Path $logPath -Value "[$(Get-Date -Format 'HH:mm:ss')] POST-COMPACT skipped (MEMD_A4_LEDGER_SURVIVAL=0)"
  exit 0
}

$bundleRoot = if ($env:MEMD_BUNDLE_ROOT) { $env:MEMD_BUNDLE_ROOT } else { ".memd" }
if (-not (Test-Path $bundleRoot)) {
  $fallback = Join-Path $HOME "Documents/projects/memd/.memd"
  if (Test-Path $fallback) { $bundleRoot = $fallback }
}

$memd = Get-Command memd -ErrorAction SilentlyContinue
if (-not $memd) {
  Add-Content -Path $logPath -Value "[$(Get-Date -Format 'HH:mm:ss')] POST-COMPACT memd CLI missing; skip"
  exit 0
}

try {
  if ($env:MEMD_HOOK_ENFORCE -eq "1") {
    & memd hooks enforce --event PostCompact --harness claude-code `
      --session-id $sessionId --output $bundleRoot `
      -- memd hook restore --session-id $sessionId --output $bundleRoot *>> $logPath
  } else {
    & memd hook restore --session-id $sessionId --output $bundleRoot *>> $logPath
  }
  $rc = $LASTEXITCODE
} catch {
  $rc = 1
  Add-Content -Path $logPath -Value "[$(Get-Date -Format 'HH:mm:ss')] POST-COMPACT exception: $_"
}

switch ($rc) {
  0 { Add-Content -Path $logPath -Value "[$(Get-Date -Format 'HH:mm:ss')] POST-COMPACT restore ok" }
  2 { Add-Content -Path $logPath -Value "[$(Get-Date -Format 'HH:mm:ss')] POST-COMPACT no-sealed-ledger (breach logged by CLI)" }
  default { Add-Content -Path $logPath -Value "[$(Get-Date -Format 'HH:mm:ss')] POST-COMPACT restore rc=$rc (non-fatal)" }
}

exit 0
