param(
  [string]$StateDir = $(if ($env:MEMD_HOOK_STATE_DIR) { $env:MEMD_HOOK_STATE_DIR } else { Join-Path $HOME ".memd/hook_state" })
)

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$inputJson = [Console]::In.ReadToEnd()
$sessionId = "unknown"
try {
  $data = $inputJson | ConvertFrom-Json
  if ($data.session_id) { $sessionId = [string]$data.session_id }
} catch {
}

Add-Content -Path (Join-Path $StateDir "hook.log") -Value "[$(Get-Date -Format 'HH:mm:ss')] PRE-COMPACT memd save triggered for session $sessionId"

@'
{
  "decision": "block",
  "reason": "COMPACTION IMMINENT. Persist everything important to memd before context is compressed. 1. checkpoint current task state, 2. write durable decisions/corrections/preferences/facts, 3. run memd hook spill --output .memd --stdin --apply for any compaction packet or turn-state delta, 4. then allow compaction."
}
'@
