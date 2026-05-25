param(
  [int]$SaveInterval = $(if ($env:MEMD_SAVE_INTERVAL) { [int]$env:MEMD_SAVE_INTERVAL } else { 15 }),
  [string]$StateDir = $(if ($env:MEMD_HOOK_STATE_DIR) { $env:MEMD_HOOK_STATE_DIR } else { Join-Path $HOME ".memd/hook_state" })
)

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

$inputJson = [Console]::In.ReadToEnd()
$data = $null
try {
  $data = $inputJson | ConvertFrom-Json
} catch {
  "{}"
  exit 0
}

$sessionId = if ($data.session_id) { [string]$data.session_id } else { "unknown" }
$stopHookActive = if ($null -ne $data.stop_hook_active) { [string]$data.stop_hook_active } else { "False" }
$transcriptPath = if ($data.transcript_path) { [string]$data.transcript_path } else { "" }

if ($stopHookActive -eq "True" -or $stopHookActive -eq "true") {
  "{}"
  exit 0
}

$exchangeCount = 0
if ($transcriptPath -and (Test-Path $transcriptPath)) {
  Get-Content $transcriptPath | ForEach-Object {
    try {
      $entry = $_ | ConvertFrom-Json
      $msg = $entry.message
      if ($msg -and $msg.role -eq "user") {
        $content = $msg.content
        if ($content -is [string] -and $content.Contains("<command-message>")) {
          return
        }
        $script:exchangeCount += 1
      }
    } catch {
    }
  }
}

$lastSaveFile = Join-Path $StateDir "${sessionId}_last_save"
$lastSave = 0
if (Test-Path $lastSaveFile) {
  $lastSave = [int](Get-Content $lastSaveFile | Select-Object -First 1)
}

$sinceLast = $exchangeCount - $lastSave
Add-Content -Path (Join-Path $StateDir "hook.log") -Value "[$(Get-Date -Format 'HH:mm:ss')] Session $sessionId: $exchangeCount exchanges, $sinceLast since last memd save"

if ($sinceLast -ge $SaveInterval -and $exchangeCount -gt 0) {
  Set-Content -Path $lastSaveFile -Value $exchangeCount -Encoding UTF8
  Add-Content -Path (Join-Path $StateDir "hook.log") -Value "[$(Get-Date -Format 'HH:mm:ss')] TRIGGERING memd stop save at exchange $exchangeCount"
  @'
{
  "decision": "block",
  "reason": "AUTO-SAVE checkpoint. Before stopping, persist the important state from this session into memd. Prefer compact truth over summary sludge: 1. run memd checkpoint for the current task state, 2. write any durable decisions/corrections/preferences, 3. if you have a compaction packet or turn-state delta, run memd hook spill --output .memd --stdin --apply, 4. then continue."
}
'@
} else {
  "{}"
}
