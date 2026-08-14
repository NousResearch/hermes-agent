<#
.SYNOPSIS
    Hermes cluster-tour global-post watchdog — detects the silent-skip of ETAPE 3.

.DESCRIPTION
    The cluster-tour cron (job 525e5650a8ac, schedule "13 */12 * * *", fires
    ~00:13/12:13 UTC) must append a [CLUSTER-HEALTH] post on the GLOBAL dashboard
    at every fire (prompt ETAPE 3, marked OBLIGATOIRE/INCONDITIONNEL). Since
    2026-08-12 it silently skips that post non-deterministically while reporting
    status=ok ("completed successfully") — 4 consecutive fires post-T#40 without a
    global post (issue jsboige/hermes-agent #3). Prompt-side emphasis reached its
    limit; a prompt cannot close a prompt-skipping defect.

    This watchdog is the ai-01-specified "organe" (design validated, 5 points):
      1. separate cron shifted ~15min after each cluster-tour fire; NEVER re-fires
         the tour (verifier, not duplicate);
      2. assertion: a [CLUSTER-HEALTH] append on global carries a timestamp
         POSTERIOR to the start of the watched tour;
      3. if missing — in order: (a) post a [WARN] on global where the post should
         have been, (b) DM ai-01 HIGH (via the visible global [WARN] read by
         cluster MCP agents + Telegram as the urgent channel);
      4. never auto re-fire;
      5. instrument the rate: every check writes its verdict (OK/MISSING) to a
         counter-able state file.

    INDEPENDENT observer: reads everything from the HOST volume, so it keeps
    working when the Hermes container itself is down (exactly when the gap is most
    likely). Sources:
      - C:\Users\jsboi\.hermes\cron\jobs.json   → job 525e5650a8ac last_run_at
      - G:\Mon Drive\Synchronisation\RooSync\.shared-state\dashboards\global.md
        → last message block timestamp containing [CLUSTER-HEALTH]

    The [WARN] is appended directly to global.md in the same format the
    roo-state-manager writes (### [<ts>] machine|workspace), so every MCP reader
    of the cluster sees the hole exactly where the post should have been.
    Telegram alert goes to the same review chat as hermes-review-watchdog.ps1.
    A cooldown (state file) prevents spam; every run logs to
    cluster-tour-watchdog.log. Date handling follows the review-watchdog pattern:
    all timestamps normalized to UTC [datetime] via ConvertTo-UtcDateTime
    (avoids the fr-FR Invoke-RestMethod/culture date bugs).

.NOTES
    Deploy as a Windows Scheduled Task on po-2026:
    - Trigger: every 30 minutes (off-minute, e.g. :07/:37)
    - Action: powershell -NoProfile -File "C:\dev\hermes-agent\roosync-cluster\scripts\hermes-cluster-tour-watchdog.ps1"
    - Run whether user is logged on or not

    Author: Hermes Agent workspace (operator Claude Code po-2026)
    Issue:  jsboige/hermes-agent #3
#>

param(
    [string]$ClusterTourJobId = "525e5650a8ac",
    [string]$JobsFile = "C:\Users\jsboi\.hermes\cron\jobs.json",
    [string]$GlobalDashFile = "G:\Mon Drive\Synchronisation\RooSync\.shared-state\dashboards\global.md",
    [string]$HostEnvFile = "C:\Users\jsboi\.hermes\.env",
    [string]$ReviewChatId = "-1003904676273",
    [double]$PostMarginMinutes = 15.0,   # how long after a fire the global post may appear
    [double]$CooldownMinutes = 120.0,    # min gap between alerts
    [string]$LogPath = "$PSScriptRoot\..\logs\cluster-tour-watchdog.log",
    [string]$StateFile = "$PSScriptRoot\..\logs\cluster-tour-watchdog-state.json"
)

$ErrorActionPreference = "Stop"
$Invariant = [Globalization.CultureInfo]::InvariantCulture

# --- helpers ---

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ", $Invariant)
    $entry = "[$timestamp] [$Level] $Message"
    Write-Output $entry
    $logDir = Split-Path $LogPath -Parent
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    Add-Content -Path $LogPath -Value $entry -Encoding UTF8
}

function Get-EnvValue {
    param([string]$Key)
    if (-not (Test-Path $HostEnvFile)) { return $null }
    $line = Get-Content $HostEnvFile -ErrorAction SilentlyContinue |
        Where-Object { $_ -match "^$Key=" } | Select-Object -First 1
    if ($line) { return ($line -replace "^$Key=", "") }
    return $null
}

function ConvertTo-UtcDateTime {
    param($Value)
    if ($null -eq $Value -or "$Value" -eq "") { return $null }
    if ($Value -is [datetime]) { return $Value.ToUniversalTime() }
    if ($Value -is [DateTimeOffset]) { return $Value.UtcDateTime }
    try {
        return [datetime]::Parse("$Value", $Invariant, [Globalization.DateTimeStyles]::AdjustToUniversal)
    } catch { return $null }
}

function Get-State {
    if (Test-Path $StateFile) {
        try { return Get-Content $StateFile -Raw | ConvertFrom-Json } catch { }
    }
    return @{ LastCheckedRunAt = $null; LastAlertAt = $null; OkCount = 0; MissingCount = 0; LastRun = $null }
}

function Set-State {
    param([hashtable]$State)
    $State | ConvertTo-Json | Set-Content $StateFile -Encoding UTF8
}

function Send-Telegram {
    param([string]$Text)
    $token = Get-EnvValue "TELEGRAM_BOT_TOKEN"
    if (-not $token) { Write-Log "TELEGRAM_BOT_TOKEN missing from $HostEnvFile — cannot alert." "WARN"; return $false }
    try {
        $body = @{ chat_id = $ReviewChatId; text = $Text; disable_web_page_preview = $true } | ConvertTo-Json
        # PS 5.1 re-encodes a string body to ANSI (cp1252) on send, which breaks
        # any non-ASCII char ("Bad Request: strings must be encoded in UTF-8").
        # Send pre-encoded UTF-8 bytes instead.
        $jsonBytes = [System.Text.Encoding]::UTF8.GetBytes($body)
        $null = Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
            -Method Post -Body $jsonBytes -ContentType "application/json; charset=utf-8" -TimeoutSec 15
        return $true
    }
    catch {
        Write-Log "Telegram alert failed: $($_.Exception.Message)" "WARN" | Out-Null
        return $false
    }
}

# Last cluster-tour fire time from jobs.json (host volume — works container-down).
function Get-ClusterTourLastRunAt {
    if (-not (Test-Path $JobsFile)) { Write-Log "jobs.json not found at $JobsFile" "WARN"; return $null }
    try {
        $data = Get-Content $JobsFile -Raw | ConvertFrom-Json
        foreach ($job in $data.jobs) {
            if ("$($job.id)" -eq $ClusterTourJobId -and $job.name -like "*cluster-tour*") {
                return ConvertTo-UtcDateTime $job.last_run_at
            }
        }
        Write-Log "cluster-tour job $ClusterTourJobId not found in jobs.json" "WARN"
        return $null
    } catch {
        Write-Log "jobs.json parse failed: $($_.Exception.Message)" "WARN"
        return $null
    }
}

# Timestamp of the last message block in global.md whose content contains
# "[CLUSTER-HEALTH]". Blocks are formatted "### [<ts>] machine|workspace".
function Get-LastClusterHealthTimestamp {
    if (-not (Test-Path $GlobalDashFile)) { Write-Log "global.md not found at $GlobalDashFile" "WARN"; return $null }
    try {
        $text = Get-Content $GlobalDashFile -Raw
        $pattern = '(?m)^### \[([0-9T:\-\.]+Z)\][^\r\n]*\r?\n([\s\S]*?)(?=^### \[|\z)'
        $blocks = [regex]::Matches($text, $pattern)
        $lastTs = $null
        foreach ($m in $blocks) {
            $blockBody = $m.Groups[2].Value
            # Match the tour's section TITLE only ("## [CLUSTER-HEALTH] T#N"), NOT a
            # bare mention of the string — the [WARN] body itself says "no append
            # [CLUSTER-HEALTH]" and would otherwise self-validate as a posted tour.
            if ($blockBody -match "(?m)^## \[CLUSTER-HEALTH\]") {
                $t = ConvertTo-UtcDateTime $m.Groups[1].Value
                if ($null -ne $t) { $lastTs = $t }
            }
        }
        return $lastTs
    } catch {
        Write-Log "global.md parse failed: $($_.Exception.Message)" "WARN"
        return $null
    }
}

# Append a [WARN] on global.md in the roo-state-manager intercom format so the
# hole is visible exactly where the tour post should have been. Read by every
# MCP dashboard reader of the cluster (ai-01 included) — no MCP proxy needed.
function Append-GlobalWarn {
    param([string]$TourTs)
    if (-not (Test-Path $GlobalDashFile)) { Write-Log "global.md not found — cannot append [WARN]" "WARN"; return $false }
    try {
        $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ", $Invariant)
        $warn = @(
            "",
            "### [$ts] myia-po-2026|hermes-agent",
            "",
            "## [WARN] cluster-tour ETAPE 3 non exécutée — tour $TourTs",
            "",
            "**Watchdog:** le fire `hermes-cluster-tour` ($TourTs UTC) a rapporté `status=ok` mais aucun append `[CLUSTER-HEALTH]` sur ce dashboard global dans les $PostMarginMinutes min suivantes. Silent-skip du post global (issue jsboige/hermes-agent #3). Vérification programmatique post-cron. Le tour n'est PAS re-firé (spec ai-01).",
            ""
        ) -join "`n"
        Add-Content -Path $GlobalDashFile -Value $warn -Encoding UTF8
        return $true
    } catch {
        Write-Log "global.md append failed: $($_.Exception.Message)" "WARN" | Out-Null
        return $false
    }
}

# --- main ---

$now = [datetime]::UtcNow
$state = Get-State

$lastRunAt = Get-ClusterTourLastRunAt
if ($null -eq $lastRunAt) {
    Write-Log "No cluster-tour last_run_at — cannot assert (source unavailable, not a MISSING)." "WARN"
    Set-State @{
        LastCheckedRunAt = $state.LastCheckedRunAt
        LastAlertAt      = $state.LastAlertAt
        OkCount          = $state.OkCount
        MissingCount     = $state.MissingCount
        LastRun          = $now.ToString("o", $Invariant)
    }
    exit 0
}

$lastHealth = Get-LastClusterHealthTimestamp

# Only assert fires that are old enough for the post to have appeared AND that
# we have not already checked.
$prevChecked = ConvertTo-UtcDateTime $state.LastCheckedRunAt
$isNewFire    = $null -eq $prevChecked -or $lastRunAt -gt $prevChecked
$fireAgeMin   = ($now - $lastRunAt).TotalMinutes

$verdict = "SKIP"
$detail = ""
if ($isNewFire -and $fireAgeMin -ge $PostMarginMinutes) {
    $postOk = $null -ne $lastHealth -and $lastHealth -ge $lastRunAt.AddMinutes(-5)
    if ($postOk) {
        $verdict = "OK"
        $detail = "global [CLUSTER-HEALTH] @ $($lastHealth.ToString('o',$Invariant)) >= fire @ $($lastRunAt.ToString('o',$Invariant))"
    } else {
        $verdict = "MISSING"
        $lastHealthStr = if ($lastHealth) { $lastHealth.ToString("o", $Invariant) } else { "never" }
        $detail = "fire @ $($lastRunAt.ToString('o',$Invariant)) -> NO global [CLUSTER-HEALTH] (last seen $lastHealthStr)"
    }
} elseif ($isNewFire -and $fireAgeMin -lt $PostMarginMinutes) {
    $verdict = "PENDING"
    $detail = "fire @ $($lastRunAt.ToString('o',$Invariant)) still within ${PostMarginMinutes}min post window"
} else {
    $detail = "no new fire since last check ($prevChecked)"
}

Write-Log "verdict=$verdict | $detail | last_run_at=$($lastRunAt.ToString('o',$Invariant)) | last_health=$($lastHealth.ToString('o',$Invariant)) | ok=$($state.OkCount) missing=$($state.MissingCount)"

$okCount    = $state.OkCount
$missingCount = $state.MissingCount
$newAlertAt = $state.LastAlertAt

if ($verdict -eq "OK") {
    $okCount = [int]$state.OkCount + 1
} elseif ($verdict -eq "MISSING") {
    $missingCount = [int]$state.MissingCount + 1

    $cooldownOk = $true
    $lastAlert = ConvertTo-UtcDateTime $state.LastAlertAt
    if ($lastAlert) {
        if ((($now - $lastAlert).TotalMinutes) -lt $CooldownMinutes) { $cooldownOk = $false }
    }

    $warnAppended = Append-GlobalWarn $lastRunAt.ToString("o", $Invariant)
    Write-Log "MISSING: [WARN] appended to global.md (appended=$warnAppended)." $(if ($warnAppended) { "INFO" } else { "ERROR" })

    if ($cooldownOk) {
        $msg = "[CLUSTER-TOUR-WATCHDOG] MISSING — le fire cluster-tour @ $($lastRunAt.ToString('o',$Invariant)) n'a pas posté [CLUSTER-HEALTH] sur global. Verdict count: OK=$okCount MISSING=$missingCount. [WARN] appended on global.md. Voir issue jsboige/hermes-agent #3."
        $sent = Send-Telegram $msg
        Write-Log "Telegram alert sent (sent=$sent)." $(if ($sent) { "INFO" } else { "WARN" })
        $newAlertAt = $now.ToString("o", $Invariant)
    } else {
        Write-Log "Alert suppressed (cooldown ${CooldownMinutes}min not elapsed)." "INFO"
    }
}

Set-State @{
    LastCheckedRunAt = if ($isNewFire) { $lastRunAt.ToString("o", $Invariant) } else { $state.LastCheckedRunAt }
    LastAlertAt      = $newAlertAt
    OkCount          = $okCount
    MissingCount     = $missingCount
    LastRun          = $now.ToString("o", $Invariant)
}
