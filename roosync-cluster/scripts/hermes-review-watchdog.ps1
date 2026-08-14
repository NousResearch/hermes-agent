<#
.SYNOPSIS
    Hermes PR-review activity watchdog — detects SILENT review gaps.

.DESCRIPTION
    The pr-review cron (c8559be3577c, hourly at :23) can complete with status=ok
    yet post ZERO reviews. This silent failure happens when:
      - the model/provider breaks (e.g. the 2026-08-07 z.ai 429 storm + 964s
        timeouts that halted all reviews for hours before the claudish switch),
      - gh auth drifts, or the reviewable-PR heuristic skips everything.

    Cron status does NOT reflect actual review output, so this watchdog checks
    the BUSINESS signal directly: GitHub reviews posted by the bot across the
    target repos. If open PRs exist but no review has been posted within
    AlertThresholdHours, it alerts.

    Designed as an INDEPENDENT observer: it reads GH_TOKEN + TELEGRAM_BOT_TOKEN
    from the host volume and queries the GitHub REST API directly (Invoke-
    RestMethod), so it keeps alerting even when the Hermes container itself is
    down — which is exactly when the gap is most likely.

    Alert channel: Telegram message to the review chat (same channel pr-review
    delivers to). A cooldown (state file) prevents spam: one alert per
    CooldownHours. All runs log to review-watchdog.log.

    Date handling: Invoke-RestMethod auto-deserializes JSON dates to [datetime]
    under the current culture (fr-FR here), which breaks naive subtraction and
    string round-tripping. All timestamps are normalized to UTC [datetime] via
    ConvertTo-UtcDateTime and formatted with InvariantCulture to avoid comma
    decimals ("722,2h" bug).

.NOTES
    Deploy as a Windows Scheduled Task on po-2026:
    - Trigger: every 30 minutes (off-minute, e.g. :07/:37)
    - Action: powershell -NoProfile -File "C:\dev\hermes-agent\roosync-cluster\scripts\hermes-review-watchdog.ps1"
    - Run whether user is logged on or not

    Author: Hermes Agent workspace
#>

param(
    [string[]]$TargetRepos = @("jsboige/CoursIA", "jsboige/roo-extensions"),
    [string]$BotLogin = "jsboige",
    [double]$AlertThresholdHours = 4.0,
    [double]$CooldownHours = 4.0,
    [int]$MaxPrsPerRepo = 20,
    # Host volume (persists across container rebuilds) — read tokens here, NOT
    # from the container, so the watchdog works when the container is down.
    [string]$HostEnvFile = "C:\Users\jsboi\.hermes\.env",
    [string]$ReviewChatId = "-1003904676273",
    [string]$ContainerName = "hermes",
    [string]$LogPath = "$PSScriptRoot\..\logs\review-watchdog.log",
    [string]$StateFile = "$PSScriptRoot\..\logs\review-watchdog-state.json"
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

# Normalize a timestamp (Invoke-RestMethod gives [datetime] in current culture;
# strings come from elsewhere) to a UTC [datetime]. Avoids the op_Subtraction
# failure and comma-decimal display under fr-FR.
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
    return @{ LastAlert = $null; LastReviewSeen = $null; LastRun = $null }
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
        Write-Log "Telegram alert failed: $($_.Exception.Message)" "WARN"
        return $false
    }
}

function Invoke-Gh {
    param([string]$Path)
    try {
        return Invoke-RestMethod -Uri "https://api.github.com/$Path" -Headers $script:GhHeaders -TimeoutSec 20 -ErrorAction Stop
    }
    catch {
        Write-Log "GitHub API call failed ($Path): $($_.Exception.Message)" "WARN"
        return $null
    }
}

# --- setup ---

$ghToken = Get-EnvValue "GH_TOKEN"
if (-not $ghToken) {
    Write-Log "GH_TOKEN missing from $HostEnvFile — cannot check review activity. Aborting." "ERROR"
    exit 1
}
$script:GhHeaders = @{
    Authorization          = "Bearer $ghToken"
    Accept                 = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent"           = "hermes-review-watchdog"
}

# --- collect review activity across target repos ---

$now = [datetime]::UtcNow
$totalOpenPrs = 0
$latestReview = $null   # UTC [datetime] of the most recent bot review seen
$repoSummary = @()

foreach ($repo in $TargetRepos) {
    $pulls = Invoke-Gh "repos/$repo/pulls?state=open&sort=created&direction=desc&per_page=$MaxPrsPerRepo"
    if ($null -eq $pulls) { $repoSummary += "$repo=(query-failed)"; continue }
    $openCount = @($pulls).Count
    $totalOpenPrs += $openCount
    $repoLatest = $null

    foreach ($pr in $pulls) {
        $reviews = Invoke-Gh "repos/$repo/pulls/$($pr.number)/reviews"
        if ($null -eq $reviews) { continue }
        foreach ($rv in ($reviews | Where-Object { $_.user.login -eq $BotLogin })) {
            $t = ConvertTo-UtcDateTime $rv.submitted_at
            if ($null -eq $t) { continue }
            if ($null -eq $repoLatest -or $t -gt $repoLatest) { $repoLatest = $t }
            if ($null -eq $latestReview -or $t -gt $latestReview) { $latestReview = $t }
        }
    }
    $age = if ($repoLatest) { (($now - $repoLatest).TotalHours).ToString("F1", $Invariant) + "h" } else { "never" }
    $repoSummary += "$repo(open=$openCount,lastBotReview=$age)"
}

# --- evaluate ---

$hoursSince = $null
if ($latestReview) { $hoursSince = ($now - $latestReview).TotalHours }

$containerUp = $false
try {
    $st = docker inspect --format='{{.State.Status}}' $ContainerName 2>$null
    if ($LASTEXITCODE -eq 0 -and $st -eq "running") { $containerUp = $true }
} catch { }

$status = "OK"
if ($totalOpenPrs -eq 0) {
    $detail = "no open PRs to review across $($TargetRepos -join ', ')"
}
elseif ($null -eq $hoursSince) {
    $status = "ALERT"
    $detail = "$totalOpenPrs open PR(s) but the bot ($BotLogin) has NEVER posted a review"
}
elseif ($hoursSince -ge $AlertThresholdHours) {
    $status = "ALERT"
    $detail = "no review in $([Math]::Round($hoursSince,1).ToString($Invariant))h (threshold ${AlertThresholdHours}h) despite $totalOpenPrs open PR(s)"
}
else {
    $detail = "last review $([Math]::Round($hoursSince,1).ToString($Invariant))h ago, $totalOpenPrs open PR(s)"
}

$summary = "$status - $detail. Repos: $($repoSummary -join '; '). Container: $(if($containerUp){'up'}else{'DOWN'})."
Write-Log $summary $(if ($status -eq "ALERT") { "ERROR" } else { "INFO" })

# --- alert (with cooldown) ---

$state = Get-State
$shouldAlert = $false
if ($status -eq "ALERT") {
    $cooldownOk = $true
    $lastAlert = ConvertTo-UtcDateTime $state.LastAlert
    if ($lastAlert) {
        if ((($now - $lastAlert).TotalHours) -lt $CooldownHours) { $cooldownOk = $false }
    }
    if ($cooldownOk) {
        $shouldAlert = $true
        $msg = "[REVIEW-WATCHDOG] $summary`n`nThe pr-review cron may be failing silently (status=ok, zero output). Check: gateway logs for 429/timeout/auth errors, gh auth in container, provider health."
        $sent = Send-Telegram $msg
        Write-Log "Alert sent to Telegram chat $ReviewChatId (sent=$sent)." $(if ($sent) { "INFO" } else { "WARN" })
    }
    else {
        Write-Log "Alert suppressed (cooldown ${CooldownHours}h not elapsed since last alert)." "INFO"
    }
}

Set-State @{
    LastRun        = $now.ToString("o", $Invariant)
    LastAlert      = if ($shouldAlert) { $now.ToString("o", $Invariant) } else { $state.LastAlert }
    LastReviewSeen = if ($latestReview) { $latestReview.ToString("o", $Invariant) } else { $null }
}
