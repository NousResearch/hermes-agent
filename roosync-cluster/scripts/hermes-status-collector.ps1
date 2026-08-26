<#
.SYNOPSIS
    Hermes status collector — couche 1 de l'organe durable de surveillance.
.DESCRIPTION
    Déterministe, sans LLM, survit reboot + mort du bot + mort de session.
    S'exécute sur le HOST (a accès au docker daemon) et écrit host-status.json
    dans le volume persistant `~/.hermes/host-status/` (= /opt/data/host-status
    dans le container).

    Sections collectées :
      1. Container Hermes (docker inspect : Running, StartedAt, Image)
      2. Gateway (grep ps dans le container)
      3. Crons bot (cat /opt/data/cron/jobs.json) — vu via docker exec
      4. Reviews gap (last_run_at des executions.db)
      5. Watchdogs host (schtasks Result + LastResultTime)
      6. Bus RooSync (probe G: côté host)
      7. Cluster-tour T#N (lecture MD dashboard)
      8. Erreurs container (docker logs depuis 24h)

    Issue: jboige/hermes-agent #5 (organe durable — couche 1)
#>

param(
    [string]$ContainerName = "hermes",
    [string]$StatusDir     = "$env:USERPROFILE\.hermes\host-status",
    [string]$LogDir        = "C:\dev\hermes-agent\roosync-cluster\logs",
    [int]$CronStaleMinutes           = 120,
    [int]$ReviewsGapMinutesThreshold = 240,
    [int]$BotWriteStaleMinutes       = 120,
    [int]$ContainerRestartThreshold  = 3,
    [int]$AlertCooldownMinutes       = 60,
    [string]$DashCoordPath = "C:\Users\jsboi\.hermes\dashboards\workspace-cluster-coordination.md",
    [string]$DashGlobalPath = "C:\Users\jsboi\.hermes\dashboards\global.md"
)

$ErrorActionPreference = "Continue"

# === Logging ===
$LogFile   = Join-Path $LogDir "status-collector.log"
$StateFile = Join-Path $LogDir "status-collector-state.json"
if (-not (Test-Path $LogDir))   { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
if (-not (Test-Path $StatusDir)){ New-Item -ItemType Directory -Path $StatusDir -Force | Out-Null }

function Log([string]$msg, [string]$lvl = "INFO") {
    $ts = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
    $line = "[$ts] [$lvl] $msg"
    Write-Output $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

# === Helpers ===

function DockerExec([string]$cmd) {
    # Renvoie un tableau de lignes (sortie brute). Vide si exec fail.
    $out = & docker exec $ContainerName sh -c $cmd 2>&1
    # Normalisation : si PS a concaténé plusieurs items dans une string,
    # splitter par ligne non-vide
    if ($null -eq $out) { return @() }
    if ($out -isnot [array]) { $out = @($out) }
    $lines = @()
    foreach ($x in $out) {
        if ($null -eq $x) { continue }
        foreach ($l in ($x -split "`n")) {
            $t = $l.Trim()
            if ($t.Length -gt 0) { $lines += $t }
        }
    }
    return $lines
}

function DockerInspect([string]$template) {
    $r = & docker inspect $ContainerName --format $template 2>&1
    if ($LASTEXITCODE -ne 0) { return "" }
    return ($r | Out-String).Trim()
}

function Jo([string[]]$arr) { if ($arr -is [array]) { ($arr -join "`n") } else { $arr } }

# === Sections ===

$ts_iso = (Get-Date).ToUniversalTime().ToString("o")

# 1. Container
$running_str = DockerInspect '{{.State.Running}}'
$started_str = DockerInspect '{{.State.StartedAt}}'
$restarts_n  = DockerInspect '{{.RestartCount}}'
$image_str   = DockerInspect '{{.Image}}'

if ($running_str -eq "true") {
    $started_dt  = [datetime]$started_str
    $uptime_min  = [int]([datetime]::UtcNow - $started_dt.ToUniversalTime()).TotalMinutes
    # Image par tag (pas digest) — docker ps --format "{{.Image}}" rend le tag
    $imageTag = (& docker ps --filter "name=$ContainerName" --format "{{.Image}}" 2>&1 | Out-String).Trim()
    if (-not $imageTag) { $imageTag = $image_str }
    $container_json = @{
        running    = $true
        image      = $imageTag
        started_at = $started_str
        uptime_min = $uptime_min
        restarts   = [int]$restarts_n
    } | ConvertTo-Json -Compress
} else {
    $container_json = @{ running = $false; image = $image_str; status = "$running_str" } | ConvertTo-Json -Compress
}

# 2. Gateway — chaîne const pour éviter interpolation PS du $
$gwCmd = 'ps -eo pid,cmd 2>/dev/null | awk ''/[g]ateway run/ {print $1; exit}'''
$gwOut = DockerExec $gwCmd
$gw_pid = $null
foreach ($line in $gwOut) {
    if ($line -match "^\s*(\d+)\s*$") { $gw_pid = [int]$Matches[1]; break }
}
$gateway_json = if ($gw_pid) {
    @{ running = $true; pid = $gw_pid } | ConvertTo-Json -Compress
} else {
    '{"running":false}' -as [string]
}

# 3. Crons — parse jobs.json via python (les valeurs 'kind'/'interval' nécessitent un parser dédié)
$jobsOut = DockerExec "cat /opt/data/cron/jobs.json"
$jobsTxt = if ($jobsOut.Count -gt 0) { ($jobsOut -join "`n") } else { "" }
$crons_json = if ($jobsTxt.Length -gt 0) {
    $tmp = Join-Path $env:TEMP "jobs.json"
    [System.IO.File]::WriteAllText($tmp, $jobsTxt, [System.Text.UTF8Encoding]::new($false))
    # Python inline via PS — robuste sur PS 5 et 7
    $script = @'
import json, sys
from datetime import datetime, timezone
THR = int(sys.argv[1])
now = datetime.now(timezone.utc)
doc = json.loads(open(sys.argv[2], encoding='utf-8-sig').read())
jobs = doc.get('jobs', doc) if isinstance(doc, dict) else doc
rows = []
for j in (jobs if isinstance(jobs, list) else jobs.values()):
    last = j.get('last_run_at')
    stale = None
    if last:
        try:
            dt = datetime.fromisoformat(last.replace('Z','+00:00'))
            stale = int((now - dt).total_seconds() / 60)
        except Exception:
            pass
    rows.append({
        'id': j.get('id',''),
        'name': j.get('name') or j.get('id') or '?',
        'last_run_at': last,
        'minutes_since': stale,
        'status': j.get('last_status') or j.get('status'),
        'stale_over_threshold': (stale is not None and stale > THR),
    })
sys.stdout.write(json.dumps(rows, ensure_ascii=False))
'@
    $scriptPath = Join-Path $env:TEMP "_crons.py"
    [System.IO.File]::WriteAllText($scriptPath, $script, [System.Text.UTF8Encoding]::new($false))
    $r = python $scriptPath $CronStaleMinutes $tmp 2>&1
    if ($LASTEXITCODE -eq 0) { ($r | Out-String).Trim() } else { "[]" }
} else { "[]" }

# 4. Reviews gap — depuis le statefile host (les watchdogs l'écrivent sur le volume)
$rstatePath = Join-Path $LogDir "review-watchdog-state.json"
$reviews_json = "unknown"
if (Test-Path $rstatePath) {
    try {
        $o = Get-Content $rstatePath -Raw | ConvertFrom-Json
        $last = $o.LastReviewSeen
        if ($last) {
            try {
                $dt = [datetime]$last
                $gap = [int](([datetime]::UtcNow - $dt.ToUniversalTime()).TotalMinutes)
                $reviews_json = @{
                    known           = $true
                    last_seen       = $last
                    gap_min         = $gap
                    over_threshold  = ($gap -gt $ReviewsGapMinutesThreshold)
                } | ConvertTo-Json -Compress
            } catch {
                $reviews_json = '{"known":true,"last_seen":"' + $last + '","error":"parse"}'
            }
        } else { $reviews_json = '{"known":true,"last_seen":null}' }
    } catch { $reviews_json = '{"known":false,"error":"statefile unparseable"}' }
}

# 5. Watchdogs host (schtasks)
function Watch-Result([string]$Name) {
    $flt = '"' + $Name + '"'
    $line = & schtasks /query /tn $flt /fo list /v 2>&1 | Select-String -Pattern 'Dernier r[ée]sultat\s*:\s*(\S+)' | Select-Object -First 1
    if ($line -and $line.Matches.Groups[1].Value -match "^\d+$") {
        $v = [int]$line.Matches.Groups[1].Value
        return @{ name = $Name; result = $v; ok = ($v -eq 0) } | ConvertTo-Json -Compress
    }
    @{ name = $Name; result = $null; ok = $false; error = "schtasks failed" } | ConvertTo-Json -Compress
}

$watchdogs_arr = @(
    Watch-Result "Hermes-Review-Watchdog"
    Watch-Result "Hermes-MCP-Watchdog"
    Watch-Result "Hermes-ClusterTour-Watchdog"
)
$watchdogs_json = "[$($watchdogs_arr -join ',')]"

# 6. Bus : probe G: côté host (le host win voit G: si GoogleDriveFS monté)
$g_ok = $false
try {
    $probe = & cmd /c "dir G:\ /b 1>nul 2>nul & echo OK or echo BAD" 2>&1
    $probeTxt = ($probe -join "`n").Trim()
    if ($probeTxt -match "OK" -and $probeTxt -notmatch "BAD") { $g_ok = $true }
} catch {}

# 7. Latest bot write — extraction regex du dernier message signé (po-2026 ou ai-01) sur coord
$latest_bot_json = "null"
if ($g_ok -and (Test-Path $DashCoordPath)) {
    try {
        $lines = Get-Content $DashCoordPath -Encoding UTF8 -ErrorAction Stop
        $pattern = '^### \[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\] ((?:po-2026|myia-ai-01|myia-po-2026)\|[\w\-]+)'
        $last = $null
        foreach ($ln in $lines) {
            if ($ln -match $pattern) { $last = @{ ts = $matches[1]; machine_ws = $Matches[2] } }
        }
        if ($last) {
            $dt = [datetime]$last.ts
            $gap = [int](([datetime]::UtcNow - $dt.ToUniversalTime()).TotalMinutes)
            $latest_bot_json = (@{
                ts                  = $last.ts
                author              = $last.machine_ws
                gap_min             = $gap
                stale_over_threshold= ($gap -gt $BotWriteStaleMinutes)
            } | ConvertTo-Json -Compress)
        }
    } catch {}
}
$bus_json = @{
    g_drive_ok         = $g_ok
    latest_bot_write   = $latest_bot_json
} | ConvertTo-Json -Compress

# 8. Cluster-tour T#N
$tour_json = '{"known":false}'
if ($g_ok -and (Test-Path $DashGlobalPath)) {
    try {
        $lines = Get-Content $DashGlobalPath -Encoding UTF8 -ErrorAction Stop
        $pat = '## \[CLUSTER-HEALTH\] T#(\d+) — (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z)'
        $lastT = $null
        foreach ($ln in $lines) {
            if ($ln -match $pat) { $lastT = @{ t = [int]$matches[1]; ts = $matches[2] } }
        }
        if ($lastT) {
            $dt = [datetime]$lastT.ts
            $gap = [int](([datetime]::UtcNow - $dt.ToUniversalTime()).TotalMinutes)
            $tour_json = (@{
                known    = $true
                t        = $lastT.t
                ts       = $lastT.ts
                gap_min  = $gap
            } | ConvertTo-Json -Compress)
        }
    } catch {}
}

# 9. Erreurs container (24h)
$err_count = 0
$logs = & docker logs $ContainerName --since 24h 2>&1
if ($logs) {
    if ($logs -isnot [array]) { $logs = @($logs) }
    $combined = ($logs -join "`n")
    $err_count = ([regex]::Matches($combined, 'model_dump|Streaming failed|\b429\b|\b401\b|crash|traceback')).Count
}
$errors_json = @{ ok = $true; error_count_24h = $err_count } | ConvertTo-Json -Compress

# === Compose final ===
# ConvertFrom-Json emballe les arrays en {value, Count} — on déballe en re-stringifiant
function DeWrap {
    param([string]$Json)
    $o = $Json | ConvertFrom-Json
    if ($o -is [array]) { return ,@($o) }
    if ($o.PSObject.Properties.Name -contains 'value' -and $o.PSObject.Properties.Name -contains 'Count') {
        return ,@($o.value)
    }
    return ,$o
}

$container_obj  = $container_json | ConvertFrom-Json
$gateway_obj    = $gateway_json | ConvertFrom-Json
$crons_arr      = DeWrap $crons_json
$reviews_obj    = $reviews_json | ConvertFrom-Json
$watchdogs_arr  = DeWrap $watchdogs_json
$bus_obj        = $bus_json | ConvertFrom-Json
$tour_obj       = $tour_json | ConvertFrom-Json
$errors_obj     = $errors_json | ConvertFrom-Json

$status = [ordered]@{
    timestamp_utc  = $ts_iso
    container      = $container_obj
    gateway        = $gateway_obj
    crons          = $crons_arr
    reviews_gap    = $reviews_obj
    watchdogs      = $watchdogs_arr
    bus            = $bus_obj
    cluster_tour   = $tour_obj
    error_patterns = $errors_obj
}

$status | ConvertTo-Json -Depth 8
$jsonPath = Join-Path $StatusDir "host-status.json"
$jsonOut = & { [System.IO.File]::WriteAllText($jsonPath, ($status | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false)) ; Write-Output "ok" } 2>&1
Log "Collected: container=$([bool]$status.container.running) gateway=$([bool]$status.gateway.running) reviews_gap=$($status.reviews_gap.gap_min)m"

# === Push JSON dans le volume (pour la couche 2 = bot) ===
$volumePath = "/opt/data/host-status/host-status.json"
$tmpFile = Join-Path $env:TEMP "host-status.json"
Copy-Item (Join-Path $StatusDir "host-status.json") $tmpFile -Force
try {
    & docker cp $tmpFile "${ContainerName}:${volumePath}" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Log "Pushed host-status.json to ${ContainerName}:${volumePath}"
    } else {
        # Fallback base64
        $b64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($tmpFile))
        $cmd = "mkdir -p /opt/data/host-status && base64 -d > $volumePath <<< '$b64'"
        DockerExec $cmd | Out-Null
        Log "Pushed host-status.json (base64 fallback)"
    }
} catch {
    Log "docker cp failed: $_" "WARN"
}

# === Rotation hôte : garder 144 snapshots (72h × 30min) ===
$rotation = Join-Path $StatusDir "host-status-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
Copy-Item (Join-Path $StatusDir "host-status.json") $rotation -Force
Get-ChildItem $StatusDir -Filter "host-status-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -Skip 144 | Remove-Item -Force

# === Alertes Telegram sur critères durs ===
$credFile = "C:\Users\jsboi\.env.secrets"
$tgToken = ""
$tgChat  = ""
if (Test-Path $credFile) {
    Get-Content $credFile | ForEach-Object {
        if ($_ -match "^TELEGRAM_BOT_TOKEN=(.+)$") { $tgToken = $matches[1] }
        elseif ($_ -match "^TELEGRAM_REVIEW_CHAT=(.+)$") { $tgChat = $matches[1] }
    }
}

function Should-Alert([string]$k) {
    if (-not (Test-Path $StateFile)) { return $true }
    try {
        $o = Get-Content $StateFile -Raw | ConvertFrom-Json
        if (-not $o.LastAlertAt) { return $true }
        $last = $o.LastAlertAt.$k
        if (-not $last) { return $true }
        return ([datetime]::UtcNow - [datetime]$last).TotalMinutes -ge $AlertCooldownMinutes
    } catch { return $true }
}

function Mark-Alert([string]$k) {
    $o = if (Test-Path $StateFile) { Get-Content $StateFile -Raw | ConvertFrom-Json } else { [pscustomobject]@{} }
    if (-not $o.LastAlertAt) { $o | Add-Member -NotePropertyName LastAlertAt -NotePropertyValue ([pscustomobject]@{}) }
    $o.LastAlertAt | Add-Member -NotePropertyName $k -NotePropertyValue ((Get-Date).ToUniversalTime().ToString("o")) -Force
    [System.IO.File]::WriteAllText($StateFile, ($o | ConvertTo-Json -Depth 5), [System.Text.UTF8Encoding]::new($false))
}

function Send-Tg([string]$msg) {
    if (-not $tgToken -or -not $tgChat) { return }
    $safe = $msg -replace '&','&amp;' -replace '<','&lt;' -replace '>','&gt;'
    $payload = @{ chat_id = $tgChat; text = $safe; parse_mode = "HTML" } | ConvertTo-Json -Compress
    try {
        Invoke-RestMethod -Uri "https://api.telegram.org/bot$tgToken/sendMessage" -Method Post -Body $payload -ContentType "application/json" -TimeoutSec 10 | Out-Null
        Write-Output "tg ok"
    } catch {
        Log "Telegram failed: $_" "WARN"
    }
}

# Critère (a) container down / spiral
if (-not $status.container.running) {
    if (Should-Alert "container-down") {
        Send-Tg "🚨 [Hermes] container DOWN — $($status.container.status)"
        Mark-Alert "container-down"
    }
} elseif ($status.container.restarts -gt $ContainerRestartThreshold) {
    if (Should-Alert "container-restart") {
        Send-Tg "⚠️ [Hermes] container restart spiral ($($status.container.restarts))"
        Mark-Alert "container-restart"
    }
}
# Critère (b) reviews > 4h
if ($status.reviews_gap.known -and $status.reviews_gap.over_threshold) {
    if (Should-Alert "reviews") {
        Send-Tg "⚠️ [Hermes] reviews gap $($status.reviews_gap.gap_min)m (>=$ReviewsGapMinutesThreshold)"
        Mark-Alert "reviews"
    }
}
# Critère (e) bus MCP possiblement down
if ($status.bus.g_drive_ok -and $null -ne $status.bus.latest_bot_write -and $status.bus.latest_bot_write.stale_over_threshold) {
    if (Should-Alert "bus") {
        Send-Tg "🚨 [Hermes] bus MCP possiblement DOWN — last write $($status.bus.latest_bot_write.gap_min)m"
        Mark-Alert "bus"
    }
}
# Watchdogs Failed
foreach ($w in $status.watchdogs) {
    if ($null -ne $w.result -and $w.result -ne 0) {
        $k = "watchdog-$($w.name)-$($w.result)"
        if (Should-Alert $k) {
            Send-Tg "⚠️ [Hermes] watchdog $($w.name) result=$($w.result)"
            Mark-Alert $k
        }
    }
}

Log "OK — criteria checked: a=down? $(-not $status.container.running); b=$($status.reviews_gap.over_threshold); e=$($status.bus.latest_bot_write.stale_over_threshold)"
