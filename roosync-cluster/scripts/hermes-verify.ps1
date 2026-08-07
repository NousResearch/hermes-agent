# hermes-verify.ps1 — 12-point post-op verification
# Usage: .\roosync-cluster\scripts\hermes-verify.ps1
# Runs checks inside the hermes container and reports PASS/FAIL.

$ErrorActionPreference = "Stop"
$Container = "hermes"

function Invoke-Hermes {
    param([string]$Command)
    $result = docker exec $Container bash -c $Command 2>&1
    return $result
}

function Check {
    param([string]$Label, [string]$Result)
    if ($Result -eq "OK") {
        Write-Host "  [PASS] $Label" -ForegroundColor Green
        return $true
    } else {
        Write-Host "  [FAIL] $Label - $Result" -ForegroundColor Red
        return $false
    }
}

$Pass = 0
$Fail = 0

Write-Host "=== HERMES VERIFICATION (12 checks) ===" -ForegroundColor Cyan
Write-Host ""

# 1. Gateway process running
$proc = Invoke-Hermes 'ps aux | grep "hermes gateway run" | grep -v grep | wc -l'
if ($proc -match "([1-9])") { if (Check "Gateway process" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "Gateway process" "not running ($proc)") { $Pass++ } else { $Fail++ } }

# 2. Telegram connected
$tg = Invoke-Hermes 'cat /opt/data/gateway_state.json /opt/data/.hermes/gateway_state.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get(\"platforms\",{}).get(\"telegram\",{}).get(\"state\",\"unknown\"))" 2>/dev/null || echo "NOT_FOUND"'
if ($tg -match "connected") { if (Check "Telegram" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "Telegram" "$tg") { $Pass++ } else { $Fail++ } }

# 3. Config readable via symlink
$cfg = Invoke-Hermes 'head -1 /opt/data/.hermes/config.yaml 2>/dev/null || echo "FAIL"'
if ($cfg -match "model:") { if (Check "Config symlink" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "Config symlink" "$cfg") { $Pass++ } else { $Fail++ } }

# 4. .env readable via symlink + TELEGRAM_BOT_TOKEN non-empty
$envTok = Invoke-Hermes 'grep "^TELEGRAM_BOT_TOKEN=" /opt/data/.hermes/.env 2>/dev/null | cut -d= -f2 | wc -c'
if ($envTok -match "([3-9]\d|[1-9]\d{2,})") { if (Check ".env symlink + token" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check ".env symlink + token" "empty or missing") { $Pass++ } else { $Fail++ } }

# 5. Symlinks intact
$symOk = $true
foreach ($f in @("config.yaml", ".env", ".env.secrets", "cron/jobs.json")) {
    $link = Invoke-Hermes "readlink /opt/data/.hermes/$f 2>/dev/null || echo MISSING"
    if ($link -match "MISSING") {
        if (Check "Symlink .hermes/$f" "MISSING") { $Pass++ } else { $Fail++ }
        $symOk = $false
    }
}
if ($symOk) { if (Check "Symlinks (4)" "OK") { $Pass++ } else { $Fail++ } }

# 6. Model correct
# Phase 2 v3 (2026-06-25): main model is claude-sonnet-4-6 routed via claudish proxy
# (po-2023), which remaps claude-sonnet-* -> gc@glm-5.2. Legacy glm-5-turbo/zai also valid.
$model = Invoke-Hermes 'grep "^  default:" /opt/data/.hermes/config.yaml 2>/dev/null | head -1'
if ($model -match "glm-5-turbo|claude-sonnet-4-6") { if (Check "Model" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "Model" "$model") { $Pass++ } else { $Fail++ } }

# 7. MCP servers in config
$mcp = Invoke-Hermes 'grep -c "mcp_servers:" /opt/data/.hermes/config.yaml 2>/dev/null || echo 0'
if ($mcp -match "([1-9])") { if (Check "MCP servers" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "MCP servers" "not found") { $Pass++ } else { $Fail++ } }

# 8. jobs.json valid
$jobs = Invoke-Hermes 'python3 -c "import json; d=json.load(open(''/opt/data/.hermes/cron/jobs.json'')); print(len(d.get(''jobs'',[])))" 2>/dev/null || echo 0'
if ($jobs -match "([1-9]\d*)") { if (Check "Cron jobs ($jobs)" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "Cron jobs" "none or invalid") { $Pass++ } else { $Fail++ } }

# 9. kanban.db writable (as hermes user)
$kanban = Invoke-Hermes 'python3 -c "
import sqlite3
conn=sqlite3.connect(''/opt/data/.hermes/kanban.db'')
conn.execute(''CREATE TABLE IF NOT EXISTS _wtest (id INTEGER)'')
conn.execute(''DROP TABLE IF EXISTS _wtest'')
conn.commit()
conn.close()
print(''OK'')" 2>/dev/null || echo FAIL'
if ($kanban -match "OK") { if (Check "Kanban DB writable" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "Kanban DB writable" "$kanban") { $Pass++ } else { $Fail++ } }

# 10. gh auth
$gh = Invoke-Hermes 'gh auth status 2>&1 | head -1'
if ($gh -match "Logged in") { if (Check "gh auth" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "gh auth" "$gh") { $Pass++ } else { $Fail++ } }

# 11. Cron enabled count
$enabled = Invoke-Hermes 'python3 -c "
import json
d=json.load(open(''/opt/data/.hermes/cron/jobs.json''))
active=[j for j in d.get(''jobs'',[]) if j.get(''enabled'',True)]
print(len(active))" 2>/dev/null || echo 0'
if ($enabled -ge 3) { if (Check "Active crons ($enabled)" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "Active crons" "only $enabled") { $Pass++ } else { $Fail++ } }

# 12. MCP connection health (no recent "giving up" in logs)
$mcpHealth = Invoke-Hermes 'tail -200 /opt/data/logs/gateways/default/current 2>/dev/null | grep -c "giving up" || true'
$mcpHealth = ($mcpHealth -replace '\D','').Trim()
if ([string]::IsNullOrWhiteSpace($mcpHealth)) { $mcpHealth = "0" }
if ([int]$mcpHealth -eq 0) { if (Check "MCP health" "OK") { $Pass++ } else { $Fail++ } }
else { if (Check "MCP health" "$mcpHealth servers gave up") { $Pass++ } else { $Fail++ } }

# Summary
Write-Host ""
if ($Fail -eq 0) {
    Write-Host "=== ALL $Pass CHECKS PASSED ===" -ForegroundColor Green
    exit 0
} else {
    Write-Host "=== $Pass passed, $Fail FAILED ===" -ForegroundColor Red
    exit 1
}
