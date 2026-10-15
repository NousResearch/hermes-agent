param(
    [string]$HermesHome = $env:HERMES_HOME,
    [string]$WebUiUrl = $env:HERMES_WEB_UI_URL,
    [int]$GatewayTimeoutSeconds = 3,
    [int]$WebUiTimeoutSeconds = 3,
    [switch]$IncludeCodex,
    [switch]$IncludeWeixin
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($HermesHome)) {
    $HermesHome = Join-Path $HOME '.hermes'
}
if ([string]::IsNullOrWhiteSpace($WebUiUrl)) {
    $WebUiUrl = 'http://127.0.0.1:8648'
}

$results = New-Object System.Collections.Generic.List[object]

function Add-Result {
    param(
        [string]$Name,
        [ValidateSet('PASS','WARN','FAIL')][string]$Status,
        [string]$Detail,
        [string]$Action = ''
    )
    $results.Add([pscustomobject]@{
        name = $Name
        status = $Status
        detail = $Detail
        action = $Action
    }) | Out-Null
}

function Test-HttpJson {
    param(
        [string]$Url,
        [int]$TimeoutSeconds
    )
    try {
        return Invoke-RestMethod -Uri $Url -TimeoutSec $TimeoutSeconds
    } catch {
        return $null
    }
}

function Get-ConfigApiServerEndpoint {
    param([string]$ConfigPath)
    $hostValue = '127.0.0.1'
    $portValue = 8642
    if (-not (Test-Path -LiteralPath $ConfigPath)) {
        return [pscustomobject]@{ host = $hostValue; port = $portValue; source = 'default' }
    }

    $lines = Get-Content -LiteralPath $ConfigPath -ErrorAction SilentlyContinue
    for ($index = 0; $index -lt $lines.Count; $index++) {
        $line = $lines[$index]
        if ($line -match '^\s*host\s*:\s*["'']?([^"'']+)["'']?\s*$') {
            $hostValue = $Matches[1].Trim()
        }
        if ($line -match '^\s*port\s*:\s*["'']?(\d+)["'']?\s*$') {
            $portValue = [int]$Matches[1]
        }
    }
    return [pscustomobject]@{ host = $hostValue; port = $portValue; source = $ConfigPath }
}

function Test-PidAlive {
    param([int]$PidValue)
    if ($PidValue -le 0) { return $false }
    try {
        $process = Get-Process -Id $PidValue -ErrorAction Stop
        return $null -ne $process
    } catch {
        return $false
    }
}

$resolvedHermesHome = Resolve-Path -LiteralPath $HermesHome -ErrorAction SilentlyContinue
if ($null -ne $resolvedHermesHome) {
    $hermesHomePath = [System.IO.Path]::GetFullPath($resolvedHermesHome.Path)
} else {
    $hermesHomePath = [System.IO.Path]::GetFullPath($HermesHome)
}
if (Test-Path -LiteralPath $hermesHomePath) {
    Add-Result 'hermes_home' 'PASS' $hermesHomePath
} else {
    Add-Result 'hermes_home' 'FAIL' $hermesHomePath 'Create or point HERMES_HOME at the active Hermes profile.'
}

$configPath = Join-Path $hermesHomePath 'config.yaml'
$endpoint = Get-ConfigApiServerEndpoint -ConfigPath $configPath
if (Test-Path -LiteralPath $configPath) {
    Add-Result 'config_yaml' 'PASS' "api_server=$($endpoint.host):$($endpoint.port)"
} else {
    Add-Result 'config_yaml' 'WARN' "missing: $configPath" 'Run Hermes setup or create config.yaml before starting the gateway.'
}

$pidPath = Join-Path $hermesHomePath 'gateway.pid'
$pidValue = $null
if (Test-Path -LiteralPath $pidPath) {
    try {
        $pidData = Get-Content -LiteralPath $pidPath -Raw | ConvertFrom-Json
        $pidValue = [int]$pidData.pid
        if (Test-PidAlive -PidValue $pidValue) {
            Add-Result 'gateway_pid' 'PASS' "pid=$pidValue"
        } else {
            Add-Result 'gateway_pid' 'WARN' "stale pid=$pidValue" 'Restart gateway or remove stale gateway.pid after confirming the process is gone.'
        }
    } catch {
        Add-Result 'gateway_pid' 'WARN' "unreadable: $pidPath" 'Recreate gateway.pid by restarting the gateway.'
    }
} else {
    Add-Result 'gateway_pid' 'WARN' "missing: $pidPath" 'If gateway is running, restart it or create PID metadata via Hermes gateway management.'
}

$gatewayUrl = "http://$($endpoint.host):$($endpoint.port)"
$gatewayHealth = Test-HttpJson -Url "$gatewayUrl/health" -TimeoutSeconds $GatewayTimeoutSeconds
if ($null -ne $gatewayHealth) {
    Add-Result 'gateway_health' 'PASS' "$gatewayUrl/health"
} else {
    Add-Result 'gateway_health' 'FAIL' "$gatewayUrl/health unreachable" 'Check gateway logs, configured api_server port, and whether another process owns the port.'
}

$gatewayDetailed = Test-HttpJson -Url "$gatewayUrl/health/detailed" -TimeoutSeconds $GatewayTimeoutSeconds
if ($null -ne $gatewayDetailed) {
    Add-Result 'gateway_detailed' 'PASS' "state=$($gatewayDetailed.gateway_state) pid=$($gatewayDetailed.pid)"
} else {
    Add-Result 'gateway_detailed' 'WARN' "$gatewayUrl/health/detailed unavailable" 'Older gateway builds may not expose detailed health; use gateway.log for details.'
}

$webHealth = Test-HttpJson -Url "$($WebUiUrl.TrimEnd('/'))/health" -TimeoutSeconds $WebUiTimeoutSeconds
if ($null -ne $webHealth) {
    Add-Result 'web_ui_health' 'PASS' "$WebUiUrl/health"
} else {
    Add-Result 'web_ui_health' 'WARN' "$WebUiUrl/health unreachable" 'Start hermes-web-ui or check the configured Web UI port.'
}

$envPath = Join-Path $hermesHomePath '.env'
if (Test-Path -LiteralPath $envPath) {
    $envText = Get-Content -LiteralPath $envPath -Raw
    if ($IncludeWeixin) {
        if ($envText -match '(?m)^\s*WEIXIN_HOME_CHANNEL\s*=\s*.+') {
            Add-Result 'weixin_home_channel' 'PASS' 'WEIXIN_HOME_CHANNEL configured'
        } else {
            Add-Result 'weixin_home_channel' 'WARN' 'WEIXIN_HOME_CHANNEL missing' 'Set WEIXIN_HOME_CHANNEL for cron/home-channel delivery.'
        }
    }
    if ($IncludeCodex) {
        if ($envText -match '(?m)^\s*CODEX_GATEWAY_ENABLED\s*=\s*(1|true|yes|on)\s*$') {
            Add-Result 'codex_gateway_enabled' 'PASS' 'CODEX_GATEWAY_ENABLED is truthy'
        } elseif ($envText -match '(?m)^\s*CODEX_GATEWAY_DISABLED\s*=\s*(1|true|yes|on)\s*$') {
            Add-Result 'codex_gateway_enabled' 'WARN' 'CODEX_GATEWAY_DISABLED is truthy' 'Unset CODEX_GATEWAY_DISABLED to allow /codex routing.'
        } else {
            Add-Result 'codex_gateway_enabled' 'WARN' 'not explicitly enabled in .env' 'Set CODEX_GATEWAY_ENABLED=1 for explicit Codex gateway routing.'
        }
    }
} else {
    Add-Result 'hermes_env' 'WARN' "missing: $envPath" 'Create .env when optional gateway integrations require runtime settings.'
}

if ($IncludeCodex) {
    $codexCommand = Get-Command codex -ErrorAction SilentlyContinue
    if ($null -ne $codexCommand) {
        Add-Result 'codex_cli' 'PASS' $codexCommand.Source
    } else {
        Add-Result 'codex_cli' 'WARN' 'codex not found on PATH' 'Install Codex CLI or set CODEX_GATEWAY_COMMAND to its full path.'
    }
}

$gatewayLog = Join-Path $hermesHomePath 'logs/gateway.log'
if (Test-Path -LiteralPath $gatewayLog) {
    $recentErrors = Get-Content -LiteralPath $gatewayLog -Tail 200 | Where-Object { $_ -match '(ERROR|Traceback|Exception|CRITICAL)' } | Select-Object -Last 5
    if ($recentErrors.Count -gt 0) {
        Add-Result 'gateway_recent_errors' 'WARN' ($recentErrors -join ' | ') 'Inspect ~/.hermes/logs/gateway.log before restarting.'
    } else {
        Add-Result 'gateway_recent_errors' 'PASS' 'no ERROR/Traceback/Exception in last 200 lines'
    }
} else {
    Add-Result 'gateway_log' 'WARN' "missing: $gatewayLog" 'Start gateway once to create logs.'
}

$failCount = @($results | Where-Object { $_.status -eq 'FAIL' }).Count
$warnCount = @($results | Where-Object { $_.status -eq 'WARN' }).Count
$summaryStatus = if ($failCount -gt 0) { 'FAIL' } elseif ($warnCount -gt 0) { 'WARN' } else { 'PASS' }

[pscustomobject]@{
    status = $summaryStatus
    generated_at = (Get-Date).ToString('o')
    hermes_home = $hermesHomePath
    web_ui_url = $WebUiUrl
    results = $results
} | ConvertTo-Json -Depth 5

if ($failCount -gt 0) { exit 2 }
if ($warnCount -gt 0) { exit 1 }
exit 0
