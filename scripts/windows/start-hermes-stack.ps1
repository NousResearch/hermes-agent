param()

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
. (Join-Path $ScriptDir "Resolve-CanonicalHermesHome.ps1")

$HermesHome = Resolve-CanonicalHermesHome -RepoRoot $RepoRoot

$LogDir = Join-Path $HermesHome "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Get-BoolEnvOrDefault {
    param(
        [string]$Name,
        [bool]$DefaultValue
    )
    $raw = [Environment]::GetEnvironmentVariable($Name)
    if (-not $raw) { return $DefaultValue }
    $v = $raw.Trim().ToLowerInvariant()
    if ($v -in @("1", "true", "yes", "on")) { return $true }
    if ($v -in @("0", "false", "no", "off")) { return $false }
    return $DefaultValue
}

function Get-EnvValue {
    param([string]$Name)
    $procValue = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ($procValue -and $procValue.Trim()) {
        return $procValue.Trim()
    }
    $userValue = [Environment]::GetEnvironmentVariable($Name, "User")
    if ($userValue -and $userValue.Trim()) {
        return $userValue.Trim()
    }
    return ""
}

function Is-CommandRunning {
    param([string]$Needle)
    try {
        $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
        foreach ($p in $procs) {
            if ($p.CommandLine -and $p.CommandLine -match [regex]::Escape($Needle)) {
                return $true
            }
        }
    } catch {}
    return $false
}

function Is-TuiRunning {
    try {
        $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
        foreach ($p in $procs) {
            if (-not $p.CommandLine) { continue }
            if (
                ($p.CommandLine -match "hermes_cli\.main" -or $p.CommandLine -match "-m\s+hermes_cli") `
                -and $p.CommandLine -notmatch "gateway\s+run" `
                -and $p.CommandLine -notmatch "\s-q\s" `
                -and $p.CommandLine -notmatch "\s--query\s"
            ) {
                return $true
            }
        }
    } catch {}
    return $false
}

function Start-DetachedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$StdoutLog,
        [string]$StderrLog,
        [string]$WindowStyle = "Minimized"
    )
    Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $RepoRoot `
        -WindowStyle $WindowStyle `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog | Out-Null
    Write-Host "Started: $Name"
}

function Start-DetachedProcessNoRedirect {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WindowStyle = "Minimized"
    )
    Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $RepoRoot `
        -WindowStyle $WindowStyle | Out-Null
    Write-Host "Started: $Name"
}

function Split-Args {
    param([string]$Raw)
    if (-not $Raw) { return @() }
    return ($Raw -split "\s+" | Where-Object { $_ -and $_.Trim() })
}

$DelaySeconds = 20
if ($env:HERMES_STARTUP_DELAY_SECONDS -and $env:HERMES_STARTUP_DELAY_SECONDS.Trim()) {
    $parsedDelay = 0
    if ([int]::TryParse($env:HERMES_STARTUP_DELAY_SECONDS, [ref]$parsedDelay) -and $parsedDelay -ge 0) {
        $DelaySeconds = $parsedDelay
    }
}
if ($DelaySeconds -gt 0) {
    Start-Sleep -Seconds $DelaySeconds
}

# ---------------------------------------------------------------------------
# Hypura serve (GGUF) + OpenAI-compatible proxy (Hermes -> /v1/chat/completions)
# ---------------------------------------------------------------------------
$hypuraStarted = $false
# Keep local GGUF servers opt-in. They reserve VRAM, so normal Hermes restarts
# should not start them unless a rollback/recovery check explicitly asks for it.
$enableHypura = Get-BoolEnvOrDefault -Name "HERMES_AUTOSTART_HYPURA" -DefaultValue $false
$enableHypuraProxy = Get-BoolEnvOrDefault -Name "HERMES_AUTOSTART_HYPURA_PROXY" -DefaultValue $false

$hypuraExe = Get-EnvValue -Name "HERMES_HYPURA_EXE"
if (-not $hypuraExe) {
    $hypuraExe = "C:\Users\downl\Desktop\hypura-main\hypura-main\target_release_rtx\release\hypura.exe"
}
$hypuraGguf = Get-EnvValue -Name "HERMES_HYPURA_GGUF"
if (-not $hypuraGguf) {
    $hypuraGguf = "C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
}
$hypuraPort = 8080
if ($env:HERMES_HYPURA_PORT -and $env:HERMES_HYPURA_PORT.Trim() -match '^\d+$') {
    $hypuraPort = [int]$env:HERMES_HYPURA_PORT.Trim()
}
$hypuraProxyPort = 8090
if ($env:HERMES_HYPURA_PROXY_PORT -and $env:HERMES_HYPURA_PROXY_PORT.Trim() -match '^\d+$') {
    $hypuraProxyPort = [int]$env:HERMES_HYPURA_PROXY_PORT.Trim()
}

$hypuraLoadWait = 15
if ($env:HERMES_HYPURA_LOAD_WAIT_SECONDS -and $env:HERMES_HYPURA_LOAD_WAIT_SECONDS.Trim() -match '^\d+$') {
    $hypuraLoadWait = [int]$env:HERMES_HYPURA_LOAD_WAIT_SECONDS.Trim()
}

if ($enableHypura -and (Test-Path -LiteralPath $hypuraExe) -and (Test-Path -LiteralPath $hypuraGguf)) {
    if (-not (Is-CommandRunning -Needle "hypura.exe serve")) {
        $hypuraCmd = "set PYTHONIOENCODING=utf-8&& set PYTHONUTF8=1&& `"$hypuraExe`" serve `"$hypuraGguf`" --host 127.0.0.1 --port $hypuraPort"
        Start-Process `
            -FilePath "cmd.exe" `
            -ArgumentList @("/c", $hypuraCmd) `
            -WorkingDirectory $RepoRoot `
            -WindowStyle "Minimized" | Out-Null
        Write-Host "Started: hypura serve (port $hypuraPort)"
        if ($hypuraLoadWait -gt 0) {
            Start-Sleep -Seconds $hypuraLoadWait
        }
        $hypuraStarted = $true
    } else {
        Write-Host "Skip hypura serve: already running"
        $hypuraStarted = $true
    }
} elseif ($enableHypura) {
    Write-Warning "Hypura autostart skipped (set HERMES_HYPURA_EXE / HERMES_HYPURA_GGUF or install files). exe=$hypuraExe gguf=$hypuraGguf"
}

if ($enableHypuraProxy -and $hypuraStarted) {
    if (-not (Is-CommandRunning -Needle "hypura_oai_proxy")) {
        $proxyCmd = @(
            "set PYTHONIOENCODING=utf-8&& set PYTHONUTF8=1&&",
            "set HYPURA_OAI_UPSTREAM=http://127.0.0.1:$hypuraPort&&",
            "set HYPURA_OAI_PROXY_PORT=$hypuraProxyPort&&",
            "py -3 -m hypura_oai_proxy"
        ) -join " "
        Start-Process `
            -FilePath "cmd.exe" `
            -ArgumentList @("/c", $proxyCmd) `
            -WorkingDirectory $RepoRoot `
            -WindowStyle "Minimized" | Out-Null
        Write-Host "Started: hypura_oai_proxy (port $hypuraProxyPort -> upstream http://127.0.0.1:$hypuraPort)"
        Write-Host "Hermes LLM: model.base_url -> http://127.0.0.1:$hypuraProxyPort/v1 (provider custom, chat_completions)"
    } else {
        Write-Host "Skip hypura_oai_proxy: already running"
    }
}

# Avoid port clash: Hypura binds 8080 by default; move Hermes FastAPI wrapper to 8765 unless user set HERMES_API_PORT.
if ($hypuraStarted) {
    $curApi = [Environment]::GetEnvironmentVariable("HERMES_API_PORT", "Process")
    if (-not $curApi -or -not $curApi.Trim() -or $curApi.Trim() -eq "8080") {
        [Environment]::SetEnvironmentVariable("HERMES_API_PORT", "8765", "Process")
        $env:HERMES_API_PORT = "8765"
        Write-Host "HERMES_API_PORT=8765 (Hypura uses $hypuraPort)"
    }
}

$enableGateway = Get-BoolEnvOrDefault -Name "HERMES_AUTOSTART_GATEWAY" -DefaultValue $true
$enableApi = Get-BoolEnvOrDefault -Name "HERMES_AUTOSTART_API" -DefaultValue $true
$enableTui = Get-BoolEnvOrDefault -Name "HERMES_AUTOSTART_TUI" -DefaultValue $true
$enableBrowser = Get-BoolEnvOrDefault -Name "HERMES_AUTOSTART_BROWSER" -DefaultValue $true
$enableNgrok = Get-BoolEnvOrDefault -Name "HERMES_AUTOSTART_NGROK" -DefaultValue $true
$enableVoicevox = Get-BoolEnvOrDefault -Name "HERMES_AUTOSTART_VOICEVOX" -DefaultValue $true
$apiKey = [Environment]::GetEnvironmentVariable("API_SERVER_KEY", "Process")
if (-not $apiKey -or $apiKey.Trim().Length -lt 16) {
    if ($enableApi -or $enableNgrok) {
        Write-Warning "API_SERVER_KEY must be a non-placeholder secret of at least 16 characters; skipping API sidecar and ngrok."
    }
    $enableApi = $false
    $enableNgrok = $false
}

$gatewayRunning = (Is-CommandRunning -Needle "hermes_cli.main gateway run") -or `
    (Is-CommandRunning -Needle "hermes_cli gateway run")
if ($enableGateway -and -not $gatewayRunning) {
    Start-DetachedProcessNoRedirect `
        -Name "gateway" `
        -FilePath "cmd.exe" `
        -ArgumentList @("/c", "set PYTHONIOENCODING=utf-8&& set PYTHONUTF8=1&& py -3 -m hermes_cli gateway run --replace")
}

if ($enableApi -and -not (Is-CommandRunning -Needle "hermes_api_server")) {
    Start-DetachedProcess `
        -Name "fastapi" `
        -FilePath "cmd.exe" `
        -ArgumentList @("/c", "set PYTHONIOENCODING=utf-8&& set PYTHONUTF8=1&& py -3 -m hermes_api_server") `
        -StdoutLog (Join-Path $LogDir "api-autostart.log") `
        -StderrLog (Join-Path $LogDir "api-autostart-error.log")
}

$ngrokTunnelPort = if ($env:HERMES_API_PORT -and $env:HERMES_API_PORT.Trim()) { $env:HERMES_API_PORT.Trim() } else { "8080" }

if ($enableNgrok -and -not (Is-CommandRunning -Needle "ngrok http")) {
    $ngrokExe = if ($env:NGROK_EXE -and $env:NGROK_EXE.Trim()) { $env:NGROK_EXE.Trim() } else { "ngrok" }
    try {
        Start-DetachedProcess `
            -Name "ngrok" `
            -FilePath $ngrokExe `
            -ArgumentList @("http", $ngrokTunnelPort) `
            -StdoutLog (Join-Path $LogDir "ngrok-autostart.log") `
            -StderrLog (Join-Path $LogDir "ngrok-autostart-error.log")
    } catch {
        Write-Warning "Failed to start ngrok: $($_.Exception.Message)"
    }
}

if ($enableVoicevox -and -not (Is-CommandRunning -Needle "VOICEVOX")) {
    $voicevoxPathFromEnv = Get-EnvValue -Name "VOICEVOX_CLI_PATH"
    $voicevoxArgsFromEnv = Get-EnvValue -Name "VOICEVOX_CLI_ARGS"
    $voicevoxExe = if ($voicevoxPathFromEnv) { $voicevoxPathFromEnv } else { "voicevox" }
    $voicevoxArgsRaw = if ($voicevoxArgsFromEnv) {
        $voicevoxArgsFromEnv
    } else {
        "--host 127.0.0.1 --port 50021"
    }
    $voicevoxArgs = Split-Args -Raw $voicevoxArgsRaw
    try {
        Start-DetachedProcess `
            -Name "voicevox" `
            -FilePath $voicevoxExe `
            -ArgumentList $voicevoxArgs `
            -StdoutLog (Join-Path $LogDir "voicevox-autostart.log") `
            -StderrLog (Join-Path $LogDir "voicevox-autostart-error.log")
    } catch {
        Write-Warning "Failed to start VOICEVOX CLI: $($_.Exception.Message)"
    }
}

if ($enableTui -and -not (Is-TuiRunning)) {
    # TUI is interactive; keep a dedicated console window.
    $tuiCmd = "Set-Location -LiteralPath '$RepoRoot'; `$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; py -3 -m hermes_cli"
    $tuiProc = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-NoProfile", "-Command", $tuiCmd) `
        -WorkingDirectory $RepoRoot `
        -WindowStyle "Normal" `
        -PassThru
    Write-Host "Started: tui (PID: $($tuiProc.Id))"
}

if ($enableBrowser) {
    $browserApiPort = if ($env:HERMES_API_PORT -and $env:HERMES_API_PORT.Trim()) { $env:HERMES_API_PORT.Trim() } else { "8080" }
    $browserUrl = if ($env:HERMES_STARTUP_BROWSER_URL -and $env:HERMES_STARTUP_BROWSER_URL.Trim()) {
        $env:HERMES_STARTUP_BROWSER_URL.Trim()
    } else {
        "http://127.0.0.1:$browserApiPort/docs"
    }
    Start-Process $browserUrl | Out-Null
    Write-Host "Started: browser ($browserUrl)"
}
