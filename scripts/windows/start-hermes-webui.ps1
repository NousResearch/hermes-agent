param(
    [string]$WebUiRoot = "",
    [string]$AgentRoot = "",
    [int]$Port = 8787,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
. (Join-Path $ScriptDir "Resolve-CanonicalHermesHome.ps1")

function Get-HermesHome {
    return (Resolve-CanonicalHermesHome -RepoRoot $RepoRoot)
}

function Read-DotEnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }

        $parts = $trimmed -split "=", 2
        $key = ($parts[0].Trim().TrimStart([char]0xFEFF) -replace "^export\s+", "")
        if ($key -ne $Name) {
            continue
        }

        $value = $parts[1].Trim()
        if ($value -match '^"(.*)"$') {
            return $Matches[1]
        }
        if ($value -match "^'(.*)'$") {
            return $Matches[1]
        }
        return $value
    }

    return $null
}

function Resolve-WebUiPassword {
    param(
        [string]$HermesHome,
        [string]$WebUiEnvPath
    )

    if ($env:HERMES_WEBUI_PASSWORD -and $env:HERMES_WEBUI_PASSWORD.Trim()) {
        return @{ Password = $env:HERMES_WEBUI_PASSWORD; Source = "process environment" }
    }

    if ($env:HERMES_WEBUI_PASSWORD_FILE -and $env:HERMES_WEBUI_PASSWORD_FILE.Trim()) {
        $passwordFile = $env:HERMES_WEBUI_PASSWORD_FILE.Trim()
        if (Test-Path -LiteralPath $passwordFile) {
            $password = (Get-Content -LiteralPath $passwordFile -Encoding UTF8 -TotalCount 1)
            if ($password -and $password.Trim()) {
                return @{ Password = $password.Trim(); Source = "HERMES_WEBUI_PASSWORD_FILE" }
            }
        }
    }

    $hermesEnv = Join-Path $HermesHome ".env"
    $passwordFromHermesEnv = Read-DotEnvValue -Path $hermesEnv -Name "HERMES_WEBUI_PASSWORD"
    if ($passwordFromHermesEnv -and $passwordFromHermesEnv.Trim()) {
        return @{ Password = $passwordFromHermesEnv.Trim(); Source = "$HermesHome\.env" }
    }

    $legacyPassword = Read-DotEnvValue -Path $WebUiEnvPath -Name "HERMES_WEBUI_PASSWORD"
    if ($legacyPassword -and $legacyPassword.Trim()) {
        return @{ Password = $legacyPassword.Trim(); Source = "legacy WebUI .env" }
    }

    return $null
}

function Start-WebUiBrowserOpener {
    param(
        [string]$Url
    )

    $encodedUrl = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($Url))
    $script = @"
`$url = [System.Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('$encodedUrl'))
for (`$i = 0; `$i -lt 60; `$i++) {
    try {
        `$response = Invoke-WebRequest -UseBasicParsing -Uri `$url -TimeoutSec 2
        if (`$response.StatusCode -ge 200 -and `$response.StatusCode -lt 500) {
            Start-Process `$url
            exit 0
        }
    } catch {}
    Start-Sleep -Seconds 1
}
Start-Process `$url
"@
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-WindowStyle",
        "Hidden",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $script
    ) -WindowStyle Hidden | Out-Null
}

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $AgentRoot) {
    $AgentRoot = $env:HERMES_WEBUI_AGENT_DIR
    if (-not $AgentRoot) { $AgentRoot = $RepoRoot }
}
if (-not $WebUiRoot) {
    $WebUiRoot = $env:HERMES_WEBUI_ROOT
    if (-not $WebUiRoot) {
        $WebUiRoot = Join-Path $env:USERPROFILE "Desktop\hermes-webui"
    }
}

$ExampleEnv = Join-Path $AgentRoot "config\hermes-webui.env.example"
$TargetEnv = Join-Path $WebUiRoot ".env"
$HermesHome = Get-HermesHome

if (-not (Test-Path -LiteralPath (Join-Path $WebUiRoot "bootstrap.py"))) {
    Write-Error "Hermes WebUI not found at $WebUiRoot"
}

if (-not (Test-Path -LiteralPath (Join-Path $AgentRoot "run_agent.py"))) {
    Write-Error "Hermes agent checkout not found at $AgentRoot"
}

if (-not (Test-Path -LiteralPath $TargetEnv)) {
    if (-not (Test-Path -LiteralPath $ExampleEnv)) {
        Write-Error "Missing template: $ExampleEnv"
    }
    Copy-Item -LiteralPath $ExampleEnv -Destination $TargetEnv
    Write-Host "Created $TargetEnv from upstream-sync template."
}

$env:HERMES_WEBUI_AGENT_DIR = $AgentRoot
$env:HERMES_WEBUI_PORT = "$Port"
$env:HERMES_HOME = $HermesHome
$resolvedPassword = Resolve-WebUiPassword -HermesHome $HermesHome -WebUiEnvPath $TargetEnv
if ($resolvedPassword) {
    $env:HERMES_WEBUI_PASSWORD = $resolvedPassword.Password
    $env:HERMES_WEBUI_PRESERVE_ENV = "1"
    Write-Host "Injected HERMES_WEBUI_PASSWORD from $($resolvedPassword.Source)."
}

$StartScript = Join-Path $WebUiRoot "start.ps1"
if (-not (Test-Path -LiteralPath $StartScript)) {
    Write-Error "Missing start.ps1 in $WebUiRoot"
}

$url = "http://127.0.0.1:$Port/"
if ($Open -or ($env:HERMES_WEBUI_OPEN_ON_START -and (($env:HERMES_WEBUI_OPEN_ON_START).Trim().ToLowerInvariant() -in @("1", "true", "yes", "on")))) {
    Start-WebUiBrowserOpener -Url $url
}

Write-Host "Starting Hermes WebUI on $url (agent: $AgentRoot)"
& $StartScript @args
