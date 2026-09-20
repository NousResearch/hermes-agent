[CmdletBinding()]
param(
    [string]$ComposeFile = "docker-compose.yml",
    [string]$ApiServerKey,
    [string]$DashboardStatusUrl = "http://127.0.0.1:9119/api/status",
    [string]$GatewayModelsUrl = "http://127.0.0.1:8789/v1/models"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposePath = Join-Path $RepoRoot $ComposeFile
$EnvPath = Join-Path $RepoRoot "data/.env"

function Write-Step {
    param([string]$Message)
    Write-Host "→ $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path $Path)) {
        return $null
    }

    foreach ($line in Get-Content -Path $Path) {
        if ($line -match '^\s*#' -or $line -notmatch '=') {
            continue
        }

        if ($line -match "^\s*$([regex]::Escape($Name))\s*=\s*(.*)\s*$") {
            return $Matches[1].Trim().Trim("'").Trim('"')
        }
    }

    return $null
}

function Invoke-HermesGet {
    param(
        [string]$Uri,
        [string]$BearerToken,
        [int]$Attempts = 1,
        [int]$DelaySeconds = 1
    )

    $lastError = $null

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        $handler = [System.Net.Http.HttpClientHandler]::new()
        $client = [System.Net.Http.HttpClient]::new($handler)

        try {
            if ($BearerToken) {
                $client.DefaultRequestHeaders.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new("Bearer", $BearerToken)
            }

            $response = $client.GetAsync($Uri).GetAwaiter().GetResult()
            $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()

            if ([int]$response.StatusCode -ge 500 -and $attempt -lt $Attempts) {
                Start-Sleep -Seconds $DelaySeconds
                continue
            }

            return [pscustomobject]@{
                StatusCode = [int]$response.StatusCode
                Body = $body
            }
        }
        catch {
            $lastError = $_
            if ($attempt -lt $Attempts) {
                Start-Sleep -Seconds $DelaySeconds
                continue
            }

            throw
        }
        finally {
            $client.Dispose()
            $handler.Dispose()
        }
    }

    throw $lastError
}

if (-not (Test-Path $ComposePath)) {
    throw "Compose file not found: $ComposePath"
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker was not found on PATH."
}

# PowerShell 7.6 on Windows can report Invoke-WebRequest ResponseEnded against
# Hermes' aiohttp API even when the endpoint is healthy. Use HttpClient probes here.
if (-not $ApiServerKey) {
    $ApiServerKey = Get-EnvValue -Path $EnvPath -Name "API_SERVER_KEY"
}

if (-not $ApiServerKey) {
    Write-Warning "API_SERVER_KEY is not set in $EnvPath. API health probe will be skipped."
    Write-Warning "Set API_SERVER_KEY in data/.env to enable the :8789 health probe."
}

Push-Location $RepoRoot
try {
    Write-Step "Pulling the latest upstream Hermes image via $ComposeFile"
    docker compose -f $ComposePath pull
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose pull failed."
    }

    Write-Step "Recreating hermes-web and hermes-gateway together"
    docker compose -f $ComposePath up -d --force-recreate hermes-web hermes-gateway
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed."
    }

    Write-Step "Checking compose service status"
    docker compose -f $ComposePath ps
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose ps failed."
    }

    Write-Step "Checking image digest alignment"
    $inspectLines = @(docker inspect --format '{{.Name}} {{.Image}}' hermes-web hermes-gateway)
    if ($LASTEXITCODE -ne 0) {
        throw "docker inspect failed."
    }

    $uniqueDigests = @(
        $inspectLines |
            ForEach-Object { ($_ -split ' ', 2)[1] } |
            Select-Object -Unique
    )

    if ($uniqueDigests.Count -ne 1) {
        throw "Hermes image drift detected: hermes-web and hermes-gateway are on different digests.`n$($inspectLines -join "`n")"
    }

    Write-Success "Both Hermes services are on $($uniqueDigests[0])"

    Write-Step "Verifying dashboard health"
    $dashboardResponse = Invoke-HermesGet -Uri $DashboardStatusUrl -Attempts 15
    if ($dashboardResponse.StatusCode -ne 200) {
        throw "Dashboard probe failed with status $($dashboardResponse.StatusCode). Body: $($dashboardResponse.Body)"
    }

    Write-Success "Dashboard endpoint returned HTTP 200"

    Write-Step "Verifying OpenSpace API health"
    $gatewayResponse = Invoke-HermesGet -Uri $GatewayModelsUrl -BearerToken $ApiServerKey -Attempts 30
    if ($gatewayResponse.StatusCode -ne 200) {
        throw "Gateway probe failed with status $($gatewayResponse.StatusCode). Body: $($gatewayResponse.Body)"
    }

    Write-Success "Gateway endpoint returned HTTP 200"
    Write-Host $gatewayResponse.Body
}
finally {
    Pop-Location
}