param(
    [Parameter(Mandatory = $true)]
    [string]$DesktopExecutable,
    [Parameter(Mandatory = $true)]
    [string]$HermesRoot,
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,
    [Parameter(Mandatory = $true)]
    [string]$Coordinator,
    [string]$NodeId = "windows",
    [string[]]$Profile = @("coding-expert"),
    [string[]]$Project = @("Hermes Agent", "LunaBot"),
    [int]$PollSeconds = 2
)

$ErrorActionPreference = "Stop"
$fleetRoot = Join-Path $HermesRoot "fleet"
$bundle = Join-Path $fleetRoot "hermes-fleet-runner.pyz"
$marker = Join-Path $fleetRoot "desktop-live"
$tokenFile = Join-Path $HermesRoot ".env"
$stdoutLog = Join-Path $fleetRoot "runner.supervisor.out.log"
$stderrLog = Join-Path $fleetRoot "runner.supervisor.err.log"
$desktopName = [IO.Path]::GetFileName($DesktopExecutable)
$desktopCommandLine = '"' + $DesktopExecutable + '"'
$script:RunnerProcess = $null

function Get-DesktopMainProcess {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq $desktopName -and
            $_.ExecutablePath -eq $DesktopExecutable -and
            $_.CommandLine -eq $desktopCommandLine
        } |
        Select-Object -First 1
}

function Get-RunnerProcess {
    if ($null -eq $script:RunnerProcess) {
        return $null
    }
    Get-Process -Id $script:RunnerProcess.Id -ErrorAction SilentlyContinue
}

function Remove-LivenessMarker {
    if (Test-Path -LiteralPath $marker) {
        Remove-Item -LiteralPath $marker -Force
    }
}

function Stop-Runner {
    $runner = Get-RunnerProcess
    if ($null -ne $runner) {
        Stop-Process -Id $runner.Id -Force -ErrorAction SilentlyContinue
    }
    $script:RunnerProcess = $null
    Remove-LivenessMarker
}

function Quote-Argument([string]$value) {
    '"' + $value.Replace('"', '\"') + '"'
}

function Get-ProfileModelConfig([string]$ProfileName) {
    $profileConfig = Join-Path $HermesRoot "config.yaml"
    if ($ProfileName -ne "default") {
        $profileConfig = Join-Path (Join-Path $HermesRoot "profiles") "$ProfileName\config.yaml"
    }
    if (-not (Test-Path -LiteralPath $profileConfig)) {
        return @("", "")
    }

    $inModel = $false
    $model = ""
    $provider = ""
    foreach ($line in Get-Content -LiteralPath $profileConfig) {
        if ($line -match "^model:\s*$") {
            $inModel = $true
            continue
        }
        if ($inModel -and $line -match "^[^\s]") {
            break
        }
        if ($inModel -and $line -match "^\s+default:\s*(.+?)\s*$") {
            $model = $Matches[1].Trim().Trim("'").Trim('"')
        }
        if ($inModel -and $line -match "^\s+provider:\s*(.+?)\s*$") {
            $provider = $Matches[1].Trim().Trim("'").Trim('"')
        }
    }
    return @($model, $provider)
}

function Get-LocalModelIds {
    $modelsDir = Join-Path $HermesRoot "models"
    if (-not (Test-Path -LiteralPath $modelsDir)) {
        return @()
    }

    @(
        Get-ChildItem -LiteralPath $modelsDir -File -Filter "*.gguf" -ErrorAction SilentlyContinue |
            ForEach-Object { $_.BaseName -replace '-\d{5}-of-\d{5}$', '' }
    )
}

function Resolve-ProfileModel([string]$Model, [string]$Provider, [string[]]$LocalModelIds) {
    $localProviders = @("local", "llamacpp", "llama.cpp", "llama-cpp")
    if (-not $Model -or $localProviders -notcontains $Provider.ToLowerInvariant() -or $LocalModelIds.Count -eq 0) {
        return $Model
    }

    $exact = @($LocalModelIds | Where-Object { $_ -ieq $Model })
    if ($exact.Count -gt 0) {
        return $exact[0]
    }

    # A profile can be shared between machines with different GGUF quantizations. Match the
    # model family before the quantization suffix so Q4 on the Mac can resolve to Q5 on Windows.
    $family = ($Model -replace '[-.]Q\d.*$', '').ToLowerInvariant()
    $variant = @(
        $LocalModelIds |
            Where-Object { $_.ToLowerInvariant().StartsWith($family) } |
            Sort-Object
    )
    if ($variant.Count -gt 0) {
        Write-Host "Fleet local model remap: $Model -> $($variant[0])"
        return $variant[0]
    }
    return $Model
}

function Start-Runner {
    if (-not (Test-Path -LiteralPath $bundle)) {
        throw "Fleet runner bundle does not exist: $bundle"
    }
    if (-not (Test-Path -LiteralPath $PythonExecutable)) {
        throw "Python executable does not exist: $PythonExecutable"
    }
    $prefix = "HERMES_FLEET_TOKEN="
    $tokenLines = @(Get-Content -LiteralPath $tokenFile | Where-Object { $_.StartsWith($prefix) })
    if ($tokenLines.Count -ne 1) {
        throw "Expected exactly one HERMES_FLEET_TOKEN entry in $tokenFile"
    }
    $env:HERMES_FLEET_TOKEN = $tokenLines[0].Substring($prefix.Length)
    New-Item -ItemType File -Force -Path $marker | Out-Null

    $profileNames = @("default") + $Profile
    $profileRoot = Join-Path $HermesRoot "profiles"
    if (Test-Path -LiteralPath $profileRoot) {
        $profileNames += @(Get-ChildItem -LiteralPath $profileRoot -Directory | Select-Object -ExpandProperty Name)
    }
    $profileNames = @($profileNames | Select-Object -Unique)
    $localModelIds = @(Get-LocalModelIds)
    $profileModelArgs = @()
    $profileProviderArgs = @()
    $hermesExecutable = Join-Path $HermesRoot "bin\\hermes.exe"
    foreach ($profileName in $profileNames) {
        $modelInfo = @(Get-ProfileModelConfig $profileName)
        if ($modelInfo.Count -ge 2 -and $modelInfo[0] -and $modelInfo[1]) {
            $resolvedModel = Resolve-ProfileModel $modelInfo[0] $modelInfo[1] $localModelIds
            $profileModelArgs += @("--profile-model", "$profileName=$resolvedModel")
            $profileProviderArgs += @("--profile-provider", "$profileName=$($modelInfo[1])")
        }
    }

    $arguments = @(
        (Quote-Argument $bundle),
        "--node-id $(Quote-Argument $NodeId)",
        "--coordinator $(Quote-Argument $Coordinator)",
        "--hermes-executable $(Quote-Argument $hermesExecutable)"
    )
    foreach ($profileName in $profileNames) {
        $arguments += "--profile $(Quote-Argument $profileName)"
    }
    foreach ($projectName in $Project) {
        $arguments += "--project $(Quote-Argument $projectName)"
    }
    foreach ($localModelId in $localModelIds) {
        $arguments += "--model $(Quote-Argument $localModelId)"
    }
    foreach ($profileModelArg in $profileModelArgs) {
        $arguments += (Quote-Argument $profileModelArg)
    }
    foreach ($profileProviderArg in $profileProviderArgs) {
        $arguments += (Quote-Argument $profileProviderArg)
    }
    $arguments += "--liveness-file $(Quote-Argument $marker)"
    $arguments += "--interval 2"
    $script:RunnerProcess = Start-Process -FilePath $PythonExecutable `
        -ArgumentList ($arguments -join " ") `
        -WorkingDirectory $HermesRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru
}

try {
    while ($true) {
        $desktop = Get-DesktopMainProcess
        $runner = Get-RunnerProcess
        if ($null -ne $desktop) {
            if ($null -eq $runner) {
                Start-Runner
            }
        } elseif ($null -ne $runner -or (Test-Path -LiteralPath $marker)) {
            Stop-Runner
        }
        Start-Sleep -Seconds ([Math]::Max(1, $PollSeconds))
    }
} finally {
    Stop-Runner
}
