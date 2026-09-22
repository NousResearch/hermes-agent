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

    $arguments = @(
        (Quote-Argument $bundle),
        "--node-id $(Quote-Argument $NodeId)",
        "--coordinator $(Quote-Argument $Coordinator)",
        "--hermes-executable $(Quote-Argument (Join-Path $HermesRoot 'bin\\hermes.exe'))"
    )
    foreach ($profileName in $Profile) {
        $arguments += "--profile $(Quote-Argument $profileName)"
    }
    foreach ($projectName in $Project) {
        $arguments += "--project $(Quote-Argument $projectName)"
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
