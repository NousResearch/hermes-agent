# Create Hermes Agent desktop shortcuts (CLI, Gateway, Harness, Grok OAuth, config folder).
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/create-hermes-desktop-shortcuts.ps1
#   powershell ... -File scripts/create-hermes-desktop-shortcuts.ps1 -IncludePublicDesktop
#
# Hermes console shortcuts run via cmd.exe so a SyntaxError or missing venv leaves
# the window open (``|| pause``) instead of flashing closed — which users often
# describe as the shortcut "disappearing".

[CmdletBinding()]
param(
    [switch]$IncludePublicDesktop,
    [switch]$RecreateSo8tLlamaShortcut
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-HermesInvoke {
    param([string]$RepoRoot)

    $venvHermes = Join-Path $RepoRoot ".venv\Scripts\hermes.exe"
    if (Test-Path -LiteralPath $venvHermes) {
        return @{
            HermesPath = $venvHermes
            PrefixArgs = ""
            Source     = "venv-hermes.exe"
            IconPath   = $venvHermes
        }
    }

    $cmdHermes = Get-Command hermes -ErrorAction SilentlyContinue
    if ($cmdHermes -and $cmdHermes.Source) {
        return @{
            HermesPath = $cmdHermes.Source
            PrefixArgs = ""
            Source     = "PATH-hermes"
            IconPath   = $cmdHermes.Source
        }
    }

    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return @{
            HermesPath = $venvPython
            PrefixArgs = "-m hermes_cli.main"
            Source     = "venv-python-module"
            IconPath   = $venvPython
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return @{
            HermesPath = $pyLauncher.Source
            PrefixArgs = "-3 -m hermes_cli.main"
            Source     = "py-launcher-module"
            IconPath   = $pyLauncher.Source
        }
    }

    throw "Could not resolve hermes launcher (install .venv or add hermes to PATH)."
}

function Build-HermesCommandLine {
    param(
        [hashtable]$Invoke,
        [string]$SubArgs
    )

    $exe = $Invoke.HermesPath
    $prefix = $Invoke.PrefixArgs.Trim()
    if ([string]::IsNullOrWhiteSpace($SubArgs)) {
        $inner = if ($prefix) { "`"$exe`" $prefix" } else { "`"$exe`"" }
    }
    else {
        $inner = if ($prefix) { "`"$exe`" $prefix $SubArgs" } else { "`"$exe`" $SubArgs" }
    }
    # On failure, pause so the user can read tracebacks (instant-close looked like a vanished shortcut).
    return "$inner || echo. & echo [Hermes exited with error — press any key] & pause >nul"
}

function New-HermesShortcut {
    param(
        [string]$LinkPath,
        [string]$TargetPath,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$Description,
        [string]$IconLocation = "$env:SystemRoot\System32\cmd.exe,0",
        [int]$WindowStyle = 1
    )

    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($LinkPath)
    $Shortcut.TargetPath = $TargetPath
    $Shortcut.Arguments = $Arguments
    $Shortcut.WorkingDirectory = $WorkingDirectory
    $Shortcut.Description = $Description
    $Shortcut.WindowStyle = $WindowStyle
    $Shortcut.IconLocation = $IconLocation
    $Shortcut.Save()

    return [PSCustomObject]@{
        Path               = $LinkPath
        TargetPath         = $TargetPath
        Arguments          = $Arguments
        WorkingDirectory   = $WorkingDirectory
        Description        = $Description
        IconLocation       = $IconLocation
    }
}

function New-HermesConsoleShortcut {
    param(
        [string]$LinkPath,
        [string]$RepoRoot,
        [hashtable]$Invoke,
        [string]$SubArgs,
        [string]$Description
    )

    $cmdLine = Build-HermesCommandLine -Invoke $Invoke -SubArgs $SubArgs
    $cmdArgs = "/k cd /d `"$RepoRoot`" && $cmdLine"
    $icon = if ($Invoke.IconPath -and (Test-Path -LiteralPath $Invoke.IconPath)) {
        "$($Invoke.IconPath),0"
    } else {
        "$env:SystemRoot\System32\cmd.exe,0"
    }

    return New-HermesShortcut `
        -LinkPath $LinkPath `
        -TargetPath "$env:SystemRoot\System32\cmd.exe" `
        -Arguments $cmdArgs `
        -WorkingDirectory $RepoRoot `
        -Description $Description `
        -IconLocation $icon
}

function Ensure-So8tLlamaShortcut {
    param(
        [string]$Desktop,
        [switch]$Force
    )

    $name = "SuperGemma4 llama-server (RTX3060).lnk"
    $lnkPath = Join-Path $Desktop $name
    $so8tScript = "C:\Users\downl\Desktop\SO8T\scripts\start-supergemma-server.ps1"
    $so8tRoot = "C:\Users\downl\Desktop\SO8T"

    if ((Test-Path -LiteralPath $lnkPath) -and -not $Force) {
        return [PSCustomObject]@{
            Path       = $lnkPath
            Status     = "exists"
            TargetPath = (New-Object -ComObject WScript.Shell).CreateShortcut($lnkPath).TargetPath
        }
    }

    if (-not (Test-Path -LiteralPath $so8tScript)) {
        return [PSCustomObject]@{
            Path   = $lnkPath
            Status = "skipped-missing-script"
            Error  = "Not found: $so8tScript"
        }
    }

    $created = New-HermesShortcut `
        -LinkPath $lnkPath `
        -TargetPath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -Arguments "-NoExit -ExecutionPolicy Bypass -File `"$so8tScript`"" `
        -WorkingDirectory $so8tRoot `
        -Description "Start SuperGemma4 llama-server on RTX3060 (SO8T)" `
        -IconLocation "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe,0"

    return [PSCustomObject]@{
        Path       = $created.Path
        Status     = "created"
        TargetPath = $created.TargetPath
        Arguments  = $created.Arguments
    }
}

$RepoRoot = Get-RepoRoot
$invoke = Resolve-HermesInvoke -RepoRoot $RepoRoot

$desktops = @([Environment]::GetFolderPath("Desktop"))
if ($IncludePublicDesktop) {
    $desktops += [Environment]::GetFolderPath("CommonDesktopDirectory")
}

$definitions = @(
    @{
        Name        = "Hermes Agent CLI.lnk"
        SubArgs     = ""
        Description = "Hermes Agent interactive CLI (hermes)"
    },
    @{
        Name        = "Hermes Gateway.lnk"
        SubArgs     = "gateway"
        Description = "Hermes messaging gateway (foreground)"
    },
    @{
        Name        = "Hermes Harness.lnk"
        SubArgs     = "harness start"
        Description = "Start Hypura / OpenClaw harness daemon"
    },
    @{
        Name        = "Hermes Grok OAuth.lnk"
        SubArgs     = "auth add xai-oauth"
        Description = "Browser login for xAI Grok OAuth (SuperGrok subscription)"
    }
)

$hermesHome = Join-Path $env:USERPROFILE ".hermes"
if (-not (Test-Path -LiteralPath $hermesHome)) {
    New-Item -ItemType Directory -Path $hermesHome -Force | Out-Null
}

$configDef = @{
    Name        = "Hermes Config (.hermes).lnk"
    TargetPath  = "$env:SystemRoot\explorer.exe"
    Arguments   = $hermesHome
    Description = "Open Hermes config folder (~/.hermes)"
}

$results = @()

foreach ($desktop in $desktops | Select-Object -Unique) {
    if (-not (Test-Path -LiteralPath $desktop)) {
        $results += [PSCustomObject]@{ Path = $desktop; Status = "error"; Error = "Desktop folder missing" }
        continue
    }

    foreach ($def in $definitions) {
        $lnk = Join-Path $desktop $def.Name
        try {
            $row = New-HermesConsoleShortcut `
                -LinkPath $lnk `
                -RepoRoot $RepoRoot `
                -Invoke $invoke `
                -SubArgs $def.SubArgs `
                -Description $def.Description
            $results += [PSCustomObject]@{
                Path             = $row.Path
                Status           = "created"
                TargetPath       = $row.TargetPath
                Arguments        = $row.Arguments
                WorkingDirectory = $row.WorkingDirectory
                LauncherSource   = $invoke.Source
            }
        }
        catch {
            $results += [PSCustomObject]@{
                Path   = $lnk
                Status = "error"
                Error  = $_.Exception.Message
            }
        }
    }

    if ($desktop -eq [Environment]::GetFolderPath("Desktop")) {
        $cfgLnk = Join-Path $desktop $configDef.Name
        try {
            $row = New-HermesShortcut `
                -LinkPath $cfgLnk `
                -TargetPath $configDef.TargetPath `
                -Arguments $configDef.Arguments `
                -WorkingDirectory $env:USERPROFILE `
                -Description $configDef.Description `
                -IconLocation "$env:SystemRoot\explorer.exe,0"
            $results += [PSCustomObject]@{
                Path             = $row.Path
                Status           = "created"
                TargetPath       = $row.TargetPath
                Arguments        = $row.Arguments
                WorkingDirectory = $row.WorkingDirectory
                LauncherSource   = "explorer"
            }
        }
        catch {
            $results += [PSCustomObject]@{
                Path   = $cfgLnk
                Status = "error"
                Error  = $_.Exception.Message
            }
        }

        $so8t = Ensure-So8tLlamaShortcut -Desktop $desktop -Force:$RecreateSo8tLlamaShortcut
        $results += $so8t
    }
}

Write-Host ""
Write-Host "Hermes launcher: $($invoke.Source) -> $($invoke.HermesPath)" -ForegroundColor Cyan
Write-Host "Repo root: $RepoRoot"
Write-Host "Console shortcuts use cmd.exe /k with pause-on-error." -ForegroundColor DarkGray
Write-Host ""
$results | Format-Table -AutoSize Path, Status, TargetPath, Arguments

$errors = $results | Where-Object { $_.Status -eq "error" -or $_.Status -like "skipped*" }
if ($errors) {
    Write-Warning "Some shortcuts were not created; see table above."
    exit 1
}

exit 0
