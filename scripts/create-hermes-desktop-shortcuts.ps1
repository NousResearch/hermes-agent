# Create / refresh Hermes Agent desktop shortcuts (single folder on Desktop).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/create-hermes-desktop-shortcuts.ps1
#   powershell ... -File scripts/create-hermes-desktop-shortcuts.ps1 -IncludePublicDesktop
#   powershell ... -File scripts/create-hermes-desktop-shortcuts.ps1 -CreateVenv
#   powershell ... -File scripts/create-hermes-desktop-shortcuts.ps1 -DesktopRoot
#
# All Hermes shortcuts land in:  %USERPROFILE%\Desktop\Hermes Agent\
# Optional -DesktopRoot: one explorer shortcut on the Desktop root that opens that folder.
# Legacy .lnk files on the Desktop root (and optional public desktop) are removed.
# Legacy OpenClaw/Hakua root shortcuts are moved under Hermes Agent\Legacy OpenClaw.
#
# Console shortcuts use cmd.exe /k with pause-on-error so tracebacks stay visible.

[CmdletBinding()]
param(
    [string]$ShortcutFolderName = "Hermes Agent",
    [switch]$IncludePublicDesktop,
    [switch]$CreateVenv,
    [switch]$IncludeLegacySo8tLlamaShortcut,
    [switch]$KeepLegacyDesktopRootShortcuts,
    [switch]$KeepLegacyOpenClawRootShortcuts,
    [switch]$DesktopRoot
)

$ErrorActionPreference = "Stop"

# Basenames we own (root + subfolder cleanup).
$HermesShortcutNames = @(
    "Hermes.lnk",
    "Hermes Agent CLI.lnk",
    "Hermes Desktop.lnk",
    "Hermes Gateway.lnk",
    "Hermes Llama Fallback (RTX3060).lnk",
    "Hermes Llama Fallback (RTX3080).lnk",
    "Hermes llama-server RTX3080.lnk",
    "Hermes Autostart (register).lnk",
    "Hermes Autostart (unregister).lnk",
    "Hermes Harness.lnk",
    "Hermes Grok OAuth.lnk",
    "Hermes Doctor.lnk",
    "Hermes Config (.hermes).lnk",
    "Hermes Stack.lnk",
    "Hermes Hypura Stack.lnk",
    "SuperGemma4 llama-server (RTX3060).lnk"
)

# Legacy OpenClaw launchers are still useful, but should not live on the
# Desktop root once Hermes owns the top-level shortcut surface.
$LegacyOpenClawShortcutNames = @(
    "OpenClaw.lnk",
    "Hakua-Sovereign-Manifestation.lnk"
)

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Ensure-ProjectVenv {
    param([string]$RepoRoot)

    $venvHermes = Join-Path $RepoRoot ".venv\Scripts\hermes.exe"
    if (Test-Path -LiteralPath $venvHermes) {
        return [PSCustomObject]@{ Status = "ok"; Message = "Found $venvHermes" }
    }

    if (-not $CreateVenv) {
        return [PSCustomObject]@{
            Status  = "skipped"
            Message = ".venv missing; pass -CreateVenv to run 'uv venv' + 'uv sync' in repo root"
        }
    }

    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "CreateVenv requested but 'uv' is not on PATH. Install uv or create .venv manually."
    }

    Push-Location $RepoRoot
    try {
        & uv venv
        if ($LASTEXITCODE -ne 0) { throw "uv venv failed with exit $LASTEXITCODE" }
        if (Test-Path -LiteralPath "pyproject.toml") {
            & uv sync
            if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit $LASTEXITCODE" }
        }
        elseif (Test-Path -LiteralPath "requirements.txt") {
            & uv pip install -r requirements.txt
            if ($LASTEXITCODE -ne 0) { throw "uv pip install failed with exit $LASTEXITCODE" }
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $venvHermes)) {
        throw "venv created but hermes.exe still missing at $venvHermes"
    }

    return [PSCustomObject]@{ Status = "created"; Message = "Created venv and synced dependencies" }
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

    throw "Could not resolve hermes launcher. Run with -CreateVenv or install .venv / hermes on PATH."
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
    return "$inner || (echo. & echo [Hermes exited with error - press any key] & pause >nul)"
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

    $parent = Split-Path -Parent $LinkPath
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

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
    }
    else {
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

function New-HermesPowerShellScriptShortcut {
    param(
        [string]$LinkPath,
        [string]$RepoRoot,
        [string]$StartScript,
        [string]$Description,
        [switch]$NoExit
    )

    $exitFlag = if ($NoExit) { "-NoExit " } else { "" }
    return New-HermesShortcut `
        -LinkPath $LinkPath `
        -TargetPath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -Arguments "${exitFlag}-NoProfile -ExecutionPolicy Bypass -File `"$StartScript`"" `
        -WorkingDirectory $RepoRoot `
        -Description $Description `
        -IconLocation "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe,0"
}

function New-HermesDesktopShortcut {
    param(
        [string]$LinkPath,
        [string]$RepoRoot,
        [string]$HermesHome,
        [string]$StartScript
    )

    $hermesIcon = Join-Path $RepoRoot ".venv\Scripts\hermes.exe"
    $icon = if (Test-Path -LiteralPath $hermesIcon) {
        "$hermesIcon,0"
    }
    else {
        "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe,0"
    }

    return New-HermesShortcut `
        -LinkPath $LinkPath `
        -TargetPath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -Arguments "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StartScript`" -HermesRoot `"$RepoRoot`" -Cwd `"$RepoRoot`" -HermesHome `"$HermesHome`"" `
        -WorkingDirectory $RepoRoot `
        -Description "Hermes Desktop connected to the canonical source checkout" `
        -IconLocation $icon `
        -WindowStyle 7
}

function New-HermesStackShortcut {
    param(
        [string]$LinkPath,
        [string]$RepoRoot,
        [string]$StartScript
    )

    return New-HermesPowerShellScriptShortcut `
        -LinkPath $LinkPath `
        -RepoRoot $RepoRoot `
        -StartScript $StartScript `
        -Description "Hermes full stack (Gateway, Hypura, proxies, TUI, FastAPI, ngrok, ...)" `
        -NoExit
}

function Remove-StaleHermesShortcuts {
    param(
        [string[]]$SearchRoots,
        [string]$ShortcutDir
    )

    $removed = @()
    foreach ($root in $SearchRoots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }

        foreach ($name in $HermesShortcutNames) {
            $atRoot = Join-Path $root $name
            if (Test-Path -LiteralPath $atRoot) {
                Remove-Item -LiteralPath $atRoot -Force
                $removed += $atRoot
            }
        }

        if ($ShortcutDir -and (Test-Path -LiteralPath $ShortcutDir)) {
            foreach ($name in $HermesShortcutNames) {
                $inShortcutDir = Join-Path $ShortcutDir $name
                if (Test-Path -LiteralPath $inShortcutDir) {
                    Remove-Item -LiteralPath $inShortcutDir -Force
                    $removed += $inShortcutDir
                }
            }

            $hypuraDup = Join-Path $ShortcutDir "Hermes Hypura Stack.lnk"
            if (Test-Path -LiteralPath $hypuraDup) {
                Remove-Item -LiteralPath $hypuraDup -Force
                $removed += $hypuraDup
            }
        }
    }

    return $removed
}

function Move-LegacyOpenClawShortcuts {
    param(
        [string[]]$SearchRoots,
        [string]$ShortcutDir
    )

    $rows = @()
    $legacyDir = Join-Path $ShortcutDir "Legacy OpenClaw"

    foreach ($root in $SearchRoots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }

        foreach ($name in $LegacyOpenClawShortcutNames) {
            $source = Join-Path $root $name
            if (-not (Test-Path -LiteralPath $source)) { continue }

            if (-not (Test-Path -LiteralPath $legacyDir)) {
                New-Item -ItemType Directory -Path $legacyDir -Force | Out-Null
            }

            $destination = Join-Path $legacyDir $name
            if (Test-Path -LiteralPath $destination) {
                Remove-Item -LiteralPath $destination -Force
            }

            Move-Item -LiteralPath $source -Destination $destination -Force
            $rows += [PSCustomObject]@{
                Path       = $source
                Status     = "moved-legacy-openclaw"
                TargetPath = $destination
            }
        }
    }

    return $rows
}

function Ensure-So8tLlamaShortcut {
    param(
        [string]$Desktop,
        [switch]$Force
    )

    $name = "SuperGemma4 llama-server (RTX3060).lnk"
    $lnkPath = Join-Path $Desktop $name
    $so8tRoot = if ($env:SO8T_ROOT) { $env:SO8T_ROOT } else { Join-Path $env:USERPROFILE "Desktop\SO8T" }
    $so8tScript = Join-Path $so8tRoot "scripts\start-supergemma-server.ps1"

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
$venvStatus = Ensure-ProjectVenv -RepoRoot $RepoRoot
$invoke = Resolve-HermesInvoke -RepoRoot $RepoRoot

$userDesktop = [Environment]::GetFolderPath("Desktop")
$shortcutDir = Join-Path $userDesktop $ShortcutFolderName
New-Item -ItemType Directory -Path $shortcutDir -Force | Out-Null

$searchRoots = @($userDesktop)
if ($IncludePublicDesktop) {
    $searchRoots += [Environment]::GetFolderPath("CommonDesktopDirectory")
}

$removed = @()
if (-not $KeepLegacyDesktopRootShortcuts) {
    $removed = Remove-StaleHermesShortcuts -SearchRoots $searchRoots -ShortcutDir $shortcutDir
}

$legacyOpenClawMoved = @()
if (-not $KeepLegacyOpenClawRootShortcuts) {
    $legacyOpenClawMoved = Move-LegacyOpenClawShortcuts -SearchRoots $searchRoots -ShortcutDir $shortcutDir
}

$definitions = @(
    @{
        Name        = "Hermes Agent CLI.lnk"
        SubArgs     = ""
        Description = "Hermes Agent interactive CLI (hermes)"
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
    },
    @{
        Name        = "Hermes Doctor.lnk"
        SubArgs     = "doctor"
        Description = "Hermes health check (hermes doctor)"
    }
)

. (Join-Path $RepoRoot "scripts\windows\Resolve-CanonicalHermesHome.ps1")
$hermesHome = Resolve-CanonicalHermesHome -RepoRoot $RepoRoot
if (-not (Test-Path -LiteralPath $hermesHome)) {
    New-Item -ItemType Directory -Path $hermesHome -Force | Out-Null
}

$startStackScript = Join-Path $RepoRoot "scripts\windows\start-hermes-stack.ps1"
$startDesktopScript = Join-Path $RepoRoot "scripts\windows\start-hermes-desktop.ps1"

$results = @()
if ($removed.Count -gt 0) {
    foreach ($r in $removed) {
        $results += [PSCustomObject]@{ Path = $r; Status = "removed-stale" }
    }
}

if ($legacyOpenClawMoved.Count -gt 0) {
    $results += $legacyOpenClawMoved
}

if ($venvStatus) {
    $results += [PSCustomObject]@{
        Path   = ".venv"
        Status = $venvStatus.Status
        Error  = $venvStatus.Message
    }
}

foreach ($def in $definitions) {
    $lnk = Join-Path $shortcutDir $def.Name
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

$cfgLnk = Join-Path $shortcutDir "Hermes Config (.hermes).lnk"
try {
    $row = New-HermesShortcut `
        -LinkPath $cfgLnk `
        -TargetPath "$env:SystemRoot\explorer.exe" `
        -Arguments $hermesHome `
        -WorkingDirectory $env:USERPROFILE `
        -Description "Open Hermes config folder (~/.hermes)" `
        -IconLocation "$env:SystemRoot\System32\explorer.exe,0"
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

if (Test-Path -LiteralPath $startDesktopScript) {
    $desktopLnk = Join-Path $shortcutDir "Hermes Desktop.lnk"
    try {
        $row = New-HermesDesktopShortcut -LinkPath $desktopLnk -RepoRoot $RepoRoot -HermesHome $hermesHome -StartScript $startDesktopScript
        $results += [PSCustomObject]@{
            Path             = $row.Path
            Status           = "created"
            TargetPath       = $row.TargetPath
            Arguments        = $row.Arguments
            WorkingDirectory = $row.WorkingDirectory
            LauncherSource   = "desktop-source-canonical"
        }
    }
    catch {
        $results += [PSCustomObject]@{
            Path   = $desktopLnk
            Status = "error"
            Error  = $_.Exception.Message
        }
    }
}
else {
    $results += [PSCustomObject]@{
        Path   = $startDesktopScript
        Status = "skipped-missing-desktop-script"
    }
}

if (Test-Path -LiteralPath $startStackScript) {
    $stackLnk = Join-Path $shortcutDir "Hermes Stack.lnk"
    try {
        $row = New-HermesStackShortcut -LinkPath $stackLnk -RepoRoot $RepoRoot -StartScript $startStackScript
        $results += [PSCustomObject]@{
            Path             = $row.Path
            Status           = "created"
            TargetPath       = $row.TargetPath
            Arguments        = $row.Arguments
            WorkingDirectory = $row.WorkingDirectory
            LauncherSource   = "start-hermes-stack.ps1"
        }
    }
    catch {
        $results += [PSCustomObject]@{
            Path   = $stackLnk
            Status = "error"
            Error  = $_.Exception.Message
        }
    }
}
else {
    $results += [PSCustomObject]@{
        Path   = $startStackScript
        Status = "skipped-missing-stack-script"
    }
}

$startGatewayScript = Join-Path $RepoRoot "scripts\windows\start-hermes-gateway.ps1"
$startLlama3060Script = Join-Path $RepoRoot "scripts\windows\start-hermes-llama-fallback-rtx3060.ps1"
$registerAutostartScript = Join-Path $RepoRoot "scripts\windows\register-hermes-autostart.ps1"
$installAutostartScript = Join-Path $RepoRoot "scripts\windows\install-hermes-autostart.ps1"

$psShortcutDefs = @(
    @{
        Name        = "Hermes Gateway.lnk"
        Script      = $startGatewayScript
        Description = "Start Hermes Gateway (llama RTX3060 fallback first, replace stale gateway)"
        NoExit      = $false
    },
    @{
        Name        = "Hermes Llama Fallback (RTX3060).lnk"
        Script      = $startLlama3060Script
        Description = "Start llama.cpp fallback server on RTX 3060 (port 8080, 64K context)"
        NoExit      = $true
    },
    @{
        Name        = "Hermes Autostart (register).lnk"
        Script      = $(if (Test-Path -LiteralPath $installAutostartScript) { $installAutostartScript } else { $registerAutostartScript })
        Description = "Register Hermes llama + gateway logon autostart (Task Scheduler)"
        NoExit      = $true
    },
    @{
        Name        = "Hermes Autostart (unregister).lnk"
        Script      = $registerAutostartScript
        Description = "Remove Hermes logon autostart tasks and stale Run entries"
        NoExit      = $true
        ExtraArgs   = "-Unregister"
    }
)

foreach ($psDef in $psShortcutDefs) {
    if (-not (Test-Path -LiteralPath $psDef.Script)) {
        $results += [PSCustomObject]@{
            Path   = (Join-Path $shortcutDir $psDef.Name)
            Status = "skipped-missing-script"
            Error  = "Not found: $($psDef.Script)"
        }
        continue
    }

    $lnk = Join-Path $shortcutDir $psDef.Name
    try {
        $startScript = $psDef.Script
        if ($psDef.ExtraArgs) {
            $row = New-HermesShortcut `
                -LinkPath $lnk `
                -TargetPath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
                -Arguments "-NoExit -NoProfile -ExecutionPolicy Bypass -File `"$startScript`" $($psDef.ExtraArgs)" `
                -WorkingDirectory $RepoRoot `
                -Description $psDef.Description `
                -IconLocation "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe,0"
        }
        else {
            $row = New-HermesPowerShellScriptShortcut `
                -LinkPath $lnk `
                -RepoRoot $RepoRoot `
                -StartScript $startScript `
                -Description $psDef.Description `
                -NoExit:($psDef.NoExit)
        }
        $results += [PSCustomObject]@{
            Path             = $row.Path
            Status           = "created"
            TargetPath       = $row.TargetPath
            Arguments        = $row.Arguments
            WorkingDirectory = $row.WorkingDirectory
            LauncherSource   = (Split-Path -Leaf $startScript)
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

if ($DesktopRoot) {
    $desktopRootHermesLnk = Join-Path $userDesktop "Hermes.lnk"
    try {
        if (Test-Path -LiteralPath $startDesktopScript) {
            $row = New-HermesDesktopShortcut -LinkPath $desktopRootHermesLnk -RepoRoot $RepoRoot -HermesHome $hermesHome -StartScript $startDesktopScript
            $results += [PSCustomObject]@{
                Path             = $row.Path
                Status           = "created-desktop-root"
                TargetPath       = $row.TargetPath
                Arguments        = $row.Arguments
                WorkingDirectory = $row.WorkingDirectory
                LauncherSource   = "desktop-source-canonical"
            }
        }
        else {
            $results += [PSCustomObject]@{
                Path   = $desktopRootHermesLnk
                Status = "skipped-missing-desktop-script"
            }
        }
    }
    catch {
        $results += [PSCustomObject]@{
            Path   = $desktopRootHermesLnk
            Status = "error"
            Error  = $_.Exception.Message
        }
    }

    $folderOpenerName = "Hermes Agent (open folder).lnk"
    $desktopHermesDir = Join-Path $userDesktop "01_Hermes_Discord_Setup"
    New-Item -ItemType Directory -Force -Path $desktopHermesDir | Out-Null
    $folderOpenerLnk = Join-Path $desktopHermesDir $folderOpenerName
    try {
        $row = New-HermesShortcut `
            -LinkPath $folderOpenerLnk `
            -TargetPath "$env:SystemRoot\explorer.exe" `
            -Arguments "`"$shortcutDir`"" `
            -WorkingDirectory $userDesktop `
            -Description "Open Desktop\Hermes Agent shortcut folder" `
            -IconLocation "$env:SystemRoot\System32\imageres.dll,3"
        $results += [PSCustomObject]@{
            Path             = $row.Path
            Status           = "created-desktop-root"
            TargetPath       = $row.TargetPath
            Arguments        = $row.Arguments
            WorkingDirectory = $row.WorkingDirectory
            LauncherSource   = "explorer-folder-opener"
        }
    }
    catch {
        $results += [PSCustomObject]@{
            Path   = $folderOpenerLnk
            Status = "error"
            Error  = $_.Exception.Message
        }
    }
}

if ($IncludeLegacySo8tLlamaShortcut) {
    $so8t = Ensure-So8tLlamaShortcut -Desktop $userDesktop -Force:$IncludeLegacySo8tLlamaShortcut
    $results += $so8t
}

if ($IncludePublicDesktop) {
    Write-Warning "Public desktop shortcuts are not mirrored into subfolders; only stale names are removed there."
    foreach ($def in $definitions) {
        Write-Host "Skipping public create for $($def.Name) — use user Desktop\Hermes Agent\" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Hermes launcher: $($invoke.Source) -> $($invoke.HermesPath)" -ForegroundColor Cyan
Write-Host "Repo root: $RepoRoot"
Write-Host "Shortcut folder: $shortcutDir" -ForegroundColor Green
Write-Host "Console shortcuts: cmd.exe /k with pause-on-error." -ForegroundColor DarkGray
Write-Host ""
$results | Format-Table -AutoSize Path, Status, TargetPath, Arguments

$errors = $results | Where-Object { $_.Status -eq "error" }
if ($errors) {
    Write-Warning "Some shortcuts failed; see table above."
    exit 1
}

exit 0

