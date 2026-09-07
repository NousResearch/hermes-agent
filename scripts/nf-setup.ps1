# nf-setup.ps1 - North Forge "Setup Run": stamp a drive with its tier + pinned edition.
<#
  scripts\nf-setup.ps1  [-DataDir <path>] [-RepoRoot <path>]
                        [-Tier full|basic] [-Pin <edition>] [-Installed a,b,c]
                        [-Passcode <string>] [-SetPasscode] [-RotatePasscode]
                        [-Force] [-NonInteractive] [-Show]

  WHAT IT DOES
    Writes a SIGNED provisioning record that the North Forge runtime reads on every
    launch to decide what the recipient can reach:

        <DataDir>\north-forge\provisioning.json   { tier, pinned_edition, ... , sig }
        <DataDir>\north-forge\.nf-key             HMAC key (created once, mode 0600)
        <DataDir>\north-forge\.nf-admin           admin-passcode hash (gates re-provision)

    An "edition" is a Hermes profile under <DataDir>\profiles\<name>\ ; the North
    Forge generic chassis is the root ("default"). Two tiers, no third:

      full   - the pin is only the default landing edition; every switch path stays
               open (hermes profile use / the dashboard / /edition). Admin + trusted
               engineers.
      basic  - the pin is the ONLY reachable edition. -p, a hand-edited
               active_profile, `hermes profile use`, the dashboard and /edition all
               refuse anything else, and HERMES_HOME never moves. Everyone else.

    This is a deliberate one-time admin action - it is NOT run by
    bootstrap-north-forge.ps1. Run it once per drive, on an admin machine, after
    the editions you want are installed.

  RE-PROVISION
    Re-running requires the admin passcode once one is set (-Passcode or the
    prompt). First run on a fresh drive family sets it (-SetPasscode, or answer the
    prompt). Editing provisioning.json by hand breaks its signature and the drive
    then refuses to start until repaired here.

  EXIT 0 = provisioning written / shown.  1 = failed or passcode rejected.
  REQUIRES  a bootstrapped North Forge venv (run bootstrap-north-forge.ps1 first).
#>
[CmdletBinding()]
param(
    [string]$DataDir,
    [string]$RepoRoot,
    [ValidateSet('full', 'basic')][string]$Tier,
    [string]$Pin,
    [string[]]$Installed,
    [string]$Passcode,
    [switch]$SetPasscode,
    [switch]$RotatePasscode,
    [switch]$Force,
    [switch]$NonInteractive,
    [switch]$Show
)

$ErrorActionPreference = 'Stop'

# --- locate the checkout, the venv python, and the data dir -------------------
if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot   # scripts\ -> repo root
}
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd([char]'\', [char]'/')
$leaf = Split-Path -Leaf $RepoRoot
$parent = Split-Path -Parent $RepoRoot
if (-not $DataDir) { $DataDir = Join-Path $parent "$leaf-data" }
$DataDir = [System.IO.Path]::GetFullPath($DataDir).TrimEnd([char]'\', [char]'/')

$venvDir = Join-Path $parent "$leaf-venv"
$pyExe = Join-Path $venvDir 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pyExe)) {
    $pyExe = Join-Path $venvDir 'bin/python'          # POSIX venv layout
}
if (-not (Test-Path -LiteralPath $pyExe)) {
    Write-Error "No venv python at $venvDir. Run scripts\bootstrap-north-forge.ps1 first."
    exit 1
}
if (-not (Test-Path -LiteralPath $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
}
$env:HERMES_HOME = $DataDir
$env:PYTHONPATH = $RepoRoot

$script:NfExit = 0
function Invoke-NfTier {
    # Runs `python -m hermes_cli.nf_tier <args>`, prints its output to the host,
    # and sets $script:NfExit to the child's exit code. Does NOT emit to the
    # pipeline (so `$x = Invoke-NfTier` never captures stdout by accident).
    #
    # PowerShell 5.1: a native process that merely writes to stderr can abort the
    # script under `$ErrorActionPreference = 'Stop'`. Relax it locally and rely on
    # $LASTEXITCODE only (same pattern upstream install.ps1 uses).
    param([string[]]$NfArgs, [string]$StdinText)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($PSBoundParameters.ContainsKey('StdinText')) {
            $lines = $StdinText | & $pyExe '-m' 'hermes_cli.nf_tier' @NfArgs 2>&1
        } else {
            $lines = & $pyExe '-m' 'hermes_cli.nf_tier' @NfArgs 2>&1
        }
        $script:NfExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    foreach ($ln in $lines) { Write-Host ([string]$ln) }
}

# --- -Show: just print current state and exit --------------------------------
if ($Show) {
    Invoke-NfTier @('show')
    exit $script:NfExit
}

Write-Host ""
Write-Host "North Forge - Setup Run" -ForegroundColor Cyan
Write-Host "  drive data dir : $DataDir"
$profilesDir = Join-Path $DataDir 'profiles'
$editionDirs = @()
if (Test-Path -LiteralPath $profilesDir) {
    $editionDirs = @(Get-ChildItem -LiteralPath $profilesDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '^\.' } | Select-Object -ExpandProperty Name)
}
Write-Host ("  editions found : " + (($editionDirs -join ', ') -replace '^$', '(none - only the North Forge chassis)'))
Write-Host ""

# --- admin passcode ---------------------------------------------------------
$adminFile = Join-Path $DataDir 'north-forge\.nf-admin'
$adminExists = Test-Path -LiteralPath $adminFile

function Read-Secret([string]$Prompt) {
    $s = Read-Host -AsSecureString $Prompt
    $b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($b) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b) }
}

if ($RotatePasscode -or (-not $adminExists -and $SetPasscode)) {
    if (-not $Passcode) {
        if ($NonInteractive) { Write-Error "Passcode required (-Passcode) with -SetPasscode/-RotatePasscode in -NonInteractive."; exit 1 }
        $Passcode = Read-Secret "  New admin passcode (>= 6 chars)"
        $again = Read-Secret "  Repeat"
        if ($Passcode -ne $again) { Write-Error "Passcodes did not match."; exit 1 }
    }
    $saArgs = @('set-admin')
    if ($RotatePasscode) { $saArgs += '--rotate' }
    Invoke-NfTier $saArgs -StdinText $Passcode
    if ($script:NfExit -ne 0) { Write-Error "Could not set the admin passcode."; exit 1 }
    $adminExists = $true
    Write-Host "  admin passcode : set" -ForegroundColor Green
    # Passcode-only invocation (no tier/pin asked for): done.
    if (-not $Tier -and -not $Pin) { exit 0 }
}
elseif ($adminExists -and -not $Passcode -and -not $NonInteractive) {
    $Passcode = Read-Secret "  Admin passcode for this drive"
}

# --- tier ----------------------------------------------------------------------
if (-not $Tier) {
    if ($NonInteractive) { Write-Error "-Tier is required in -NonInteractive mode."; exit 1 }
    $ans = Read-Host "  Tier - [F]ull (admin/engineer, switcher on) or [B]asic (locked to one edition)?  [B]"
    switch (($ans + 'b').Substring(0, 1).ToLower()) {
        'f' { $Tier = 'full' }
        default { $Tier = 'basic' }
    }
}
Write-Host "  tier           : $Tier"

# --- pinned edition ----------------------------------------------------------
if (-not $Pin) {
    if ($NonInteractive) { Write-Error "-Pin is required in -NonInteractive mode ('default' = the chassis)."; exit 1 }
    $hint = if ($editionDirs) { " (installed: $($editionDirs -join ', '); or 'default' for the plain chassis)" } else { " ('default' = the plain North Forge chassis)" }
    $Pin = Read-Host "  Pinned edition$hint"
    if (-not $Pin) { $Pin = 'default' }
}
Write-Host "  pinned edition : $Pin"

# --- installed editions (Full tier only; Basic records exactly its pin) -------
$installedArg = ''
if ($Tier -eq 'full') {
    if (-not $Installed -or $Installed.Count -eq 0) { $Installed = $editionDirs }
    $installedArg = ($Installed -join ',')
    Write-Host ("  installed list : " + ($installedArg -replace '^$', '(none)'))
}
else {
    $others = @($editionDirs | Where-Object { $_.ToLower() -ne $Pin.ToLower() })
    if ($others.Count -gt 0) {
        Write-Warning ("Basic tier pins '$Pin' but this drive also carries other editions: " + ($others -join ', ') +
            ". A Basic drive should not ship other editions' content - consider removing those profile folders.")
    }
}

# --- confirm -----------------------------------------------------------------
if (-not $NonInteractive) {
    $verb = if (Test-Path -LiteralPath (Join-Path $DataDir 'north-forge\provisioning.json')) { "RE-PROVISION" } else { "provision" }
    $ok = Read-Host "  $verb this drive as [$Tier] pinned to [$Pin]?  [y/N]"
    if ($ok -notmatch '^(y|yes)$') { Write-Host "  aborted."; exit 1 }
}

# --- write it --------------------------------------------------------------
$nfArgs = @('provision', '--tier', $Tier, '--pin', $Pin)
if ($installedArg) { $nfArgs += @('--installed', $installedArg) }
if ($Force) { $nfArgs += '--force' }
$whoami = "$env:COMPUTERNAME\$env:USERNAME"
$nfArgs += @('--by', $whoami)

if ($adminExists) {
    if (-not $Passcode) { Write-Error "Admin passcode required to (re-)provision this drive."; exit 1 }
    $nfArgs += '--passcode-stdin'
    Invoke-NfTier $nfArgs -StdinText $Passcode
}
else {
    Invoke-NfTier $nfArgs
    if ($script:NfExit -eq 0 -and -not $NonInteractive) {
        Write-Host ""
        Write-Warning "No admin passcode is set on this drive - anyone can re-run nf-setup.ps1 and change the tier."
        Write-Host "  Set one now:  scripts\nf-setup.ps1 -SetPasscode" -ForegroundColor Yellow
    }
}
if ($script:NfExit -ne 0) { Write-Error "Provisioning failed (exit $script:NfExit)."; exit 1 }

Write-Host ""
Write-Host "Provisioned." -ForegroundColor Green
Invoke-NfTier @('show')
exit 0
