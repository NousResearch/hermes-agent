# ============================================================================
# windows-bundled-drive-update.ps1 — drive the REAL in-app update surface of
# the INSTALLED (MSIX) desktop app via UI Automation.
# ============================================================================
# The packaged arm's whole point: a real user trigger starts the update
# against the controlled feed. No Playwright Electron-driver attach (the
# package must run AS INSTALLED, not from a checkout), so this helper drives
# UIA exactly like the reference technique (package-update-acceptance.md):
# the installed Electron UI exposes text and controls once Chromium's
# accessibility engages for a UIA client.
#
# Sequence, each step polled with screenshots into -ProofDir:
#   1. find the Hermes window (by process id, NOT window-title guesswork)
#   2. click the in-app update trigger: the "Update now" button (i18n 'en'
#      updates-overlay: "A new version of Hermes is ready. Update now and
#      Windows will finish it for you."). Names matched case-insensitively
#      on 'update' for the button the app owns; NEVER an internal apply call.
#   3. when the OS App Installer confirmation window appears, click its
#      Update/Install button (UpdateSettings OnLaunch ShowPrompt=false means
#      it usually does NOT appear; tolerate both).
#   4. wait for the app processes to exit (the hand-off quits the app) —
#      exit 0 = the trigger was clicked by THIS driver and the app exited.
#
# The driver never launches a new app instance: proving the NEW process
# without driver launch is the parent driver's job (windows-bundled-e2e.ps1).
# ============================================================================

param(
    [Parameter(Mandatory = $true)][int]$OldProcessId,
    [Parameter(Mandatory = $true)][string]$ProofDir,
    [string]$ResultPath = "",
    # How long to keep looking for the in-app button (the app may still be
    # booting its backend when the driver starts).
    [int]$ButtonTimeoutSec = 300,
    # How long to wait for the App Installer confirmation dialog, if the OS
    # shows one at all.
    [int]$ConfirmTimeoutSec = 120,
    # How long to wait for the app processes to exit after the click.
    [int]$ExitTimeoutSec = 600
)

$ErrorActionPreference = "Stop"
if ($env:GITHUB_ACTIONS -ne 'true') { throw 'Disposable CI runner required' }
$oldProcess = Get-Process -Id $OldProcessId
$oldBirth = $oldProcess.StartTime
$ProcessName = $oldProcess.ProcessName
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false

Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

New-Item -ItemType Directory -Path $ProofDir -Force | Out-Null
if (-not $ResultPath) { $ResultPath = Join-Path $ProofDir "drive-update-result.json" }

function Save-Shot([string]$Name) {
    try {
        $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
        $bmp = New-Object System.Drawing.Bitmap($b.Width, $b.Height)
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
        $bmp.Save((Join-Path $ProofDir "$Name.png"), [System.Drawing.Imaging.ImageFormat]::Png)
        $g.Dispose(); $bmp.Dispose()
    } catch { Write-Host "  [uia] screenshot $Name failed: $($_.Exception.Message)" }
}

function Get-AppPids {
    $current = Get-Process -Id $OldProcessId -ErrorAction SilentlyContinue
    if ($current -and $current.StartTime -eq $oldBirth) { return @($OldProcessId) }
    return @()
}

# UIA element helpers --------------------------------------------------------

function Find-AppWindow {
    param([int[]]$Pids)
    if (-not $Pids -or $Pids.Count -eq 0) { return $null }
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $Pids[0])
    # The process hosts several top-level windows (main + hidden helpers);
    # prefer one with a window handle (visible).
    $wins = $root.FindAll([System.Windows.Automation.TreeScope]::Children, $cond)
    foreach ($w in $wins) {
        if ($w.Current.NativeWindowHandle -ne 0) { return $w }
    }
    if ($wins.Count -gt 0) { return $wins[0] }
    return $null
}

function Find-InvokableButton {
    # Depth-first walk for a button whose Name matches one of the regexes.
    param($Root, [string[]]$Patterns)
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Button)
    $buttons = $Root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
    foreach ($b in $buttons) {
        $name = ""
        try { $name = $b.Current.Name } catch {}
        if (-not $name) { continue }
        foreach ($pat in $Patterns) {
            if ($name -match $pat) { return @{ Element = $b; Name = $name } }
        }
    }
    return $null
}

function Invoke-Button($Element) {
    $invoke = $null
    if ($Element.TryGetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern, [ref]$invoke)) {
        $invoke.Invoke()
        return $true
    }
    # Fall back to a legacy click through the legacy pattern.
    $legacy = $null
    if ($Element.TryGetCurrentPattern([System.Windows.Automation.LegacyIAccessiblePattern]::Pattern, [ref]$legacy)) {
        try { $legacy.DoDefaultAction(); return $true } catch {}
    }
    return $false
}

# 1. the in-app Update trigger ------------------------------------------------

Write-Host "[uia] waiting for the $ProcessName window and its Update now button (up to ${ButtonTimeoutSec}s) ..."
$pids = Get-AppPids
if ($pids.Count -eq 0) { throw "no $ProcessName process is running - the install phase must leave the OLD app running" }

$deadline = (Get-Date).AddSeconds($ButtonTimeoutSec)
$clicked = $null
while ((Get-Date) -lt $deadline -and -not $clicked) {
    $pids = Get-AppPids
    $win = Find-AppWindow -Pids $pids
    if ($win) {
        $hit = Find-InvokableButton -Root $win -Patterns @(
            '^Update now',            # en: the packaged overlay's primary action
            'Update Hermes',          # en settings: About surface
            '^Update$'
        )
        if ($hit) {
            Save-Shot "01-before-update-click"
            Write-Host "  [uia] clicking button '$($hit.Name)'"
            if (Invoke-Button $hit.Element) { $clicked = $hit } else { Write-Host "  [uia] invoke pattern unavailable; retrying" }
        } else {
            # Use the visible settings flow; never call the updater bridge.
            $navigation = Find-InvokableButton -Root $win -Patterns @('^Open settings$', '^About$', '^Check now$', 'choose a provider later')
            if ($navigation) { [void](Invoke-Button $navigation.Element) }
        }
    }
    if (-not $clicked) { Start-Sleep -Seconds 3 }
}
if (-not $clicked) {
    Save-Shot "zz-no-update-button"
    throw "never found an invokable Update button in the $ProcessName UI after $ButtonTimeoutSec s"
}
Write-Host "  [uia] in-app update trigger clicked: $($clicked.Name)"

# 2. the OS App Installer confirmation, if the OS shows one -------------------

Write-Host "[uia] watching for an App Installer confirmation (up to ${ConfirmTimeoutSec}s) ..."
$cDeadline = (Get-Date).AddSeconds($ConfirmTimeoutSec)
$confirmed = $false
while ((Get-Date) -lt $cDeadline -and -not $confirmed) {
    $installer = Get-Process -Name "AppInstaller" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($installer) {
        $root = [System.Windows.Automation.AutomationElement]::RootElement
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $installer.Id)
        $win = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $cond)
        if ($win) {
            $hit = Find-InvokableButton -Root $win -Patterns @('^Update$', '^Install$', '^(Reinstall|Update) .*')
            if ($hit) {
                Save-Shot "02-appinstaller-confirm"
                Write-Host "  [uia] App Installer dialog: clicking '$($hit.Name)'"
                if (Invoke-Button $hit.Element) { $confirmed = $true }
            }
        }
    }
    # The app quits before App Installer opens; keep watching its confirmation.
    Start-Sleep -Seconds 3
}
if ($confirmed) { Write-Host "  [uia] App Installer confirmation clicked" }
else { Write-Host "  [uia] no App Installer dialog appeared (OnLaunch ShowPrompt=false + real source registration): ok" }
Save-Shot "03-after-trigger"

# 3. wait for the app to exit (the hand-off quits it; never kill it here) -----

Write-Host "[uia] waiting for $ProcessName to exit (up to ${ExitTimeoutSec}s) ..."
$eDeadline = (Get-Date).AddSeconds($ExitTimeoutSec)
while ((Get-Date) -lt $eDeadline) {
    if ((Get-AppPids).Count -eq 0) {
        Save-Shot "04-app-exited"
        Write-Host "  [uia] OLD app exited for the package swap"
        @{ ok = $true; clicked = $clicked.Name; appInstallerConfirmed = [bool]$confirmed; exited = $true } |
            ConvertTo-Json | Set-Content -LiteralPath $ResultPath -Encoding UTF8
        exit 0
    }
    Start-Sleep -Seconds 5
}
Save-Shot "zz-app-never-exited"
@{ ok = $false; clicked = $clicked.Name; appInstallerConfirmed = [bool]$confirmed; exited = $false } |
    ConvertTo-Json | Set-Content -LiteralPath $ResultPath -Encoding UTF8
throw "the $ProcessName process never exited after the update trigger ($ExitTimeoutSec s) - a surviving process would hold package files during the swap"
