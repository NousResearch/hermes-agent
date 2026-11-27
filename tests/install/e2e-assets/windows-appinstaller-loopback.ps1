<#
Exempt App Installer from AppContainer loopback isolation for one update leg, or undo that.

The leg serves its feed on 127.0.0.1. App Installer 1.26 (the version on the windows-2025 runner
image, which gets no Store updates) fetches the .appinstaller descriptor from inside its
AppContainer, which Windows keeps off loopback unless it is exempted. The fetch fails with
0x80072efd and App Installer shows "Cannot open app package. Reason: Error in parsing the app
package." for a package it never downloaded. Installs from the public feed are unaffected.

-Action Add returns $true when it added the exemption and $false when one already existed, so the
caller removes only an exemption it added. Both actions re-read the exemption list and throw when it
did not change: Windows PowerShell 5.1 ignores CheckNetIsolation's exit code.
#>
param([Parameter(Mandatory = $true)][ValidateSet('Add', 'Remove')][string]$Action)

$family = 'microsoft.desktopappinstaller_8wekyb3d8bbwe'

function Test-AppInstallerLoopbackExempt {
    [bool](CheckNetIsolation.exe LoopbackExempt -s | Select-String -SimpleMatch "Name: $family")
}

if ($Action -eq 'Remove') {
    CheckNetIsolation.exe LoopbackExempt -d "-n=$family" | Out-Null
    if (Test-AppInstallerLoopbackExempt) { throw "Could not remove App Installer's loopback exemption" }
    return
}
if (Test-AppInstallerLoopbackExempt) { return $false }
CheckNetIsolation.exe LoopbackExempt -a "-n=$family" | Out-Null
if (-not (Test-AppInstallerLoopbackExempt)) { throw 'Could not exempt App Installer from loopback isolation' }
return $true
