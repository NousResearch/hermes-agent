# Pipeline/array contracts only; no package installation or live process changes.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'desktop-smoke-windows.ps1')

# Like a CIM process record, this object has no Count property of its own.
class BundledProcessRecord {
    [int]$ProcessId
    [string]$ExecutablePath
    [string]$CommandLine
    BundledProcessRecord([int]$Id, [string]$Exe, [string]$Command) {
        $this.ProcessId = $Id
        $this.ExecutablePath = $Exe
        $this.CommandLine = $Command
    }
}
$exe = 'C:\fixture\Hermes.exe'
$script:processRecords = @()
function Get-CimInstance([string]$ClassName) {
    if ($ClassName -cne 'Win32_Process') { throw 'Unexpected CIM query' }
    $script:processRecords
}

foreach ($count in 0, 1, 2) {
    $script:processRecords = @(for ($i = 0; $i -lt $count; $i++) {
        [BundledProcessRecord]::new(100 + $i, $exe, 'Hermes.exe')
    })
    # The OLD identity check and NEW relaunch poll both assign the helper directly.
    $rows = Main-Processes $exe
    if ($rows -isnot [array] -or $rows.Count -ne $count) { throw "Lost process array for $count row(s)" }
    for ($i = 0; $i -lt $count; $i++) {
        if ($rows[$i].ProcessId -ne 100 + $i) { throw 'Process identity/order lost' }
    }
}
Write-Output 'PASS: zero, one and multiple process rows preserve Count and indexing'

$script:processRecords = @(
    [BundledProcessRecord]::new(100, $exe.ToUpperInvariant(), 'Hermes.exe'),
    [BundledProcessRecord]::new(101, $exe, 'Hermes.exe --type=renderer'),
    [BundledProcessRecord]::new(102, $exe, 'Hermes.exe --type utility'),
    [BundledProcessRecord]::new(103, 'C:\other\Hermes.exe', 'Hermes.exe'),
    [BundledProcessRecord]::new(104, '', 'Hermes.exe')
)
$rows = Main-Processes $exe
if ($rows.Count -ne 1 -or $rows[0].ProcessId -ne 100) { throw 'Installed main-process filtering changed' }
Write-Output 'PASS: exact executable identity excludes Electron children and unrelated processes'
