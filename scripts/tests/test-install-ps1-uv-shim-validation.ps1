# Behavioral tests for install.ps1 managed-uv acceptance (issue #110350).
#
# `& uv.exe --version` never throws on a nonzero exit, so Install-Uv used to
# trust any file at $HermesHome\bin\uv.exe -- including the Chocolatey ShimGen
# launcher it had itself copied there, which resolves the real uv RELATIVE to
# its own location and is therefore dead after the copy.  The installer is
# dot-sourced without running its entry point; the uv installer rungs and PATH
# lookup are replaced with in-process stubs; the fake uv binaries are tiny
# compiled console apps so the real spawn/exit-code contract is exercised.

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts\install.ps1'
$testRoot = Join-Path $env:TEMP ("hermes-uv-shim-test-" + [Guid]::NewGuid().ToString('N'))
$HermesHome = Join-Path $testRoot 'home'
$InstallDir = Join-Path $testRoot 'missing-checkout'
New-Item -ItemType Directory -Force -Path $testRoot | Out-Null
. $installScript -HermesHome $HermesHome -InstallDir $InstallDir

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:Failures = 0
function Assert-Equal {
    param($Expected, $Actual, [string]$Label)
    if ($Expected -ceq $Actual) {
        Write-Host "PASS: $Label"
    } else {
        Write-Host "FAIL: $Label"
        Write-Host "  expected: [$Expected]"
        Write-Host "  actual:   [$Actual]"
        $script:Failures++
    }
}

# -- fake uv binaries -------------------------------------------------------
# Compiled with Windows PowerShell 5.1 (always present on a Windows host) so
# the result is a real console .exe regardless of which host runs this test.
function New-FakeExe {
    param([string]$Name, [string]$Source)
    $srcPath = Join-Path $testRoot "$Name.cs"
    $exePath = Join-Path $testRoot "$Name.exe"
    [IO.File]::WriteAllText($srcPath, $Source)
    $winPs = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    & $winPs -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -Path '$srcPath' -OutputAssembly '$exePath' -OutputType ConsoleApplication" | Out-Null
    if (-not (Test-Path $exePath)) { throw "failed to compile $Name" }
    return $exePath
}

# A standalone uv: always answers `uv 0.1.0`.
$okExe = New-FakeExe 'fake-uv-ok' @'
public static class FakeUv {
    public static int Main(string[] args) {
        System.Console.WriteLine("uv 0.1.0");
        return 0;
    }
}
'@

# A ShimGen-style launcher: works only while ..\lib\uv\tools\uv.exe exists
# relative to ITS OWN location -- exactly what Chocolatey's bin\uv.exe does.
$shimExe = New-FakeExe 'fake-uv-shim' @'
using System.IO;
public static class FakeUvShim {
    public static int Main(string[] args) {
        string self = System.Reflection.Assembly.GetExecutingAssembly().Location;
        string target = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(self), "..", "lib", "uv", "tools", "uv.exe"));
        if (File.Exists(target)) {
            System.Console.WriteLine("uv 0.1.0");
            return 0;
        }
        System.Console.Error.WriteLine("Cannot find file at '..\\lib\\uv\\tools\\uv.exe' (" + target + "). This usually indicates a missing or moved file.");
        return 1;
    }
}
'@

# Chocolatey layout: bin\uv.exe (shim) -> lib\uv\tools\uv.exe (real).
$chocoRoot = Join-Path $testRoot 'chocolatey'
New-Item -ItemType Directory -Force -Path (Join-Path $chocoRoot 'bin'), (Join-Path $chocoRoot 'lib\uv\tools') | Out-Null
$chocoShim = Join-Path $chocoRoot 'bin\uv.exe'
$chocoReal = Join-Path $chocoRoot 'lib\uv\tools\uv.exe'
Copy-Item $shimExe $chocoShim
Copy-Item $okExe $chocoReal

# Scoop layout: shims\uv.exe + shims\uv.shim (`path = "..."`) -> apps\uv\current\uv.exe.
$scoopRoot = Join-Path $testRoot 'scoop'
New-Item -ItemType Directory -Force -Path (Join-Path $scoopRoot 'shims'), (Join-Path $scoopRoot 'apps\uv\current') | Out-Null
$scoopShim = Join-Path $scoopRoot 'shims\uv.exe'
$scoopReal = Join-Path $scoopRoot 'apps\uv\current\uv.exe'
Copy-Item $shimExe $scoopShim
Copy-Item $okExe $scoopReal
"path = `"$scoopReal`"" | Set-Content -LiteralPath (Join-Path $scoopRoot 'shims\uv.shim') -Encoding Ascii

# A relocated launcher: the shim copied somewhere without a lib\ tree.
$strayDir = Join-Path $testRoot 'stray'
New-Item -ItemType Directory -Force -Path $strayDir | Out-Null
$strayShim = Join-Path $strayDir 'uv.exe'
Copy-Item $shimExe $strayShim

Write-Host '-- Test-ManagedUvBinary --'
Assert-Equal 'uv 0.1.0' (Test-ManagedUvBinary $okExe) 'standalone uv is accepted with its version line'
Assert-Equal 'uv 0.1.0' (Test-ManagedUvBinary $chocoShim) 'launcher runs at its original location'
Assert-Equal $null (Test-ManagedUvBinary $strayShim) 'relocated launcher (exit 1, no throw) is rejected'
Assert-Equal $null (Test-ManagedUvBinary (Join-Path $testRoot 'absent\uv.exe')) 'missing path is rejected'

Write-Host ''
Write-Host '-- Resolve-UvShimTarget --'
Assert-Equal $chocoReal (Resolve-UvShimTarget $chocoShim) 'Chocolatey shim resolves to lib\uv\tools\uv.exe'
Assert-Equal $scoopReal (Resolve-UvShimTarget $scoopShim) 'Scoop shim resolves through its .shim sidecar'
Assert-Equal $okExe (Resolve-UvShimTarget $okExe) 'plain executable resolves to itself'

# -- Install-Uv flow with stubbed installer rungs and PATH lookup ------------
$managedUv = Join-Path $HermesHome 'bin\uv.exe'
$script:InstallerCalls = 0
$script:FakePathUv = $null
$script:InfoLog = @()
$env:USERPROFILE = $testRoot   # no ~\.local\bin\uv.exe candidate

function Invoke-FakeUvInstaller { $script:InstallerCalls++ }
function Get-PowerShellHostExe { 'Invoke-FakeUvInstaller' }
function Get-Command {
    [CmdletBinding()]
    param([Parameter(Position = 0)][string]$Name, [object]$CommandType)
    if ($Name -eq 'uv' -and $script:FakePathUv) {
        return [pscustomobject]@{ Source = $script:FakePathUv }
    }
    return $null
}
function Write-Info { param([string]$Message) $script:InfoLog += $Message }
function Write-Success { param([string]$Message) }
function Write-Err { param([string]$Message) }

function Invoke-InstallUvScenario {
    param([string]$PreplacedManaged, [string]$PathUv)
    if (Test-Path $HermesHome) { Remove-Item -LiteralPath $HermesHome -Recurse -Force }
    New-Item -ItemType Directory -Force -Path (Join-Path $HermesHome 'bin') | Out-Null
    if ($PreplacedManaged) { Copy-Item $PreplacedManaged $managedUv }
    $script:FakePathUv = $PathUv
    $script:InstallerCalls = 0
    $script:InfoLog = @()
    $script:UvCmd = $null
    return (Install-Uv)
}

Write-Host ''
Write-Host '-- working uv already at the managed location --'
$ok = Invoke-InstallUvScenario -PreplacedManaged $okExe
Assert-Equal $true $ok 'stage succeeds'
Assert-Equal 0 $script:InstallerCalls 'no reinstall attempted'
Assert-Equal $managedUv $script:UvCmd 'managed uv is the resolved command'

Write-Host ''
Write-Host '-- broken uv pre-placed at the managed location (re-run recovery) --'
$ok = Invoke-InstallUvScenario -PreplacedManaged $strayShim
Assert-Equal $false $ok 'stage fails honestly when nothing valid can be installed'
Assert-Equal 2 $script:InstallerCalls 'broken copy is replaced: both installer rungs run'
Assert-Equal $false (Test-Path $managedUv) 'nothing broken is left at the managed location'

$ok = Invoke-InstallUvScenario -PreplacedManaged $strayShim -PathUv $chocoShim
Assert-Equal $true $ok 'broken managed copy is replaced from a Chocolatey uv on PATH'
Assert-Equal 'uv 0.1.0' (Test-ManagedUvBinary $managedUv) 'the salvaged managed copy works at its new location'
Assert-Equal (Get-Item $okExe).Length (Get-Item $managedUv).Length 'the real binary was copied, not the launcher'

Write-Host ''
Write-Host '-- broken candidate on PATH is not copied --'
$ok = Invoke-InstallUvScenario -PathUv $strayShim
Assert-Equal $false $ok 'stage fails instead of trusting a dead launcher'
Assert-Equal $false (Test-Path $managedUv) 'dead launcher was never copied into bin'
Assert-Equal $true (@($script:InfoLog -like '*does not run*').Count -gt 0) 'the reason is logged'

Write-Host ''
Write-Host '-- working uv on PATH is salvaged --'
$ok = Invoke-InstallUvScenario -PathUv $okExe
Assert-Equal $true $ok 'stage succeeds via salvage'
Assert-Equal 'uv 0.1.0' (Test-ManagedUvBinary $managedUv) 'salvaged copy validates'

if ($script:Failures -gt 0) {
    Write-Host ''
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ''
Write-Host 'all assertions passed'

if (Test-Path $testRoot) {
    Remove-Item -LiteralPath $testRoot -Recurse -Force
}
