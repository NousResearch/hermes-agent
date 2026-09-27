# Regression: uv's UTF-8 stdout must remain an executable path under PS 5.1.
# The existing test_install_ps1_script_suites.py runs this under 5.1 and 7.
# Native fixtures write raw bytes to a pipe: a PowerShell function returning
# a .NET string would bypass the decoder and silently miss this regression.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installer = Join-Path $repoRoot 'scripts/install.ps1'
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('hermes-utf8-' + [guid]::NewGuid())
$savedEncoding = [Console]::OutputEncoding
$envNames = @('HERMES_HOME', 'UTF8_PROBE_PYTHON', 'UTF8_PROBE_CALLS', 'UTF8_PROBE_PM', 'UTF8_PROBE_NEEDS_INSTALL')
$savedEnv = @{}
foreach ($name in $envNames) { $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name) }

function Assert-Equal($Expected, $Actual, [string]$Label) {
    if ($Expected -cne $Actual) { throw "${Label}: expected '$Expected', got '$Actual'" }
}

try {
    # ASCII source works when PS 5.1 reads this BOM-less test from a checkout.
    # The actual on-disk path includes CJK, an accent, a space and an apostrophe.
    $profileName = ([string][char]0x5F20) + [char]0x4E09 + ' Ren' + [char]0xE9 + " O'Brien"
    $profileDir = Join-Path $testRoot $profileName
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    $env:HERMES_HOME = Join-Path $profileDir 'hermes-home'
    $testInstallDir = Join-Path $profileDir 'hermes-agent'
    $packageDir = Join-Path $testInstallDir 'pm'
    New-Item -ItemType Directory -Path $packageDir -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $packageDir 'lock.json'), '{"packages":{"python":{"version":"3.14.0+fixture"}}}')
    $env:UTF8_PROBE_PYTHON = Join-Path $profileDir 'python.exe'
    $env:UTF8_PROBE_CALLS = Join-Path $testRoot 'uv-calls.txt'
    $env:UTF8_PROBE_PM = Join-Path $testRoot 'pm-call.txt'
    $fixtureSource = Join-Path $testRoot 'fixture.cs'
    [IO.File]::WriteAllText($fixtureSource, @'
using System;
using System.IO;
using System.Text;
class Fixture {
    static string Env(string key) { return Environment.GetEnvironmentVariable(key); }
    static int Main(string[] args) {
        if (args[0] == "-m") {
            File.WriteAllText(Env("UTF8_PROBE_PM"), String.Join(" ", args));
            return 0;
        }
        if (args[0] == "python") {
            string calls = Env("UTF8_PROBE_CALLS");
            bool first = !File.Exists(calls);
            File.AppendAllText(calls, args[1] + "\n");
            if (args[1] == "install") return 0;
            if (first && Env("UTF8_PROBE_NEEDS_INSTALL") == "1") {
                Console.Error.WriteLine("interpreter not installed");
                return 1;
            }
        }
        byte[] bytes = new UTF8Encoding(false).GetBytes(Env("UTF8_PROBE_PYTHON") + "\r\n");
        Stream stdout = Console.OpenStandardOutput();
        stdout.Write(bytes, 0, bytes.Length);
        return args[0] == "exit7" ? 7 : 0;
    }
}
'@)
    # Use the inbox .NET Framework compiler so both PowerShell editions run
    # the same standalone native process; no uv/Python download is needed.
    $compiler = Join-Path $env:SystemRoot 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    & $compiler /nologo /target:exe "/out:$env:UTF8_PROBE_PYTHON" $fixtureSource
    if ($LASTEXITCODE) { throw 'native fixture compilation failed' }

    . $installer -InstallDir $testInstallDir -HermesHome $env:HERMES_HOME
    # Override acquisition only. Production Get-BootstrapPython, both native
    # find call sites, caching and Stage-PythonDeps still execute unchanged.
    function Get-Uv { return $env:UTF8_PROBE_PYTHON }

    foreach ($codePage in @(936, 437, 65001)) {
        [Console]::OutputEncoding = [Text.Encoding]::GetEncoding($codePage)

        # Contract 2: the actual dependency stage launches the Unicode path on
        # both a cache hit in uv and a miss followed by install + find.
        foreach ($needsInstall in @('0', '1')) {
            $env:UTF8_PROBE_NEEDS_INSTALL = $needsInstall
            Remove-Item -LiteralPath $env:UTF8_PROBE_CALLS, $env:UTF8_PROBE_PM -Force -ErrorAction SilentlyContinue
            $script:BootstrapPython = $null
            Stage-PythonDeps
            Assert-Equal 0 $LASTEXITCODE 'dependency stage succeeds'
            Assert-Equal '-m pm.cli install' ([IO.File]::ReadAllText($env:UTF8_PROBE_PM)) 'resolved executable was invoked'
            $expectedCalls = if ($needsInstall -eq '1') { "find`ninstall`nfind`n" } else { "find`n" }
            Assert-Equal $expectedCalls ([IO.File]::ReadAllText($env:UTF8_PROBE_CALLS)) 'lookup and retry call sequence'
            Assert-Equal $codePage ([Console]::OutputEncoding.CodePage) 'stage preserves caller encoding'
            Assert-Equal $env:UTF8_PROBE_PYTHON (Get-BootstrapPython) 'cached path remains intact'
            Assert-Equal $expectedCalls ([IO.File]::ReadAllText($env:UTF8_PROBE_CALLS)) 'cached lookup does not rerun uv'
        }

        # Contract 1: UTF-8 capture preserves bytes, exit status and caller state,
        # including a terminating failure inside the supplied scriptblock.
        $captured = Invoke-Native -Utf8Output { & $env:UTF8_PROBE_PYTHON exit7 }
        Assert-Equal 7 $LASTEXITCODE 'native exit code survives'
        Assert-Equal $env:UTF8_PROBE_PYTHON $captured 'Unicode output survives'
        Assert-Equal $codePage ([Console]::OutputEncoding.CodePage) 'encoding restored'
        Assert-Equal 'Stop' $ErrorActionPreference 'error preference restored'
        $caught = $false
        try { Invoke-Native -Utf8Output { throw 'fixture failure' } }
        catch { $caught = $true }
        Assert-Equal $true $caught 'terminating errors propagate'
        Assert-Equal $codePage ([Console]::OutputEncoding.CodePage) 'encoding restored on throw'

    }
    Write-Host 'UTF-8 native capture and dependency-stage regression tests passed.'
} finally {
    [Console]::OutputEncoding = $savedEncoding
    foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name, $savedEnv[$name]) }
    Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
}
