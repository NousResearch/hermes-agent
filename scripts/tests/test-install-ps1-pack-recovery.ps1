param([string]$WorkRoot = $env:TEMP, [string]$PackagingRoot, [string]$InstallerPath,
    [string[]]$Scenarios = @('failure', 'timeout', 'custom-timeout', 'recovery-failure'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
if (-not $PackagingRoot) { $PackagingRoot = Join-Path $repoRoot 'apps/desktop' }
if (-not $InstallerPath) { $InstallerPath = Join-Path $repoRoot 'scripts/install.ps1' }
$nodeExe = (Get-Command node -CommandType Application | Select-Object -First 1).Source
$utf8 = [Text.UTF8Encoding]::new($false)
foreach ($scenario in $Scenarios) {
    $caseRoot = Join-Path $WorkRoot ('pack-recovery-' + $scenario + '-' + [guid]::NewGuid().ToString('N'))
    $install = Join-Path $caseRoot 'repo'
    $desktop = Join-Path $install 'apps/desktop'
    $scripts = Join-Path $desktop 'scripts'
    $bin = Join-Path $caseRoot 'bin'
    $output = if ($scenario -eq 'custom-timeout') { Join-Path $caseRoot 'custom output/win-unpacked' } else { Join-Path $desktop 'release/win-unpacked' }
    $package = Join-Path $desktop 'node_modules/electron-builder'
    foreach ($dir in @($scripts, $bin, $output, $package)) { [IO.Directory]::CreateDirectory($dir) | Out-Null }
    foreach ($file in @('desktop-pack-runner.mjs', 'desktop-pack-transaction.mjs', 'before-pack.mjs', 'stage-native-deps.mjs', 'utils.mjs')) {
        [IO.File]::Copy((Join-Path $PackagingRoot "scripts/$file"), (Join-Path $scripts $file))
    }
    # Only electron-builder's unrelated Arch enum is substituted. The actual
    # beforePack, wrapper, journal, settlement and Windows native boundary run.
    [IO.File]::WriteAllText((Join-Path $package 'package.json'), '{"type":"module","exports":"./index.js"}', $utf8)
    [IO.File]::WriteAllText((Join-Path $package 'index.js'), 'export const Arch = {ia32:0,x64:1,arm64:3};', $utf8)
    [IO.File]::WriteAllText((Join-Path $desktop 'package.json'), '{}', $utf8)
    $original = New-Object byte[] 1024
    $original[0] = 0x4d; $original[1] = 0x5a; $original[0x3c] = 0x80
    $original[0x80] = 0x50; $original[0x81] = 0x45
    $original[0x84] = 0x64; $original[0x85] = 0x86; $original[0x86] = 1
    $original[0x98 + 17] = 2; $original[0x98 + 21] = 2; $original[1023] = 0x42
    [IO.File]::WriteAllBytes((Join-Path $output 'Hermes.exe'), $original)
    [IO.File]::WriteAllText((Join-Path $scripts 'fixture-builder.mjs'), @'
import fs from 'node:fs';
import path from 'node:path';
import beforePack from './before-pack.mjs';
const output = process.env.PACK_FIXTURE_OUTPUT;
await beforePack({appOutDir: output, electronPlatformName: 'win32'});
fs.mkdirSync(output, {recursive:true});
fs.writeFileSync(path.join(output, 'Hermes.exe'), 'MZ-truncated');
fs.writeFileSync(path.join(process.env.PACK_FIXTURE_ROOT, 'journal-path'), process.env.HERMES_DESKTOP_PACK_JOURNAL);
if (process.env.PACK_FIXTURE_SCENARIO === 'failure') process.exit(7);
if (process.env.PACK_FIXTURE_SCENARIO === 'recovery-failure') fs.appendFileSync(process.env.HERMES_DESKTOP_PACK_JOURNAL, '{incomplete');
setInterval(() => fs.appendFileSync(path.join(process.env.PACK_FIXTURE_ROOT, 'writer.log'), 'x'), 50);
'@, $utf8)
    [IO.File]::WriteAllText((Join-Path $scripts 'fixture-npm.mjs'), @'
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {runDesktopBuilder} from './desktop-pack-runner.mjs';
if (process.argv[2] !== 'run') process.exit(0);
fs.appendFileSync(path.join(process.env.PACK_FIXTURE_ROOT, 'pack-attempts'), 'attempt\n');
const result = runDesktopBuilder(process.execPath, [fileURLToPath(new URL('./fixture-builder.mjs', import.meta.url))]);
process.exit(result.status);
'@, $utf8)
    $shim = '@"' + $nodeExe + '" "' + (Join-Path $scripts 'fixture-npm.mjs') + '" %*'
    [IO.File]::WriteAllText((Join-Path $bin 'npm.cmd'), $shim + "`r`n", [Text.Encoding]::ASCII)
    $electron = Join-Path $install 'node_modules/electron'
    [IO.Directory]::CreateDirectory($electron) | Out-Null
    [IO.File]::WriteAllText((Join-Path $electron 'package.json'), '{}', $utf8)
    [IO.File]::WriteAllText((Join-Path $electron 'install.js'), 'process.exit(9);', $utf8)
    $env:PATH = $bin + ';' + [IO.Path]::GetDirectoryName($nodeExe) + ';' + $PSHOME
    $env:TEMP = $caseRoot; $env:TMP = $caseRoot
    $env:PACK_FIXTURE_ROOT = $caseRoot
    $env:PACK_FIXTURE_OUTPUT = $output
    $env:PACK_FIXTURE_SCENARIO = $scenario
    $env:ELECTRON_MIRROR = 'fixture-download-fails'
    $env:GITHUB_SHA = '0000000000000000000000000000000000000001'
    $env:GITHUB_REF_NAME = 'fixture'
    $env:ELECTRON_CACHE = Join-Path $caseRoot 'electron-cache'
    [IO.Directory]::CreateDirectory($env:ELECTRON_CACHE) | Out-Null
    [IO.File]::WriteAllText((Join-Path $env:ELECTRON_CACHE 'electron-fixture.zip'), 'bad download')
    . $InstallerPath -InstallDir $install -HermesHome (Join-Path $caseRoot 'home')
    function Test-Node { $true }
    function New-DesktopShortcuts { }
    function icacls { $global:LASTEXITCODE = 0 }
    function Get-ChildItem {
        [CmdletBinding()] param([string]$LiteralPath, [string]$Path, [switch]$Recurse, [string]$Filter, [switch]$File, [switch]$Directory)
        $target = if ($LiteralPath) { $LiteralPath } else { $Path }
        if ($target -and [IO.Path]::GetFullPath($target).StartsWith($caseRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
            Microsoft.PowerShell.Management\Get-ChildItem @PSBoundParameters
        }
    }
    $script:InstallerCommandTimeouts.Desktop = 8
    $failure = ''
    try { Install-Desktop } catch { $failure = [string]$_ }
    if (-not $failure) { throw "$scenario falsely reported installation success" }
    if ($scenario -eq 'recovery-failure') {
        if ($failure -notmatch 'rollback did not complete') { throw "Wrong rollback failure: $failure" }
        $held = Join-Path ($output + '.bak') 'Hermes.exe'
        $journalPath = [IO.File]::ReadAllText((Join-Path $caseRoot 'journal-path'))
        if (-not (Test-Path -LiteralPath $journalPath)) { throw 'Failed recovery discarded its journal' }
    } else {
        $held = Join-Path $output 'Hermes.exe'
        if ($failure -notmatch 'build failed') { throw "Wrong packaging failure: $failure" }
    }
    if (-not (Test-Path -LiteralPath $held) -or [Convert]::ToBase64String([IO.File]::ReadAllBytes($held)) -ne [Convert]::ToBase64String($original)) {
        throw "$scenario lost the original packaged app after builder failure/cache cleanup"
    }
    if ([IO.File]::ReadAllLines((Join-Path $caseRoot 'pack-attempts')).Count -ne 1) { throw 'Failed recovery launched another pack' }
    $writer = Join-Path $caseRoot 'writer.log'
    if (Test-Path -LiteralPath $writer) {
        $before = (Get-Item -LiteralPath $writer).Length
        Start-Sleep -Milliseconds 500
        if ((Get-Item -LiteralPath $writer).Length -ne $before) { throw 'Expired builder continued writing during recovery' }
    }
    Write-Host "PASS: $scenario preserves the original generation through the native installer path"
}
