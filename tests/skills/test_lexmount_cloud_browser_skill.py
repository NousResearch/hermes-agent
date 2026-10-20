"""Native installer checks: reject bad checksums and propagate CLI failures."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SKILL = Path(__file__).resolve().parents[2] / 'optional-skills/research/lexmount-cloud-browser'
CASES = ['valid', 'mismatch', 'missing', 'cli_failure']


def _check_installer(tmp_path, mode, target, windows=False):
    skill = tmp_path / 'package with spaces'
    shutil.copytree(SKILL / 'scripts', skill / 'scripts')
    source = tmp_path / ('download.exe' if windows else 'download')
    sums = tmp_path / 'sums'
    marker = tmp_path / 'executed'
    # Pass only process-location variables; never provider keys or release overrides.
    env = {k: os.environ[k] for k in (
        'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PROCESSOR_ARCHITECTURE',
        'PROCESSOR_ARCHITEW6432', 'TEMP', 'TMP',
    ) if k in os.environ}
    env.update({'PATH': os.defpath, 'TEST_MARKER': str(marker),
                'TEST_SUMS': str(sums), 'TEST_DOWNLOAD': str(source),
                'TEST_BOOTSTRAP': str(skill / 'scripts/bootstrap.ps1')})
    if windows:
        shell = str(Path(os.environ['SYSTEMROOT']) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
        # Compile a tiny native test executable with Windows' existing .NET compiler.
        code = ('using System; using System.IO; public class InstallerProbe {'
                'public static void Main(string[] args) {'
                'File.WriteAllText(Environment.GetEnvironmentVariable("TEST_MARKER"), "verified");'
                f'Environment.Exit({9 if mode == "cli_failure" else 0});' + '}}')
        command = [shell, '-NoProfile', '-NonInteractive', '-Command']
        compiled = subprocess.run(command + [
            "Add-Type -TypeDefinition '" + code + "' -OutputAssembly $env:TEST_DOWNLOAD -OutputType ConsoleApplication"
        ], env=env, capture_output=True, text=True, timeout=60)
        assert compiled.returncode == 0, compiled.stderr
        assert source.exists(), compiled.stdout + compiled.stderr
        mock = '''
$ErrorActionPreference = 'Stop'
function Invoke-WebRequest {
    param([switch]$UseBasicParsing, [string]$Uri, [string]$OutFile)
    if ($Uri.EndsWith('/SHA256SUMS')) { Copy-Item $env:TEST_SUMS $OutFile }
    else { Copy-Item $env:TEST_DOWNLOAD $OutFile }
}
& $env:TEST_BOOTSTRAP
'''
        command += [mock]
    else:
        source.write_bytes(b'#!/bin/sh\nprintf verified > "$TEST_MARKER"\n' +
                           (b'exit 9\n' if mode == 'cli_failure' else b''))
        mockbin = tmp_path / 'mockbin'
        mockbin.mkdir()
        curl = mockbin / 'curl'
        curl.write_text('#!/bin/sh\nfor arg do case "$arg" in https://*) url="$arg";; esac; done\n'
                        'while [ "$1" != "-o" ]; do shift; done\n'
                        'case "$url" in */SHA256SUMS) cp "$TEST_SUMS" "$2";; '
                        '*) cp "$TEST_DOWNLOAD" "$2";; esac\n')
        curl.chmod(0o755)
        env['PATH'] = str(mockbin) + os.pathsep + os.defpath
        command = ['sh', str(skill / 'scripts/bootstrap.sh')]
    asset = f'browser-cli-v1.1.15-{target}'
    digest = hashlib.sha256(source.read_bytes()).hexdigest() if mode != 'mismatch' else '0' * 64
    sums.write_text('' if mode == 'missing' else f'{digest}  {asset}\n')
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    expected = {'valid': 0, 'mismatch': 4, 'missing': 3, 'cli_failure': 9}[mode]
    if windows and expected:
        expected = 1
    assert result.returncode == expected, result.stdout + result.stderr
    verified = mode in ('valid', 'cli_failure')
    assert marker.exists() == verified
    assert (skill / 'bin' / ('browser-cli.exe' if windows else 'browser-cli')).exists() == verified
    if mode != 'valid':
        assert 'Installed browser-cli' not in result.stdout
    if mode in ('missing', 'mismatch'):
        assert ('No checksum' if mode == 'missing' else 'SHA-256 mismatch') in result.stderr


@pytest.mark.macos_only
@pytest.mark.parametrize('mode', CASES)
def test_macos_installer(tmp_path, mode):
    _check_installer(tmp_path, mode, 'aarch64-apple-darwin')


@pytest.mark.linux_only
@pytest.mark.parametrize('mode', CASES)
def test_linux_installer(tmp_path, mode):
    _check_installer(tmp_path, mode, 'x86_64-unknown-linux-musl')


@pytest.mark.windows_only
@pytest.mark.parametrize('mode', CASES)
def test_windows_installer(tmp_path, mode):
    _check_installer(tmp_path, mode, 'x86_64-pc-windows-msvc.exe', windows=True)
