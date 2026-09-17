"""Offline installer checks: checksum failures must never execute a download."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SKILL = Path(__file__).resolve().parents[2] / 'optional-skills/research/lexmount-cloud-browser'


@pytest.mark.macos_only
@pytest.mark.parametrize('mode,expected', [('valid', 0), ('mismatch', 4), ('missing', 3)])
def test_bootstrap_verifies_before_execution(tmp_path, mode, expected):
    skill = tmp_path / 'skill'
    shutil.copytree(SKILL / 'scripts', skill / 'scripts')
    mockbin = tmp_path / 'mockbin'
    mockbin.mkdir()
    payload = b'#!/bin/sh\nprintf verified > "$TEST_MARKER"\n'
    asset = 'browser-cli-v1.1.15-aarch64-apple-darwin'
    source = tmp_path / 'download'
    source.write_bytes(payload)
    sums = tmp_path / 'sums'
    digest = hashlib.sha256(payload).hexdigest() if mode == 'valid' else '0' * 64
    sums.write_text('' if mode == 'missing' else f'{digest}  {asset}\n')
    for name, body in {
        'curl': '#!/bin/sh\nfor arg do case "$arg" in https://*) url="$arg";; esac; done\nwhile [ "$1" != "-o" ]; do shift; done\ncase "$url" in */SHA256SUMS) cp "$TEST_SUMS" "$2";; *) cp "$TEST_DOWNLOAD" "$2";; esac\n',
    }.items():
        path = mockbin / name
        path.write_text(body)
        path.chmod(0o755)
    marker = tmp_path / 'executed'
    # Deliberately exclude user credentials and release override variables.
    env = {'PATH': str(mockbin) + os.pathsep + os.defpath,
           'TEST_MARKER': str(marker), 'TEST_SUMS': str(sums),
           'TEST_DOWNLOAD': str(source)}
    result = subprocess.run(['sh', str(skill / 'scripts/bootstrap.sh')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    assert marker.exists() == (expected == 0)
    assert (skill / 'bin/browser-cli').exists() == (expected == 0)
