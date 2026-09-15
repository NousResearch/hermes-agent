"""Termux packaging refuses identity mistakes before modifying its payload."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

from tests.ci.test_desktop_release_tag_admission import _BASH, _GIT, _child_env, _git, _seed_repo


ROOT = Path(__file__).resolve().parents[2]


def test_deb_identity_refusal_leaves_payload_and_output_untouched(tmp_path):
    _, repo = _seed_repo(tmp_path)
    commit = _git('rev-parse', 'HEAD', cwd=repo)
    payload = tmp_path / 'payload'
    (payload / 'app').mkdir(parents=True)
    (payload / 'app/pyproject.toml').write_bytes((repo / 'pyproject.toml').read_bytes())
    (payload / 'venv').mkdir()
    witness = payload / 'venv/keep'
    witness.write_bytes(b'prior dependency tree')
    out = tmp_path / 'output'
    helper = tmp_path / 'bin'
    helper.mkdir()
    boundary = tmp_path / 'native-boundary'
    for name in ('docker', 'dpkg-deb', 'jq'):
        tool = helper / name
        tool.write_text(
            f'#!/bin/sh\nprintf %s {shlex.quote(name)} > {shlex.quote(str(boundary))}\nexit 87\n',
            encoding='utf-8', newline='\n')
        tool.chmod(0o755)
    python = helper / 'python3'
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n',
                      encoding='utf-8', newline='\n')
    python.chmod(0o755)
    env = _child_env(HERMES_PAYLOAD_TAG='', HERMES_BUILD_COMMIT='', GIT_ALLOW_PROTOCOL='file',
                     PYTHONUTF8='1')
    for name in ('MSYS_NO_PATHCONV', 'MSYS2_ARG_CONV_EXCL', 'GIT_DIR', 'GIT_WORK_TREE', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    env['PATH'] = os.pathsep.join([str(helper), str(Path(_GIT).parent), env.get('PATH', '')])
    args = ['--repo', str(repo), '--payload', str(payload), '--out', str(out)]

    def invoke(identity):
        result = subprocess.run([_BASH, str(ROOT / 'scripts/termux/build_deb.sh'), *args, *identity],
                                cwd=tmp_path, env=env, capture_output=True, text=True,
                                encoding='utf-8', timeout=30)
        assert result.returncode != 0, result.stdout + result.stderr
        assert not boundary.exists(), 'identity refusal reached the native builder'
        assert witness.read_bytes() == b'prior dependency tree'
        assert not out.exists(), result.stdout + result.stderr
        return result.stdout + result.stderr

    assert 'not the requested commit' in invoke(['--commit', 'a' * 40])
    assert 'exact full' in invoke(['--commit', 'short'])
    assert 'usage:' in invoke(['--tag', 'v0.1.2', '--commit', commit])
    # The version in the archived payload must agree with the admitted commit.
    (payload / 'app/pyproject.toml').write_text('[project]\nversion="9.9.9"\n', encoding='utf-8')
    assert 'version' in invoke(['--commit', commit]).lower()
    (payload / 'app/pyproject.toml').write_bytes((repo / 'pyproject.toml').read_bytes())
    assert '--tui-product is required' in invoke(['--commit', commit])
