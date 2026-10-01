"""Native macOS sshd, actual Desktop SSH exec/forward, and an isolated owner."""
import json
import os
from pathlib import Path
import pwd
import shlex
import socket
import tempfile
import shutil
import subprocess
import sys
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon


@pytest.mark.platforms('macos')
def test_desktop_ssh_attaches_to_owner_without_owning_its_lifetime(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700); user.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'model': {'provider': 'custom', 'default': 'fixture',
        'base_url': 'http://127.0.0.1:9/v1'}, 'auxiliary': {'title_generation': {'enabled': False}}}))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', PYTHONUNBUFFERED='1')
    for name in ('host', 'client'):
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(tmp_path / name)], check=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = tmp_path / 'sshd_config'
    config.write_text(f'Port {port}\nListenAddress 127.0.0.1\nHostKey {tmp_path}/host\nPidFile {tmp_path}/pid\n'
        f'UsePAM no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nAuthorizedKeysFile {tmp_path}/client.pub\nStrictModes no\n')
    known = tmp_path / 'known_hosts'
    known.write_text(f'[127.0.0.1]:{port} ' + (tmp_path / 'host.pub').read_text())
    launcher = tmp_path / 'hermes'
    launcher.write_text('#!/bin/sh\ncd ' + shlex.quote(str(root)) + '\nexec env -i ' +
        ' '.join(shlex.quote(k + '=' + v) for k, v in env.items()) + ' ' + shlex.quote(sys.executable) +
        ' -m hermes_cli.main "$@"\n')
    launcher.chmod(0o700)
    fixture = tmp_path / 'desktop-ssh.mjs'
    subprocess.run([str(root / 'node_modules/.bin/esbuild'),
        str(root / 'apps/desktop/electron/ssh-gateway-live-fixture.ts'), '--bundle', '--platform=node', '--format=esm',
        "--banner:js=import { createRequire } from 'node:module'; const require = createRequire(import.meta.url);",
        '--external:electron', '--outfile=' + str(fixture)], cwd=root, check=True, capture_output=True)
    control_dir = tempfile.mkdtemp(prefix='hss-', dir='/tmp')
    with (tmp_path / 'sshd.log').open('w+') as log:
        server = subprocess.Popen(['/usr/sbin/sshd', '-D', '-e', '-f', str(config)], stdout=log, stderr=log)
        try:
            time.sleep(.5)
            assert server.poll() is None
            with daemon(root, home, env, barrier=False) as (owner, desc):
                request = dict(user=pwd.getpwuid(os.getuid()).pw_name, port=port, key=str(tmp_path / 'client'),
                    knownHosts=str(known), controlDir=control_dir, hermes=str(launcher))
                for _ in range(2):
                    result = subprocess.run(['node', str(fixture)], input=json.dumps(request), text=True,
                        capture_output=True, timeout=60)
                    assert result.returncode == 0, result.stderr + '\n' + (tmp_path / 'sshd.log').read_text()
                    assert json.loads(result.stdout) == {'canonical': True, 'instance_id': desc['instance_id'], 'http': 200}
                    assert owner.poll() is None
        finally:
            server.terminate(); server.wait(timeout=10)
            shutil.rmtree(control_dir)
