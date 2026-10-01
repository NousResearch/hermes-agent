"""Native macOS sshd, actual Desktop SSH exec/forward, and an isolated owner."""
import json
import os
from pathlib import Path
import shlex
import socket
import tempfile
import shutil
import subprocess
import sys
import time

import pytest

pytest_plugins = ['tests.conftest']
pytestmark = pytest.mark.platforms('macos')

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon


def test_desktop_ssh_attaches_to_owner_without_owning_its_lifetime(tmp_path):
    import pwd
    root = Path(__file__).resolve().parents[4]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700); user.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'model': {'provider': 'custom', 'default': 'fixture',
        'base_url': 'http://127.0.0.1:9/v1'}, 'auxiliary': {'title_generation': {'enabled': False}}}))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', PYTHONUNBUFFERED='1')
    # The SSH login shell and configured launcher deliberately have different
    # synthetic homes. No real user's Hermes profile inventory is read.
    ambient = tmp_path / 'ambient-home'
    (ambient / 'profiles' / 'ambient-only').mkdir(parents=True)
    (ambient / 'install_id').write_text('0123456789abcdef0123456789abcdef')
    selected = home / 'profiles' / 'selected-only'
    selected.mkdir(parents=True)
    (selected / 'config.yaml').write_text((home / 'config.yaml').read_text().replace('fixture', 'selected-model'))
    shell = tmp_path / 'ssh-shell'
    shell.write_text('#!/bin/sh\nexport HOME=' + shlex.quote(str(user)) +
        '\nexport HERMES_HOME=' + shlex.quote(str(ambient)) + '\nexec /bin/sh -c "$SSH_ORIGINAL_COMMAND"\n')
    shell.chmod(0o700)
    for name in ('host', 'client'):
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(tmp_path / name)], check=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = tmp_path / 'sshd_config'
    config.write_text(f'Port {port}\nListenAddress 127.0.0.1\nHostKey {tmp_path}/host\nPidFile {tmp_path}/pid\n'
        f'ForceCommand {shell}\nUsePAM no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nAuthorizedKeysFile {tmp_path}/client.pub\nStrictModes no\n')
    known = tmp_path / 'known_hosts'
    known.write_text(f'[127.0.0.1]:{port} ' + (tmp_path / 'host.pub').read_text())
    launcher = tmp_path / 'hermes'
    launcher.write_text('#!/bin/sh\ncd ' + shlex.quote(str(root)) + '\nexec env -i ' +
        ' '.join(shlex.quote(k + '=' + v) for k, v in env.items()) + ' ' + shlex.quote(sys.executable) +
        ' -m hermes_cli.main "$@"\n')
    launcher.chmod(0o700)
    classic = tmp_path / 'classic-hermes'
    classic.write_text("#!/bin/sh\nprintf 'usage: hermes gateway\n  run Run gateway\n  status Show status\n'\n")
    classic.chmod(0o700)
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
                    knownHosts=str(known), controlDir=control_dir, hermes=str(launcher), classicHermes=str(classic), selectedHome=str(selected))
                for _ in range(2):
                    result = subprocess.run(['node', str(fixture)], input=json.dumps(request), text=True,
                        capture_output=True, timeout=60)
                    assert result.returncode == 0, result.stderr + '\n' + (tmp_path / 'sshd.log').read_text()
                    assert json.loads(result.stdout) == {'canonical': True, 'instance_id': desc['instance_id'], 'http': 200, 'selectedInventoryOnly': True}
                    assert owner.poll() is None
        finally:
            server.terminate(); server.wait(timeout=10)
            shutil.rmtree(control_dir)


def test_native_peer_setup_lost_issuance_recovery_and_new_viewer(tmp_path):
    from tests.gateway.test_session_group_peer_daemons import _gateway, _model
    root = Path(__file__).resolve().parents[4]
    home_model, peer_model = _model('HOME_REPLY'), _model('PEER_REPLY')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); api_port = sock.getsockname()[1]
    home, home_env = _gateway(tmp_path, 'home', home_model, root)
    peer, peer_env = _gateway(tmp_path, 'peer', peer_model, root, api_port=api_port)
    fixture = tmp_path / 'room-setup.mjs'
    subprocess.run([str(root / 'node_modules/.bin/esbuild'),
        str(root / 'apps/desktop/electron/room-setup-live-fixture.ts'), '--bundle', '--platform=node', '--format=esm',
        "--banner:js=import { createRequire } from 'node:module'; const require = createRequire(import.meta.url);",
        '--outfile=' + str(fixture)], cwd=root, check=True, capture_output=True)
    try:
        with daemon(root, home, home_env, barrier=False) as (_, home_desc), daemon(root, peer, peer_env, barrier=False) as (_, peer_desc):
            def descriptor(home, desc):
                return {'baseUrl': desc['api_origin'], 'gatewayEndpoint': desc | {'profile_id': str(home)}}
            request = {'home': descriptor(home, home_desc), 'peer': descriptor(peer, peer_desc),
                       'directory': str(tmp_path / 'custody')}
            result = subprocess.run(['node', str(fixture)], input=json.dumps(request), text=True,
                                    capture_output=True, timeout=120)
            assert result.returncode == 0, result.stderr
            assert all(json.loads(result.stdout).values())
            assert len(peer_model.requests) == 1
    finally:
        for model in (home_model, peer_model):
            model.shutdown(); model.server_close()
