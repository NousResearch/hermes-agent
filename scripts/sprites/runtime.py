"""Native Services launcher. NAS delivers the private environment on exec stdin."""
from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile

ROOT = Path('/opt/hermes')
CONFIG = Path('/etc/hermes-sprites/environment.json')


def private_replace(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def environment() -> dict[str, str]:
    env = json.loads(CONFIG.read_text(encoding="utf-8"))
    env.update(HERMES_HOME='/opt/data', HOME='/opt/data',
               PATH=f'{ROOT}/.venv/bin:/usr/local/bin:/usr/bin:/bin:/.sprite/bin',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               HERMES_WEB_DIST=str(ROOT / 'hermes_cli/web_dist'),
               HERMES_TUI_DIR=str(ROOT / 'ui-tui'),
               HERMES_WRITE_SAFE_ROOT='/opt/data', HERMES_DISABLE_LAZY_INSTALLS='1',
               HERMES_LAZY_INSTALL_TARGET='/opt/data/lazy-packages')
    return env


def configure() -> None:
    value = json.load(sys.stdin)
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) and '\0' not in k + v for k, v in value.items()):
        raise ValueError('Invalid runtime environment')
    private_replace(CONFIG, json.dumps(value))
    sys.path.insert(0, str(ROOT))
    from scripts.sprites.runtime_env import sync_environment
    owner = pwd.getpwnam('hermes')
    sync_environment(Path('/opt/data'), value, owner.pw_uid, owner.pw_gid)
    if not Path('/opt/data/config.yaml').exists():
        subprocess.run(['s6-setuidgid', 'hermes', str(ROOT / '.venv/bin/python'), '-c',
                        'from hermes_cli.config import load_config, save_config; save_config(load_config())'],
                       env=environment(), check=True)
    subprocess.run([str(ROOT / 'docker/stage2-hook.sh')], env=environment(), check=True)


def run(service: str, profile: str = 'default') -> None:
    if Path('/etc/hermes-sprites-state/stopped').exists():
        return
    env = environment()
    executable = str(ROOT / '.venv/bin/hermes')
    commands = {
        'dashboard': [executable, 'dashboard', '--host', '0.0.0.0', '--port', '9119', '--no-open'],
        'gateway': [executable, '-p', profile, 'gateway', 'run', '--external-supervisor'],
        'ingress': [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/sprites/ingress.py')],
    }
    command = commands[service]
    owner = pwd.getpwnam('hermes')
    os.initgroups(owner.pw_name, owner.pw_gid)
    os.setgid(owner.pw_gid)
    os.setuid(owner.pw_uid)
    os.chdir('/opt/data')
    os.execve(command[0], command, env)


def activate() -> None:
    # Serializes service registration against NAS's deliberate-stop fence.
    sys.path.insert(0, str(ROOT))
    from hermes_cli import sprites_api
    with open('/etc/hermes-sprites/lifecycle.lock', 'a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if Path('/etc/hermes-sprites-state/stopped').exists():
            return
        existing = {s['name'] for s in sprites_api.request('GET', '/services')}
        for name, command in [('dashboard', 'dashboard'), ('gateway-default', 'gateway'), ('ingress', 'ingress')]:
            if name in existing:
                continue
            definition = {'cmd': '/usr/bin/sudo', 'args': ['-n', '/usr/bin/python3', str(ROOT / 'scripts/sprites/runtime.py'), 'run', command]}
            if name == 'ingress':
                definition['http_port'] = 8080
            sprites_api.request('PUT', '/services/' + name, definition)


if __name__ == '__main__':
    if sys.argv[1] == 'configure':
        configure()
    elif sys.argv[1] == 'activate':
        activate()
    elif sys.argv[1] == 'run':
        run(*sys.argv[2:])
    else:
        raise ValueError('Unknown runtime operation')
