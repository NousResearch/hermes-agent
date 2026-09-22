"""Stage and health-check a pinned release before switching a stopped instance.

The test dashboard gets an empty temporary home, no credentials and no gateway.
User data is never copied back or restored. A later code rollback is the same
explicit operation; administrators must consider application schema compatibility.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from runtime import private_replace

ROOT = Path('/etc/hermes-sprites')
STOPPED = Path('/etc/hermes-sprites-state/stopped')


def health_check(release: Path) -> None:
    owner = pwd.getpwnam('hermes')
    with tempfile.TemporaryDirectory(prefix='hermes-upgrade-health-') as home:
        os.chown(home, owner.pw_uid, owner.pw_gid)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        env = {'HOME': home, 'HERMES_HOME': home, 'PATH': f'{release}/.venv/bin:/usr/bin:/bin',
               'HERMES_WEB_DIST': str(release / 'hermes_cli/web_dist'),
               'PYTHONDONTWRITEBYTECODE': '1', 'HERMES_DISABLE_LAZY_INSTALLS': '1'}
        proc = subprocess.Popen(['s6-setuidgid', 'hermes', str(release / '.venv/bin/hermes'),
                                 'dashboard', '--host', '127.0.0.1', '--port', str(port), '--no-open'],
                                env=env, cwd=home, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 60
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    with opener.open(f'http://127.0.0.1:{port}/api/status', timeout=2) as response:
                        status = json.load(response)
                    if status.get('version') and not status.get('gateway_running'):
                        return
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            raise RuntimeError('New release dashboard health check failed')
        finally:
            import signal
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)


def upgrade(revision: str) -> None:
    if not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Invalid revision')
    sys.path.insert(0, '/opt/hermes')
    from hermes_cli import sprites_api
    receipt = ROOT / 'upgrade-result.json'
    done = threading.Event()

    def lease():
        while not done.is_set():
            try:
                sprites_api.request('PUT', '/tasks/hermes-upgrade', {'expire': '90s'})
            except (OSError, RuntimeError):
                pass
            done.wait(15)

    with (ROOT / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Services may restart after failure/cold boot. One NAS request gets one
        # attempt; an interrupted attempt stays inspectable, never auto-retried.
        if receipt.exists():
            return
        previous = Path('/opt/hermes').resolve().name
        private_replace(receipt, json.dumps({'revision': revision, 'previous': previous, 'status': 'building'}))
        thread = threading.Thread(target=lease, daemon=True)
        thread.start()
        try:
            if not STOPPED.exists():
                raise RuntimeError('Stop required')
            active = [s for s in sprites_api.request('GET', '/services')
                      if s['name'] != 'hermes-upgrade' and s.get('state', {}).get('status') == 'running']
            if active:
                raise RuntimeError('Runtime services must be stopped')
            url = f'https://raw.githubusercontent.com/NousResearch/hermes-agent/{revision}/scripts/sprites/install.sh'
            with urllib.request.urlopen(url, timeout=60) as response:
                script = response.read(128 * 1024)
            installer = ROOT / 'upgrade-install.sh'
            private_replace(installer, script.decode())
            subprocess.run(['bash', str(installer), revision, 'stage'], check=True, timeout=900,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            release = Path('/opt/hermes-releases') / revision
            if not (release / '.sprites-ready').is_file():
                raise RuntimeError('Release not ready')
            health_check(release)
            if not STOPPED.exists():
                raise RuntimeError('Stop fence changed')
            link = Path('/opt/hermes.next')
            link.unlink(missing_ok=True)
            link.symlink_to(release)
            link.replace('/opt/hermes')
            private_replace(receipt, json.dumps({'revision': revision, 'previous': previous, 'status': 'ready'}))
        except Exception:
            # No diagnostic bodies or environment values in service logs.
            private_replace(receipt, json.dumps({'revision': revision, 'previous': previous, 'status': 'failed'}))
            raise RuntimeError('Native upgrade failed; inspect its receipt. User data was not restored.') from None
        finally:
            done.set()
            thread.join(timeout=20)
            try:
                sprites_api.request('DELETE', '/tasks/hermes-upgrade')
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    upgrade(sys.argv[1])
