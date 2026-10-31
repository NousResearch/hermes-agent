"""A real gateway owner (``python -m gateway.run``) in a throwaway home, for tests that dial it."""
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def owner_env(home: Path, user: Path, model_url: str) -> dict:
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(ROOT),
               PYTHONUNBUFFERED='1', OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=model_url)
    return env


def write_owner_config(home: Path, model_url: str) -> None:
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': model_url},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))


@contextlib.contextmanager
def running_owner(tmp_path: Path, model_peer, *, prepare=None, extra_env=None):
    """Yield ``(home, descriptor, env)`` once the owner answers ``identify`` as ready. ``prepare(home)``
    runs before boot (seed the store the owner will open); ``extra_env`` is added to the owner's
    environment."""
    from tests.gateway.test_normal_runtime_boot import control
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model_url = f'http://127.0.0.1:{model_peer.server_port}/v1'
    write_owner_config(home, model_url)
    if prepare is not None:
        prepare(home)
    env = {**owner_env(home, user, model_url), **(extra_env or {})}
    with (tmp_path / 'gateway.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            descriptor, deadline = {}, time.monotonic() + 35
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, 'identify')
                    if descriptor.get('state') == 'ready':
                        break
                except (OSError, ValueError, AssertionError):
                    pass
                time.sleep(.1)
            log.flush()
            log.seek(0)
            assert descriptor.get('state') == 'ready', (descriptor, log.read())
            yield home, descriptor, env
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
