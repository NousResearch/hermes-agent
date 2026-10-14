"""Desktop's per-boot owner backfill no longer 409s against the live owner (#106742 live sweep row 24).

``POST /api/sessions/owner-backfill`` took the offline maintenance lock, which a running gateway
always holds, so every Desktop boot logged a 409 "Exclusive maintenance refused". With a composed
gateway the idempotent stamp runs on the owner's own store instead.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import pytest

from tests.gateway.test_normal_runtime_boot import control


@pytest.mark.platforms('linux')
def test_owner_backfill_runs_on_the_live_owner(tmp_path):
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = {'model': {'provider': 'custom', 'default': 'm', 'base_url': 'http://127.0.0.1:1/v1'},
             'auxiliary': {'title_generation': {'enabled': False}}}
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': True}, **model}))
    work = home / 'profiles' / 'work'
    work.mkdir(parents=True)
    (work / 'config.yaml').write_text(json.dumps(model))
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED='1')
    with (tmp_path / 'gateway.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline, descriptor = time.monotonic() + 60, {}
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    descriptor = control(home, 'identify')
                    if descriptor.get('state') == 'ready':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            log.seek(0)
            assert descriptor.get('state') == 'ready', (descriptor, log.read())

            def headers(profile_home):
                # The Desktop mints each profile's own native grant (the profile its body names).
                return {'X-Hermes-Gateway-Ticket': control(home, 'session-ticket', {
                    'profile_id': str(profile_home.resolve()), 'instance_id': descriptor['instance_id'],
                    'purpose': 'native-http'})['ticket']}
            with httpx.Client(base_url=descriptor['api_origin'], trust_env=False, timeout=30) as client:
                for body, owner, owner_home in (({}, 'default', home), ({'profile': 'work'}, 'work', work)):
                    response = client.post('/api/sessions/owner-backfill', json=body, headers=headers(owner_home))
                    assert response.status_code == 200, (body, response.text)
                    assert response.json()['profile'] == owner and response.json()['stamped'] == 0
                # A grant still reaches only its own profile.
                assert client.post('/api/sessions/owner-backfill', json={'profile': 'work'},
                                   headers=headers(home)).status_code == 403
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
