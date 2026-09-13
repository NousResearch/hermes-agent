"""A secondary-profile native ticket binds every implicit profile selector to its own home."""
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


@pytest.mark.linux_only
def test_secondary_native_ticket_implicit_selectors_touch_only_the_secondary_home(tmp_path):
    """R2: omitted / empty / ``current`` selectors used to pass ``_own_profile`` unbound, so the
    route resolved through the LAUNCH home and a valid secondary ticket read and wrote the launch
    profile's config. One read route and one write route, no explicit selector, must resolve
    against the ticket's profile and leave the launch home byte-identical."""
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    beta = home / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    (beta / 'config.yaml').write_text(json.dumps({'model': {'provider': 'custom', 'default': 'beta-model'}}))
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': True},
        'model': {'provider': 'custom', 'default': 'launch-model', 'base_url': 'http://127.0.0.1:1/v1'},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    launch_before = (home / 'config.yaml').read_bytes()
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED='1', HERMES_DASHBOARD_SESSION_TOKEN='normal-http-owner')
    with (tmp_path / 'gateway.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            descriptor = {}
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
            served = {p['profile_id'] for p in descriptor['served_profiles']}
            assert str(beta.resolve()) in served and str(home.resolve()) in served, descriptor

            def ticket():
                return control(home, 'session-ticket', {'profile_id': str(beta.resolve()),
                    'instance_id': descriptor['instance_id'], 'purpose': 'native-http'})['ticket']

            with httpx.Client(base_url=descriptor['api_origin'], trust_env=False, timeout=30) as client:
                for params in ({}, {'profile': ''}, {'profile': 'current'}):
                    response = client.get('/api/config', params=params,
                                          headers={'X-Hermes-Gateway-Ticket': ticket()})
                    assert response.status_code == 200, (params, response.text)
                    assert 'beta-model' in response.text and 'launch-model' not in response.text, (params, response.text)
                for body in ({}, {'profile': 'current'}):
                    updated = client.put('/api/config', json={**body, 'config': {'display': {'skin': 'ares'}}},
                                         headers={'X-Hermes-Gateway-Ticket': ticket()})
                    assert updated.status_code == 200, (body, updated.text)
            import yaml
            assert yaml.safe_load((beta / 'config.yaml').read_text())['display']['skin'] == 'ares'
            assert (home / 'config.yaml').read_bytes() == launch_before
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
