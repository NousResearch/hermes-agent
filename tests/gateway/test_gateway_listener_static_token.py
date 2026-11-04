"""The gateway's own listener, behind the sign-in gate, takes the deployment static token as serve does.

``dashboard.public_url`` gates the gateway's private listener, and ``HERMES_DASHBOARD_SESSION_TOKEN``
registers the bundled session-token provider. A remote client presenting that token as a bearer gets
the dashboard API; a wrong or missing token gets 401. On a multiplexed gateway the same token with a
profile selector reads that profile's own config, not the launch profile's.
"""
import json
from pathlib import Path
import signal
import subprocess
import sys

import httpx
import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env
from tests.gateway.test_normal_runtime_boot import control
from tests.waiting import wait_until

TOKEN = 'deployment-static-token-for-the-gated-listener'


def _identify_ready(home, process) -> dict | None:
    """The ready descriptor, ``{'exited': code}`` once the gateway died, else None (keep waiting)."""
    if process.poll() is not None:
        return {'exited': str(process.returncode)}
    try:
        descriptor = control(home, 'identify')
    except (OSError, ValueError, AssertionError):
        return None
    return descriptor if descriptor.get('state') == 'ready' else None


@pytest.fixture(scope='module')
def gated_gateway(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('gated-gateway')
    home, user = tmp / 'state', tmp / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    beta = home / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    (beta / 'config.yaml').write_text(json.dumps({'model': {'provider': 'custom', 'default': 'beta-model'}}))
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': True},
        'model': {'provider': 'custom', 'default': 'launch-model', 'base_url': 'http://127.0.0.1:1/v1'},
        'auxiliary': {'title_generation': {'enabled': False}},
        'dashboard': {'public_url': 'https://hermes.example.test'},
    }))
    root = Path(__file__).resolve().parents[2]
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED='1', HERMES_DASHBOARD_SESSION_TOKEN=TOKEN)
    with (tmp / 'gateway.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, '-m', 'gateway.run'], cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            descriptor = wait_until(lambda: _identify_ready(home, process), timeout=90,
                                    what='the gateway to report ready')
            log.seek(0)
            assert descriptor.get('state') == 'ready', (descriptor, log.read())
            served = {p['profile_id'] for p in descriptor['served_profiles']}
            assert {str(home.resolve()), str(beta.resolve())} <= served, descriptor
            with httpx.Client(base_url=descriptor['api_origin'], trust_env=False, timeout=30) as client:
                yield client
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@pytest.mark.platforms('linux')
def test_the_gated_gateway_listener_accepts_the_static_token_and_refuses_a_wrong_one(gated_gateway):
    accepted = gated_gateway.get('/api/config', headers={'Authorization': f'Bearer {TOKEN}'})
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()['model'] == 'launch-model'
    for headers in ({'Authorization': 'Bearer not-the-deployment-static-token'}, {}):
        refused = gated_gateway.get('/api/config', headers=headers)
        assert refused.status_code == 401, (headers, refused.text)
        assert 'launch-model' not in refused.text


@pytest.mark.platforms('linux')
def test_the_static_token_with_a_profile_selector_reads_that_profiles_config(gated_gateway):
    beta = gated_gateway.get('/api/config', params={'profile': 'beta'},
                             headers={'Authorization': f'Bearer {TOKEN}'})
    assert beta.status_code == 200, beta.text
    assert beta.json()['model'] == 'beta-model'
    refused = gated_gateway.get('/api/config', params={'profile': 'beta'},
                                headers={'Authorization': 'Bearer not-the-deployment-static-token'})
    assert refused.status_code == 401, refused.text
    assert 'beta-model' not in refused.text
