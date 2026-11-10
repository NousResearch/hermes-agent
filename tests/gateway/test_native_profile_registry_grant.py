"""Desktop rename/delete of a served profile over a native grant (#134889 DB-5, #140345, #140791).

A multiplexing gateway refused ``PATCH``/``DELETE /api/profiles/<name>`` under every native grant,
even one minted for that very profile, so the Desktop Profiles screen could never rename or delete a
profile while a gateway ran. Rename/delete now ride a dedicated ``native-profile-registry`` grant
(the twin of ``native-profile-create``): minted for the named profile's own home, redeemable only
for ``PATCH``/``DELETE`` of that one profile, never for the gateway's launch profile's home move,
and never for any other route. An ordinary ``native-http`` grant keeps refusing both.
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
def test_registry_grant_renames_and_deletes_only_its_own_served_profile(tmp_path):
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = {'model': {'provider': 'custom', 'default': 'm', 'base_url': 'http://127.0.0.1:1/v1'},
             'auxiliary': {'title_generation': {'enabled': False}}}
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': True}, **model}))
    for name in ('beta', 'gamma', 'keep'):
        (home / 'profiles' / name).mkdir(parents=True)
        (home / 'profiles' / name / 'config.yaml').write_text(json.dumps(model))
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
                    if descriptor.get('state') == 'ready' and len(descriptor.get('served_profiles', [])) == 4:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            log.seek(0)
            assert descriptor.get('state') == 'ready', (descriptor, log.read())

            def home_of(profile):
                return str(home.resolve() if profile == 'default' else (home / 'profiles' / profile).resolve())

            def grant(profile, purpose):
                return {'X-Hermes-Gateway-Ticket': control(home, 'session-ticket', {
                    'profile_id': home_of(profile), 'purpose': purpose,
                    'instance_id': control(home, 'identify')['instance_id']})['ticket']}

            def ready():
                # A rename/delete rescans the served set; wait for the descriptor to settle so the
                # next grant is not refused as "a profile being hot-served".
                d = {}
                for _ in range(200):
                    d = control(home, 'identify')
                    if d.get('state') == 'ready':
                        return d
                    time.sleep(.05)
                return d

            with httpx.Client(base_url=descriptor['api_origin'], trust_env=False, timeout=60) as client:
                # The Desktop's request shape before this fix: a native-http grant for the profile the
                # path names. It stays refused (a profile REST grant is not a registry operation).
                for method, body in (('DELETE', None), ('PATCH', {'new_name': 'renamed'})):
                    response = client.request(method, '/api/profiles/beta', json=body,
                                              headers=grant('beta', 'native-http'))
                    assert response.status_code == 403, (method, response.text)
                assert (home / 'profiles' / 'beta').is_dir()

                registry = 'native-profile-registry'
                # Bound to the profile it was minted for: beta's grant cannot touch gamma.
                response = client.delete('/api/profiles/gamma', headers=grant('beta', registry))
                assert response.status_code == 403, response.text
                assert (home / 'profiles' / 'gamma').is_dir()
                # Bound to the two registry writes: no read, no create, no SOUL write, no import.
                for method, path, body in (('GET', '/api/config', None), ('GET', '/api/profiles/beta/soul', None),
                                           ('PUT', '/api/profiles/beta/soul', {'content': 'x'}),
                                           ('POST', '/api/profiles', {'name': 'sneaky'}),
                                           ('POST', '/api/profiles/import', {'archive': 'x'})):
                    response = client.request(method, path, json=body, headers=grant('beta', registry))
                    assert response.status_code == 401, (method, path, response.text)
                assert not (home / 'profiles' / 'sneaky').exists()

                # The fix: a registry grant for the named profile renames and deletes it live.
                response = client.patch('/api/profiles/gamma', json={'new_name': 'delta'},
                                        headers=grant('gamma', registry))
                assert response.status_code == 200, response.text
                assert response.json()['name'] == 'delta'
                assert not (home / 'profiles' / 'gamma').exists()
                assert (home / 'profiles' / 'delta' / 'config.yaml').is_file()
                ready()
                response = client.delete('/api/profiles/beta', headers=grant('beta', registry))
                assert response.status_code == 200, response.text
                assert not (home / 'profiles' / 'beta').exists()
                ready()
                assert client.delete('/api/profiles/keep', headers=grant('keep', registry)).status_code == 200
                # The launch (default) home never moves: rename is its presentation-only display name.
                served = ready()
                assert home_of('default') in {p['profile_id'] for p in served['served_profiles']}
                once = grant('default', registry)
                response = client.patch('/api/profiles/default', json={'new_name': 'Main'}, headers=once)
                assert response.status_code == 200, response.text
                assert (response.json()['name'], response.json()['display_name']) == ('default', 'Main')
                # Single use, against a profile that stays served (a display-name rename never moves
                # the default home): the same ticket's identical second PATCH is refused, while a
                # fresh ticket's identical PATCH succeeds, so only the reuse explains the 401.
                ready()
                assert client.patch('/api/profiles/default', json={'new_name': 'Main'},
                                    headers=once).status_code == 401
                response = client.patch('/api/profiles/default', json={'new_name': 'Main'},
                                        headers=grant('default', registry))
                assert response.status_code == 200, response.text
                response = client.delete('/api/profiles/default', headers=grant('default', registry))
                assert response.status_code == 403, response.text
                assert (home / 'config.yaml').is_file()
                # The surviving profile still answers its ordinary grant (the gateway kept serving).
                ready()
                assert client.get('/api/profiles/delta/soul',
                                  headers=grant('delta', 'native-http')).status_code == 200
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
