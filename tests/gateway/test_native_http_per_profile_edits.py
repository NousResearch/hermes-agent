"""Old local rows take the Desktop's sidebar edits over each profile's own native grant (#106742).

main's Desktop pinned, archived and renamed old chats in every profile. With per-profile native
grants the Bots sweep and the sidebar PATCH each row with a grant minted for the profile its body
names, but the REST actor was a shared ``'http'`` transport, never a native legacy transport, so
``require_history_claim`` refused every unclaimed pre-upgrade row (403 permission_denied), and
``authorize_history`` did not count ``desktop``/``oneshot`` rows as local history at all.
"""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import httpx
import pytest
from websockets.asyncio.client import connect

from tests.gateway.test_normal_runtime_boot import control


def _seed(home, rows):
    """Rows as the pre-authority clients wrote them: no route, no owner binding."""
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / 'state.db')
    try:
        for sid, source, user_id in rows:
            db.create_session(sid, source=source, user_id=user_id)
            db.append_message(sid, 'user', 'OLD_' + sid)
            db.append_message(sid, 'assistant', 'REPLY_' + sid)
    finally:
        db.close()


@pytest.mark.platforms('linux')
def test_per_profile_grants_edit_old_local_rows_of_their_own_profile(tmp_path):
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = {'model': {'provider': 'custom', 'default': 'm', 'base_url': 'http://127.0.0.1:1/v1'},
             'auxiliary': {'title_generation': {'enabled': False}}}
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': True}, **model}))
    beta = home / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    (beta / 'config.yaml').write_text(json.dumps(model))
    _seed(home, [('old-desktop-root', 'desktop', None), ('old-tui-root', 'tui', None)])
    _seed(beta, [('old-desktop-beta', 'desktop', None), ('old-oneshot-beta', 'oneshot', None),
                 ('old-desktop-nous', 'desktop', 'nous:account'), ('old-desktop-foreign', 'desktop', 'telegram:42')])
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
            homes = {'default': str(home.resolve()), 'beta': str(beta.resolve())}

            def ticket(profile, purpose='native-http'):
                return control(home, 'session-ticket', {'profile_id': homes[profile], 'purpose': purpose,
                                                       'instance_id': descriptor['instance_id']})['ticket']

            def grant(profile):
                return {'X-Hermes-Gateway-Ticket': ticket(profile)}

            with httpx.Client(base_url=descriptor['api_origin'], trust_env=False, timeout=30) as client:
                def pin(sid, profile, minted_for):
                    snap = client.get(f'/api/sessions/{sid}/mutation-snapshot', params={'profile': profile},
                                      headers=grant(minted_for))
                    if snap.status_code != 200:
                        return snap.status_code
                    return client.patch(f'/api/sessions/{sid}', headers=grant(minted_for), json={
                        'profile': profile, 'pinned': True, 'request_id': uuid.uuid4().hex,
                        'expected_revision': snap.json()['runtime_revision']}).status_code

                # Each old local row, in each profile, through a grant for that profile (403 before).
                for sid, profile in (('old-desktop-root', 'default'), ('old-tui-root', 'default'),
                                     ('old-desktop-beta', 'beta'), ('old-oneshot-beta', 'beta'),
                                     ('old-desktop-nous', 'beta')):
                    assert pin(sid, profile, profile) == 200, (sid, profile)
                # The grant still reaches only its own profile, and a foreign identity stays refused.
                assert pin('old-desktop-beta', 'beta', 'default') == 403
                assert pin('old-desktop-foreign', 'beta', 'beta') == 403

            async def stored_picker():
                async with connect(descriptor['api_origin'].replace('http:', 'ws:') + '/api/ws', subprotocols=[
                        'hermes-gateway-v1', 'hermes-gateway-ticket.' + ticket('default', 'interactive')]) as ws:
                    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'session.list',
                                              'params': {'scope': 'stored', 'limit': 50}}))
                    while (reply := json.loads(await asyncio.wait_for(ws.recv(), 30))).get('id') != 1:
                        pass
                    return reply
            listed = asyncio.run(stored_picker())
            # The resume picker offers old Desktop chats like main's listing did (tui only before).
            assert {s['session_id'] for s in listed['result']['sessions']} == {
                'old-desktop-root', 'old-tui-root'}, listed
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
