"""B1-17: an unattended (auto-approving) finite turn is ``hermes -z`` only. Main set HERMES_YOLO_MODE inside
the one-shot's own process; over the wire only the local bootstrap principal on its own CLI session may
ask for it. Every other submitter (dashboard identity, operator attach, a Desktop/TUI/Bot Chat session)
gets invalid_params and nothing is admitted. Real SessionStore/SessionAuthority/AuthorityConnection."""
import json

import pytest

from tests.gateway.test_session_follow_profile_config import _owner

CAPS = ['session:create', 'session:read', 'session:submit', 'session:control', 'session:approve', 'session:respond']


def _native(authority, user):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, object(), {
        'user_id': user, 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'capabilities': CAPS, 'native_bootstrap': True})


async def _rpc(connection, method, **params):
    return await connection.dispatch({'id': method, 'method': method, 'params': params})


@pytest.mark.asyncio
async def test_only_the_local_cli_owner_may_submit_an_unattended_turn(monkeypatch):
    from gateway import run
    from gateway.session_controls import AuthorityConnection
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    (home / 'config.yaml').write_text(json.dumps({'model': {'provider': 'custom', 'default': 'm',
                                                            'base_url': 'http://127.0.0.1:1/v1'}}))
    store, _runner, authority, dashboard = _owner(home)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    cli_owner = _native(authority, 'me')
    operator = AuthorityConnection(authority, object(), {'user_id': 'verified'}, operator=True)
    try:
        sessions = {}
        for name, connection, source in (('cli', cli_owner, 'oneshot'), ('gui', cli_owner, 'gui'),
                                         ('dashboard', dashboard, 'cli')):
            created = await _rpc(connection, 'session.create', request_id=name, source=source, cwd=str(home),
                                 toolsets=[])
            assert 'result' in created, created
            sessions[name] = created['result']['session_id']
        await _rpc(operator, 'session.resume', session_id=sessions['cli'])

        def submit(connection, name, request_id):
            return _rpc(connection, 'prompt.submit', session_id=sessions[name], input_id=request_id,
                        text='rm -rf', finite=True, unattended=True)
        accepted = await submit(cli_owner, 'cli', 'own-oneshot')
        assert accepted['result']['status'] == 'queued', accepted
        for connection, name, request_id in ((cli_owner, 'gui', 'desktop-window'),
                                             (dashboard, 'dashboard', 'dashboard-identity'),
                                             (operator, 'cli', 'operator-attach')):
            refused = await submit(connection, name, request_id)
            assert refused.get('error', {}).get('message') == 'invalid_params', (request_id, refused)
            assert not [row for row in list_session_admissions(authority.db, session_id=sessions[name])
                        if row['request_id'] == request_id]
        # A plain finite turn (``-q``) stays available to every submitter.
        assert 'result' in await _rpc(operator, 'prompt.submit', session_id=sessions['cli'], input_id='q',
                                      text='hi', finite=True)
    finally:
        for connection in (cli_owner, operator, dashboard):
            await connection.close()
        store.close_all_db_handles()
