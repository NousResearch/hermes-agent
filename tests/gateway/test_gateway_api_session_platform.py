"""#134889 T22: a session created over the gateway's own API is platform ``LOCAL`` whether the client
is on this host or remote (Desktop -> remote gateway). The platform is not renamed (stored rows and
shipped client filters read ``LOCAL``); the connection's mode is what tells a remote client apart:
``AuthorityConnection.native_owner`` (a ticket minted over the private control socket), carried
into the frozen policy as ``client_local`` / ``client_env_remote``. Real SessionStore /
SessionAuthority / AuthorityConnection."""
import json

import pytest

from tests.gateway.test_finite_unattended_wire_grant import CAPS
from tests.gateway.test_session_follow_profile_config import _owner


def _connection(authority, user, *, native):
    from gateway.session_controls import AuthorityConnection
    identity = {'user_id': user, 'provider': 'local' if native else 'fixture', 'profile_id': authority.profile_id,
                'instance_id': authority.instance_id, 'capabilities': CAPS}
    if native:
        identity['native_bootstrap'] = True
    return AuthorityConnection(authority, object(), identity)


@pytest.mark.asyncio
async def test_api_created_sessions_are_local_and_the_connection_mode_marks_remote(monkeypatch):
    from gateway import run
    from gateway.config import Platform
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    (home / 'config.yaml').write_text(json.dumps({'model': {'provider': 'custom', 'default': 'm',
                                                            'base_url': 'http://127.0.0.1:1/v1'}}))
    store, _runner, authority, owner = _owner(home)
    try:
        monkeypatch.setattr(authority, '_schedule', lambda ref: None)
        local, remote = _connection(authority, 'here', native=True), _connection(authority, 'away', native=False)
        assert (local.native_owner, remote.native_owner) == (True, False)
        created = {}
        for name, connection in (('local', local), ('remote', remote)):
            reply = await connection.dispatch({'id': name, 'method': 'session.create', 'params': {
                'request_id': name, 'source': 'gui', 'cwd': str(home), 'toolsets': [],
                'client_env': {'PATH': '/usr/bin'}}})
            assert 'result' in reply, reply
            sid = reply['result']['session_id']
            source = authority.sessions[sid].source
            from gateway.session_local_recovery import local_adapter_map
            policy = local_adapter_map(authority)[Platform.LOCAL].policies[source.chat_id]
            created[name] = (source.platform, policy.client_env_remote)
        # Same platform either way; only the connection-derived policy flag differs.
        assert created == {'local': (Platform.LOCAL, False), 'remote': (Platform.LOCAL, True)}
    finally:
        await owner.close()
        store.close_all_db_handles()
