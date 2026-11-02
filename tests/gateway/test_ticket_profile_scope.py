"""An issued interactive ticket keeps its profile scope across sibling selectors (#106742 W1)."""
import json
from pathlib import Path

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


class _Transport:
    def write(self, frame):
        return True

    def close(self):
        pass


@pytest.mark.asyncio
async def test_profile_bound_ticket_never_reaches_a_sibling_authority(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session import SessionSource
    from gateway.session_authorities import authority_for_profile_id, owner_scope
    from gateway.session_authority import LiveSession
    from gateway.session_controls import AuthorityConnection
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    connections, runner = [], None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        store = runner.session_ticket_store
        alpha_id, beta_id = (str(home.resolve()) for _, home in homes[1:])
        beta = authority_for_profile_id(runner, beta_id)
        beta.db.create_session('beta-private', source='api_server')
        beta.db.append_message('beta-private', 'user', 'PRIVATE_BETA_MARKER')
        beta.sessions['beta-private'] = LiveSession(
            SessionSource(platform=Platform.LOCAL, chat_id='local-x', user_id='uid:other'), 'route')

        def connect(profile_id, **scope):
            # Exactly gateway/run_api.py's redeem → identity → operator connection binding.
            ticket = store.mint(profile_id=profile_id, subject='uid:attacker', purpose='interactive', **scope)
            grant = store.redeem(ticket, profile_id=None, purpose='interactive')
            authority = authority_for_profile_id(runner, grant['profile_id'])
            identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                        'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                        'native_bootstrap': True, 'profile_scope': grant.get('scope', 'profile')}
            with owner_scope(authority):
                connection = AuthorityConnection(authority, _Transport(), identity, operator=True)
            connections.append(connection)
            return connection

        async def call(connection, method, **params):
            with owner_scope(connection.authority):
                return await connection.dispatch({'id': 1, 'method': method, 'params': params})

        alpha, root_ticket = connect(alpha_id), connect(str(root.resolve()))
        for connection in (alpha, root_ticket):
            for method, params in (('session.list', {}), ('session.resume', {'session_id': 'beta-private'}),
                                   ('prompt.submit', {'session_id': 'beta-private', 'submission_id': 'x', 'text': 't'}),
                                   ('session.interrupt', {'session_id': 'beta-private'}),
                                   ('session.mutate', {'session_id': 'beta-private'}), ('session.create', {})):
                reply = await call(connection, method, profile='beta', **params)
                assert reply.get('error', {}).get('message') == 'profile_mismatch', (method, reply)
                assert 'PRIVATE_BETA_MARKER' not in json.dumps(reply)
            assert (await call(connection, 'session.list', profile='gamma'))['error']['message'] == 'profile_mismatch'
        assert 'result' in await call(alpha, 'session.list')
        assert 'result' in await call(alpha, 'session.list', profile='alpha')  # own-profile selector
        assert (await call(alpha, 'session.list', profile='default'))['error']['message'] == 'profile_mismatch'
        # The Desktop's shared-primary socket asks for a host-scoped grant and keeps its sibling route.
        host = connect(str(root.resolve()), scope='host')
        listed = await call(host, 'session.list', profile='beta')
        assert [row['session_id'] for row in listed['result']['sessions']] == ['beta-private'], listed
        assert (await call(host, 'session.list', profile='gamma'))['error']['message'] == 'profile_mismatch'
        with pytest.raises(PermissionError):
            store.mint(profile_id=alpha_id, subject='uid:1', purpose='native-http', scope='host')
    finally:
        for connection in connections:
            with owner_scope(connection.authority):
                await connection.close()
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(Path(home))
