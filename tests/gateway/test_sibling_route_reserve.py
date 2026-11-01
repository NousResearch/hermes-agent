"""A host-scoped connection's sibling route follows the profile's current authority."""
from pathlib import Path

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
from tests.gateway.test_ticket_profile_scope import _Transport


@pytest.mark.asyncio
async def test_sibling_route_rebinds_after_the_profile_is_unserved_and_served_again(tmp_path, monkeypatch):
    """The Desktop's shared socket routed ``profile=beta`` once; beta is then removed and hot-served
    again (a new authority). The cached sibling connection must not keep routing to the retired
    authority, which refuses every admission for the rest of the socket's life."""
    from gateway.run_runtime import initialize_gateway_runtime, serve_profile_runtime, unserve_profile_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import authority_for_profile_id, owner_scope
    from gateway.session_controls import AuthorityConnection
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    beta_home = homes[2][1]
    retired, connection, runner = [], None, None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        store = runner.session_ticket_store
        ticket = store.mint(profile_id=str(root.resolve()), subject='uid:desk', purpose='interactive', scope='host')
        grant = store.redeem(ticket, profile_id=None, purpose='interactive')
        launch = authority_for_profile_id(runner, grant['profile_id'])
        identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                    'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                    'native_bootstrap': True, 'profile_scope': 'host'}
        with owner_scope(launch):
            connection = AuthorityConnection(launch, _Transport(), identity, operator=True)

        async def create():
            with owner_scope(launch):
                return await connection.dispatch({'id': 1, 'method': 'session.create', 'params': {
                    'profile': 'beta', 'request_id': f'c{len(retired)}', 'source': 'cli', 'cwd': str(tmp_path), 'model': 'm',
                    'toolsets': []}})
        assert 'result' in await create()
        old = runner.session_authorities.for_home(beta_home)
        assert await unserve_profile_runtime(runner, beta_home) is True
        runner._profile_adapters.pop('beta', None)  # as GatewayRunner._unserve_profile does
        retired.append(old)
        new = await serve_profile_runtime(runner, 'beta', beta_home)
        assert new is not old
        reply = await create()
        assert 'result' in reply, f'the socket still routes beta to its retired authority: {reply}'
        assert reply['result']['session_id'] in new.sessions
    finally:
        if connection is not None:
            with owner_scope(connection.authority):
                await connection.close()
        for authority in [*retired, *(getattr(runner, 'session_authorities', None) or [])]:
            authority.db.close()
        for _, home in homes:
            process_ownership.release(Path(home))
