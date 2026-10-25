"""A host-scoped attach while another profile is still starting.

Hot-serve reserves the profile's registry slot before its authority exists. A shared
socket (Desktop, gateway-hosted Webapp) must still open a session on a profile that is
already running, and the attach to the profile being started must wait for that start
and then succeed.
"""
import asyncio
from pathlib import Path

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
from tests.gateway.test_ticket_profile_scope import _Transport


@pytest.mark.asyncio
async def test_host_attach_waits_while_another_profile_is_starting(tmp_path, monkeypatch):
    from gateway import session_authority
    from gateway.run_runtime import initialize_gateway_runtime, serve_profile_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import authority_for_profile_id, owner_scope
    from gateway.session_controls import AuthorityConnection

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    alpha_home = homes[1][1]
    beta_home = root / 'profiles' / 'beta'
    beta_home.mkdir(parents=True, mode=0o700)
    (beta_home / 'config.yaml').write_text('{}')
    process_ownership.reserve([*(home for _, home in homes), beta_home])
    entered, release = asyncio.Event(), asyncio.Event()
    connection = runner = None
    serve_task = beta_task = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        real_initialize = session_authority.initialize_session_authority

        async def hold_initialize(*args, **kwargs):
            entered.set()
            await release.wait()
            return await real_initialize(*args, **kwargs)

        monkeypatch.setattr(session_authority, 'initialize_session_authority', hold_initialize)
        serve_task = asyncio.create_task(serve_profile_runtime(runner, 'beta', beta_home))
        await asyncio.wait_for(entered.wait(), timeout=30)
        assert beta_home in runner.session_authorities
        assert runner.session_authorities.for_home(beta_home) is None

        store = runner.session_ticket_store
        ticket = store.mint(profile_id=str(root.resolve()), subject='uid:desk', purpose='interactive', scope='host')
        grant = store.redeem(ticket, profile_id=None, purpose='interactive')
        launch = authority_for_profile_id(runner, grant['profile_id'])
        identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                    'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                    'native_bootstrap': True, 'profile_scope': 'host'}
        with owner_scope(launch):
            connection = AuthorityConnection(launch, _Transport(), identity, operator=True)

        async def create(profile, request_id):
            with owner_scope(launch):
                return await connection.dispatch({'id': request_id, 'method': 'session.create', 'params': {
                    'profile': profile, 'request_id': request_id, 'source': 'cli', 'cwd': str(tmp_path),
                    'model': 'm', 'toolsets': []}})

        alpha = await create('alpha', 'alpha-while-beta-starts')
        assert 'result' in alpha, alpha
        alpha_authority = runner.session_authorities.for_home(alpha_home)
        assert alpha['result']['session_id'] in alpha_authority.sessions

        beta_task = asyncio.create_task(create('beta', 'beta-while-starting'))
        await asyncio.sleep(0)  # one turn: the attach either waits or raises
        if beta_task.done():
            exc = beta_task.exception()
            raise AssertionError(f'attach while beta is still starting finished early: {exc!r}')

        release.set()
        served, beta = await asyncio.gather(serve_task, beta_task)
        assert 'result' in beta, beta
        assert beta['result']['session_id'] in served.sessions
        assert alpha['result']['session_id'] in alpha_authority.sessions
    finally:
        release.set()
        for task in (serve_task, beta_task):
            if task is None:
                continue
            if not task.done():
                await asyncio.wait([task], timeout=30)
            if task.done() and not task.cancelled():
                task.exception()
        if connection is not None:
            with owner_scope(connection.authority):
                await connection.close()
        for authority in list(getattr(runner, 'session_authorities', None) or []):
            db = getattr(authority, 'db', None)
            if db is not None:
                db.close()
        for home in [*(path for _, path in homes), beta_home]:
            process_ownership.release(Path(home))
