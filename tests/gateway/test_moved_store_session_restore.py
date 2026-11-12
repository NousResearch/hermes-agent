"""Session receipts of a store that moved with its profile restore, under that profile only.

A ``state.db`` that moved with its profile (another machine, a moved home) keeps receipts whose
``profile_id`` names the old path. Restoring one is that store's own row, but an API receipt's
stored origin still places the turn's runtime scope, so one naming another profile is refused. A
``hermes profile rename`` moves the store AND changes the name, so its identity migration must move
the receipts' origin and route namespace too, API and local receipts alike (#150651, #149384).
"""
import contextlib
import json

import pytest

from hermes_agent.hermes_state_local import API_BINDING_PREFIX
from hermes_agent.hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


def _rehome(authority, session_id, **origin):
    key = API_BINDING_PREFIX + session_id
    receipt = json.loads(authority.db.get_meta(key))
    receipt['profile_id'] = '/departed/host/profiles/alpha'
    receipt['entry']['origin'].update(origin)
    authority.db.set_meta(key, json.dumps(receipt))


def _shutdown(runner):
    """Close every store handle a (possibly half-initialized) runner opened; one failing close
    must not leave the rest open."""
    if runner is None:
        return
    for authority in list(getattr(runner, 'session_authorities', None) or []):
        with contextlib.suppress(Exception):
            authority.db.close()
    with contextlib.suppress(Exception):
        runner.session_store.close_all_db_handles()
    with contextlib.suppress(Exception):
        runner.close_all_session_db_handles()


@contextlib.asynccontextmanager
async def _serving(root, homes, home):
    """A multiplexed runner serving *homes*; yields *home*'s authority, every store closed and every
    home released on exit."""
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    runner = None
    process_ownership.reserve([h for _, h in homes])
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        yield runner.session_authorities.for_home(home)
    finally:
        _shutdown(runner)
        for _, h in homes:
            process_ownership.release(h)


def _rename_alpha(root, homes):
    """What ``rename_profile`` does with no live multiplexer: move ``alpha``'s home to ``gamma``,
    then migrate its identity. Returns the renamed homes and gamma's home."""
    from hermes_cli.profile_identity import _migrate_profile_identity
    renamed = root / 'profiles' / 'gamma'
    homes[1][1].rename(renamed)
    assert _migrate_profile_identity('alpha', 'gamma', False)
    return [homes[0], ('gamma', renamed), homes[2]], renamed


@pytest.mark.asyncio
async def test_moved_api_receipt_restores_only_under_its_own_profile(tmp_path, monkeypatch):
    from gateway.session_api import bind_api_session, restore_api_session
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    async with _serving(root, homes, homes[1][1]) as alpha:
        for sid in ('api-moved', 'api-sibling'):
            alpha.db.create_session(sid, source='api_server')
            bind_api_session(alpha, sid)
        _rehome(alpha, 'api-moved')
        _rehome(alpha, 'api-sibling', profile='beta')
        alpha.sessions.clear()

        restored = restore_api_session(alpha, 'api-moved')
        assert alpha.sessions[restored.session_id].source.profile == 'alpha'
        with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
            restore_api_session(alpha, 'api-sibling')


@pytest.mark.asyncio
async def test_renamed_profile_api_session_restores(tmp_path, monkeypatch):
    from gateway.session_api import bind_api_session, declared_api_session, restore_api_session
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    async with _serving(root, homes, homes[1][1]) as alpha:
        alpha.db.create_session('api-renamed', source='api_server')
        bind_api_session(alpha, 'api-renamed', declared_key='my-convo')

    renamed_homes, renamed = _rename_alpha(root, homes)
    async with _serving(root, renamed_homes, renamed) as gamma:
        gamma.sessions.clear()  # drive restore_api_session itself, whatever boot recovery loaded
        restored = restore_api_session(gamma, 'api-renamed')
        live = gamma.sessions[restored.session_id]
        assert (live.source.profile, live.route) == ('gamma', 'agent:gamma:api_server:dm:api-renamed')
        # The API edge binds again on the next request, and the caller's declared key still resolves.
        bind_api_session(gamma, 'api-renamed', declared_key='my-convo')
        assert declared_api_session(gamma.db, 'my-convo') == 'api-renamed'


@pytest.mark.asyncio
async def test_renamed_profile_local_session_restores(tmp_path, monkeypatch):
    """A local (Desktop / Bot Chat) session of a renamed profile resumes under the new name and
    stays a native chat (its turns keep the routing label, not an adopted row's stored label)."""
    from gateway.session_a2a import storage_source
    from gateway.session_authorities import owner_scope
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    async with _serving(root, homes, homes[1][1]) as alpha:
        actor = SessionActor('owner', alpha.profile_id, frozenset({'session:create', 'session:read'}), 'socket')
        with owner_scope(alpha):
            sid = create_local_session(alpha, actor, {'request_id': 'renamed', 'source': 'cli',
                                                      'cwd': str(tmp_path), 'model': 'frozen',
                                                      'toolsets': []}).session_id

    renamed_homes, renamed = _rename_alpha(root, homes)
    async with _serving(root, renamed_homes, renamed) as gamma:
        gamma.sessions.clear()  # drive restore_local_session itself, whatever boot recovery loaded
        with owner_scope(gamma):
            restored = restore_local_session(gamma, sid)
        live = gamma.sessions[restored.session_id]
        assert (live.source.profile, live.route) == ('gamma', f'agent:gamma:local:dm:{sid}')
        assert storage_source(gamma.db, live.source, sid, 'local') == 'local'
