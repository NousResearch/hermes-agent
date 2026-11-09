"""An API session receipt minted under an older home of its store restores, under its own profile only.

A ``state.db`` that moved with its profile (another machine, a moved home) keeps API receipts whose
``profile_id`` names the old path. Restoring one is that store's own row, but the receipt's stored
origin still places the turn's runtime scope, so one naming another profile is refused. A
``hermes profile rename`` moves the store AND changes the name, so its identity migration must move
the receipt's origin and route namespace too (#150651).
"""
import contextlib
import json

import pytest

from hermes_state_local import API_BINDING_PREFIX
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


def _rehome(authority, session_id, **origin):
    key = API_BINDING_PREFIX + session_id
    receipt = json.loads(authority.db.get_meta(key))
    receipt['profile_id'] = '/departed/host/profiles/alpha'
    receipt['entry']['origin'].update(origin)
    authority.db.set_meta(key, json.dumps(receipt))


@pytest.mark.asyncio
async def test_moved_api_receipt_restores_only_under_its_own_profile(tmp_path, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_api import bind_api_session, restore_api_session
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    runner = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        alpha = runner.session_authorities.for_home(homes[1][1])
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
    finally:
        _shutdown(runner)
        for _, home in homes:
            process_ownership.release(home)


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


@pytest.mark.asyncio
async def test_renamed_profile_api_session_restores(tmp_path, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_api import bind_api_session, declared_api_session, restore_api_session
    from hermes_cli.profile_identity import _migrate_profile_identity
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    renamed = root / 'profiles' / 'gamma'
    renamed_homes = [homes[0], ('gamma', renamed), homes[2]]
    runner = None
    process_ownership.reserve([home for _, home in homes])
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        alpha = runner.session_authorities.for_home(homes[1][1])
        alpha.db.create_session('api-renamed', source='api_server')
        bind_api_session(alpha, 'api-renamed', declared_key='my-convo')
    finally:
        _shutdown(runner)
        for _, home in homes:
            process_ownership.release(home)

    # What ``rename_profile`` does with no live multiplexer: move the home, then migrate identity.
    homes[1][1].rename(renamed)
    assert _migrate_profile_identity('alpha', 'gamma', False)

    runner = None
    process_ownership.reserve([home for _, home in renamed_homes])
    try:
        runner = _runner(root, renamed_homes)
        await initialize_gateway_runtime(runner)
        gamma = runner.session_authorities.for_home(renamed)
        gamma.sessions.clear()  # drive restore_api_session itself, whatever boot recovery loaded
        restored = restore_api_session(gamma, 'api-renamed')
        live = gamma.sessions[restored.session_id]
        assert (live.source.profile, live.route) == ('gamma', 'agent:gamma:api_server:dm:api-renamed')
        # The API edge binds again on the next request, and the caller's declared key still resolves.
        bind_api_session(gamma, 'api-renamed', declared_key='my-convo')
        assert declared_api_session(gamma.db, 'my-convo') == 'api-renamed'
    finally:
        _shutdown(runner)
        for _, home in renamed_homes:
            process_ownership.release(home)
