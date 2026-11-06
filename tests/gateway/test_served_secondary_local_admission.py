"""A served secondary's LOCAL sessions admit through its own LocalSessionAdapter.

A hosted-room member turn on a served secondary's Bot Chat session is admitted by
``check_api_turn``, which resolves the answering adapter from the session source: the profile's
own ``LOCAL`` entry (``local_adapter_map``). A live local turn builds that adapter on first use
(``restore_local_session``, ``session_detached``) — but the FIRST admission for such a session
(the member turn, before any local turn ever ran for the profile) must already resolve it.
Serving the profile therefore builds the bound adapter; without it the profile with no stored
local receipt yet refused every member turn as ``runtime_draining`` immediately.

Nothing borrows across profile lines: a LOCAL session of a profile this gateway does not serve
refuses as before and the launch profile's ``LOCAL`` slot stays untouched.

Gateway imports stay at module level: they run at collection, before the per-test home-I/O guard
installs; importing inside the test body would trip on the import chain's own (read-only) probes.
"""
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.run_runtime import (initialize_gateway_runtime, serve_profile_runtime,
                                unserve_profile_runtime)
from gateway.runtime_ownership import process_ownership
from gateway.session import SessionSource
from gateway.session_api_turn import check_api_turn
from gateway.session_authority import LiveSession
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError

# A member-turn admission whose settings are self-contained: no store read for ``api_settings``,
# no room policy, so the only behaviour under test is the adapter the turn admits through.
_TURN = {'api_turn_v1': {'history': None, 'settings': {'session_history_delivery': 'x'}}}


def _runner(root, homes):
    """A multiplexing runner with every home reserved; boot serves them per secondary."""
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    return GatewayRunner(config)


def _reserve_homes(tmp_path, monkeypatch, names=('server',)):
    """One fresh root with the default home plus every named profile home."""
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [('default', root)]
    for name in names:
        home = root / 'profiles' / name
        home.mkdir(parents=True, mode=0o700)
        (home / 'config.yaml').write_text('{}')
        homes.append((name, home))
    return root, homes


def _live_local_session(authority, profile, chat_id, sid):
    """A live LOCAL session of *authority*, as a Bot Chat session lives before any turn of its
    own has ever been admitted (the member turn's first admission)."""
    authority.sessions[sid] = LiveSession(
        SessionSource(platform=Platform.LOCAL, chat_id=chat_id, chat_type='dm', user_id='stefan',
                      profile=profile),
        f'agent:{profile or "main"}:local:dm:{chat_id}')
    return SessionRef(authority.profile_id, sid)


@pytest.mark.asyncio
async def test_a_served_secondary_admits_its_bot_chat_member_turn(tmp_path, monkeypatch):
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('server',))
    process_ownership.reserve([home for _, home in homes])
    runner = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        authority = runner.session_authorities.for_home(homes[-1][1])
        assert authority is not None
        adapter = (getattr(runner, '_profile_adapters', None) or {}).get('server', {}).get(Platform.LOCAL)
        assert adapter is not None, 'the served profile takes up with its own LocalSessionAdapter'
        assert adapter.authority is authority and adapter.gateway_runner is runner

        ref = _live_local_session(authority, 'server', 'bot-chat', 'bt')
        assert check_api_turn(authority, ref, _TURN) is adapter
    finally:
        for authority in list(getattr(runner, 'session_authorities', None) or []) if runner else []:
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_a_local_session_of_an_unserved_profile_still_refuses(tmp_path, monkeypatch):
    """Fail-closed: a profile this gateway does not serve has no adapter and no slot; the turn
    refuses as before and nothing borrows the launch profile's LOCAL adapter."""
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('server',))
    process_ownership.reserve([home for _, home in homes])
    runner = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        launch = runner.session_authority
        assert launch is not None

        ref = _live_local_session(launch, 'ghost', 'ghost-chat', 'gc')
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            check_api_turn(launch, ref, _TURN)
        # No slot was minted for the unknown profile and the launch profile's LOCAL adapter was
        # not built (or borrowed) by serving secondaries either.
        assert 'ghost' not in (getattr(runner, '_profile_adapters', None) or {})
        assert Platform.LOCAL not in (getattr(runner, 'adapters', None) or {})
    finally:
        for authority in list(getattr(runner, 'session_authorities', None) or []) if runner else []:
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_reserving_a_served_profile_again_rebinds_its_local_adapter(tmp_path, monkeypatch):
    """A removed profile served again gets a fresh authority and its own adapter bound to it —
    the previous serve's adapter never answers for the new one."""
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('server',))
    server_home = homes[-1][1]
    process_ownership.reserve([home for _, home in homes])
    retired, runner = [], None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        old_authority = runner.session_authorities.for_home(server_home)
        old_adapter = runner._profile_adapters['server'][Platform.LOCAL]
        assert old_adapter.authority is old_authority

        assert await unserve_profile_runtime(runner, server_home) is True
        runner._profile_adapters.pop('server', None)  # as _unserve_profile does
        retired.append(old_authority)
        new_authority = await serve_profile_runtime(runner, 'server', server_home)
        assert new_authority is not old_authority

        adapter = (runner._profile_adapters.get('server') or {}).get(Platform.LOCAL)
        assert adapter is not None and adapter.authority is new_authority and adapter is not old_adapter
        ref = _live_local_session(new_authority, 'server', 'bot-chat', 'bt')
        assert check_api_turn(new_authority, ref, _TURN) is adapter
    finally:
        for authority in [*retired, *(list(getattr(runner, 'session_authorities', None) or []) if runner else [])]:
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
