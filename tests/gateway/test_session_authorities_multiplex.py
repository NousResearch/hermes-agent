"""One session authority per served profile home under gateway.multiplex_profiles."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _reserve_homes(tmp_path, monkeypatch, names=('alpha', 'beta')):
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    # The autouse isolation fixture pins hermes_state.DEFAULT_DB_PATH to its own fake home once
    # hermes_state is imported; per-home stores must follow the scoped HERMES_HOME here.
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [('default', root)]
    for name in names:
        home = root / 'profiles' / name
        home.mkdir(parents=True, mode=0o700)
        (home / 'config.yaml').write_text('{}')
        homes.append((name, home))
    return root, homes


def _runner(root, homes):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    return runner


@pytest.mark.asyncio
async def test_multiplex_builds_one_authority_per_reserved_home(tmp_path, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from hermes_constants import hermes_home_key
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        registry = runner.session_authorities
        assert len(registry) == 3
        # The launch authority is the default profile's; secondaries bind their own stores.
        assert runner.session_authority is registry.launch
        for _name, home in homes:
            authority = registry.for_home(home)
            assert authority is not None
            assert Path(authority.db.db_path).resolve().parent == home.resolve()
            assert (home / 'state.db').exists()
        epochs = {hermes_home_key(h): registry.for_home(h).epoch for _, h in homes}
        assert all(epoch >= 1 for epoch in epochs.values())
        # Descriptor + ticket store carry the whole served set.
        served = {entry['home'] for entry in runner.session_runtime_descriptor['served_profiles']}
        assert served == {str(h.resolve()) for _, h in homes}
        assert runner.session_ticket_store.profile_ids == frozenset(str(h.resolve()) for _, h in homes)
        # A ticket minted for a secondary redeems under the served-set rule and names that home.
        ticket = runner.session_ticket_store.mint(profile_id=str(homes[1][1].resolve()), subject='uid:1', purpose='interactive')
        grant = runner.session_ticket_store.redeem(ticket, profile_id=None, purpose='interactive')
        assert grant['profile_id'] == str(homes[1][1].resolve())
    finally:
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_scoped_lookup_follows_routed_home_and_never_borrows_launch(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import active_authority, authority_for_home
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        alpha = homes[1][1]
        with _profile_runtime_scope(alpha, {}):
            assert active_authority(runner) is authority_for_home(runner, alpha)
            assert active_authority(runner) is not runner.session_authority
        assert active_authority(runner) is runner.session_authority  # unscoped = launch profile
        foreign = tmp_path / 'elsewhere'
        foreign.mkdir()
        with _profile_runtime_scope(foreign, {}):
            assert active_authority(runner) is None
        assert authority_for_home(runner, foreign) is None
    finally:
        for authority in list(runner.session_authorities):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


def test_single_profile_runtime_is_a_map_of_one(tmp_path, monkeypatch):
    """Without multiplex the registry holds only the launch home and every lookup returns it."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_authorities import SessionAuthorities, active_authority
    root, _ = _reserve_homes(tmp_path, monkeypatch, names=())
    registry = SessionAuthorities(root)
    authority = SimpleNamespace(profile_id=str(root), epoch=1)
    registry.add(root, authority, name='default')
    runner = SimpleNamespace(session_authorities=registry, session_authority=authority)
    assert registry.served_profiles() == [{'profile_id': str(root), 'home': str(root)}]
    other = tmp_path / 'other'
    other.mkdir()
    with _profile_runtime_scope(other, {}):
        assert active_authority(runner) is authority
