"""Inert authority construction keeps Files reclamation inside the owning scope.

No gateway bootstrap, listener, adapter, service runtime or model is started.
"""
from types import SimpleNamespace

import pytest

from gateway import run_input_reclamation
from gateway.run import _profile_runtime_scope
from gateway.run_runtime import _build_profile_authority
from gateway.runtime_ownership import process_ownership
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from gateway.session_cron import unbind_owner
from hermes_constants import get_hermes_home
from hermes_state import SessionDB


@pytest.mark.asyncio
@pytest.mark.parametrize('ingress_ready', [False, True], ids=['pre-ingress', 'hot-serve-refused'])
async def test_scoped_authority_reclaims_before_publication(tmp_path, monkeypatch, ingress_ready):
    from agent import secret_scope
    from gateway.hosted_room_input_reclamation import READY_KEY
    from tools.terminal_scope import terminal_env

    homes = [tmp_path / 'launch', tmp_path / 'secondary']
    for home in homes:
        home.mkdir()
        (home / 'config.yaml').write_text(
            f'terminal:\n  backend: local\n  cwd: {home}\n')
        (home / '.env').write_text(f'BACKEND_REFRESH_MARKER={home.name}\n')
    monkeypatch.setenv('HERMES_HOME', str(homes[0]))
    monkeypatch.setattr(secret_scope, '_MULTIPLEX_ACTIVE', True)
    databases = {home: SessionDB(home / 'state.db') for home in homes}

    class Runner:
        @property
        def _session_db(self):
            return databases[get_hermes_home()]

    runner = Runner()
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner.session_store = SimpleNamespace()
    runner.session_authorities = SessionAuthorities(homes[0])
    runner.session_runtime_descriptor = {
        'instance_id': 'synthetic-owner', 'state': 'ready' if ingress_ready else 'starting'}
    runner._running, runner._draining = ingress_ready, False
    runner.adapters, runner._profile_adapters = {}, {}
    runner.session_api = runner.session_control_server = None
    observations, authorities = [], []
    original = run_input_reclamation.collect_legacy_copies_before_ingress

    def forbidden(*args, **kwargs):
        raise AssertionError('execution is outside this inert construction test')

    def observe(current, authority):
        home = get_hermes_home()
        assert current is runner and authority.db is databases[home]
        assert current.session_authorities.for_home(home) is None
        assert secret_scope.get_secret('BACKEND_REFRESH_MARKER') == home.name
        assert terminal_env('TERMINAL_CWD') == str(home)
        original(current, authority)
        with authority.db._read_ctx() as conn:
            ready = conn.execute('SELECT value FROM state_meta WHERE key=?', (READY_KEY,)).fetchone()
        assert bool(ready) is not ingress_ready
        observations.append(home)

    monkeypatch.setattr(SessionAuthority, '_schedule', forbidden)
    monkeypatch.setattr(run_input_reclamation, 'collect_legacy_copies_before_ingress', observe)
    process_ownership.reserve(homes)
    try:
        with _profile_runtime_scope(homes[0], hydrate_secrets=False):
            for index, home in enumerate(homes):
                authority = await _build_profile_authority(
                    runner, home.name, home, register=index == 0)
                authorities.append(authority)
                assert runner.session_authorities.for_home(home) is authority
                assert get_hermes_home() == homes[0]
                assert secret_scope.get_secret('BACKEND_REFRESH_MARKER') == homes[0].name
                assert terminal_env('TERMINAL_CWD') == str(homes[0])
        assert observations == homes
        assert runner.session_authority is authorities[0]
        assert runner.session_store._local_authority_epochs == {
            home / 'state.db': authority.epoch for home, authority in zip(homes, authorities)}
    finally:
        for authority in authorities:
            unbind_owner(authority)
        for home, database in databases.items():
            database.close()
            process_ownership.release(home)
