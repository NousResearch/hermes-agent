"""Continuing an adopted pre-authority session keeps the label main wrote on its row (T20).

Every turn of a local session refreshes its routing peer (``SessionStore._record_gateway_session_peer``),
which rewrites ``sessions.source`` through ``session_a2a.storage_source``. For an adopted legacy row that
must stay main's own label: ``hermes sessions list --source``, ``-c`` and the stale-open sweeps key on it.
"""
from types import SimpleNamespace

import pytest
from persistence_assembly import close_stores_under
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session


@pytest.mark.asyncio
@pytest.mark.parametrize('label, title', [('cli', 'old cli'), ('tui', 'old tui'), ('desktop', 'Bot Chat'),
                                          ('gui', 'old gui')])
async def test_a_turn_on_an_adopted_legacy_row_keeps_its_source(tmp_path, monkeypatch, label, title):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local_title import resolve_titled_session

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store.persistence
    seed_session(db, 'old', label, title=title, model='old', cwd=str(tmp_path))
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first')
    native = SessionActor('native', str(tmp_path), frozenset({'session:read', 'session:create'}), 'socket')
    authority._native_legacy_transports = {native.transport_id: native}
    try:
        ref = resolve_titled_session(authority, native, title)
        assert session_row(db, 'old')['source'] == label, 'adoption itself keeps the label'
        live = authority.sessions[ref.session_id]
        store._record_gateway_session_peer('old', live.route, live.source)  # what every turn does
        row = session_row(db, 'old')
        assert (row['source'], row['title']) == (label, title)
    finally:
        close_stores_under(db.scope.profile_dir)
