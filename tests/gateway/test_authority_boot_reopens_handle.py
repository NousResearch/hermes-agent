"""The authority boot without a handle takes the one the session store resolves for the active scope, and
refuses to start while that handle cannot open, the failed open recorded for the park reason."""
from types import SimpleNamespace

import pytest


def _runner(home, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setenv('HERMES_HOME', str(home))
    store = SessionStore(home / 'sessions', GatewayConfig())
    return SimpleNamespace(session_store=store, adapters={}, _draining=False, persistence=None,
                           _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                           _evict_cached_agent=lambda route: None)


@pytest.mark.asyncio
async def test_boot_takes_the_stores_handle_for_the_scope(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    runner = _runner(tmp_path, monkeypatch)
    try:
        authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='boot')
        assert authority.persistence is runner.session_store._scoped_persistence()
        assert authority.persistence.runtime.current_epoch() == authority.epoch
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_boot_refuses_while_the_stores_handle_cannot_open(tmp_path, monkeypatch):
    import persistence_assembly
    from gateway.session_authority import initialize_session_authority
    from persistence import StoreUnavailable

    def refuse(*args, **kwargs):
        raise OSError('the interface handle cannot open')
    monkeypatch.setattr(persistence_assembly, 'open_persistence', refuse)
    runner = _runner(tmp_path, monkeypatch)
    try:
        with pytest.raises(StoreUnavailable):
            await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='boot')
        assert 'cannot open' in runner.session_store._db_handle_cache.last_error(tmp_path / 'state.db')
    finally:
        runner.session_store.close_all_db_handles()
