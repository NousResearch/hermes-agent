"""#139650: a LOCAL reset must not require initialize_session_authority to have bound the
store. The fence-epoch lookup read the launch attribute as dict.get()'s default — evaluated
eagerly — so any store built directly (embedders, plugins that run their own SessionStore
against the profile DB) crashed with AttributeError instead of reaching the store's own
semantics: the receipt check, or a successful reset when the epoch registry applies."""
from types import SimpleNamespace

import pytest
import pytest_asyncio


def test_unbound_store_fails_with_a_store_error_not_a_crash(tmp_path, monkeypatch):
    """The issue's minimal repro: a store built directly, no authority ever bound it."""
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore, SessionSource
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig.from_dict({}))
    src = SessionSource(platform=Platform.LOCAL, chat_id='c1', chat_type='dm', user_id='u1')
    entry = store.get_or_create_session(src)
    assert not hasattr(store, '_local_authority_epoch')
    with pytest.raises(RuntimeStoreError) as raised:
        store.reset_session(entry.session_key)
    assert raised.value.reason == 'storage_unavailable'


@pytest_asyncio.fixture
async def bound_local_session(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from hermes_state import SessionDB

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    bound = SessionStore(tmp_path / 'sessions', GatewayConfig())
    bound._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=bound, _session_db=bound._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    owner = Principal('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit',
                                                        'session:control'}), 'native')
    ref = create_local_session(authority, owner, {'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path),
                                                  'model': 'fixture', 'toolsets': []})
    yield SimpleNamespace(bound=bound, authority=authority, ref=ref)
    bound._db.close()


@pytest.mark.asyncio
async def test_registry_epoch_resolves_without_the_launch_attribute(bound_local_session):
    """dict.get() evaluated its default eagerly, so even a store whose per-DB registry held
    the epoch read — and crashed on — a launch attribute it never had. The registry entry
    alone must resolve the fence and the reset must rotate the target."""
    from pathlib import Path

    t = bound_local_session
    entry = t.bound._entries[next(iter(t.bound._entries))]
    with t.bound._db._read_ctx() as conn:
        current = conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()[0]
    # A second store over the same sessions dir and DB, mirroring the live entry, whose only
    # binding is the per-DB epochs registry initialize_session_authority leaves behind.
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    bare = SessionStore(t.bound.sessions_dir, GatewayConfig())
    bare._db = t.bound._db
    bare._entries[entry.session_key] = entry
    bare._local_authority_epochs = {Path(t.bound._db.db_path).resolve(): current}
    new = bare.reset_session(entry.session_key)
    assert new.session_id != t.ref.session_id and new.is_fresh_reset
    row = t.bound._db.get_session(t.ref.session_id)
    assert row is not None and row['end_reason'] == 'session_reset'
