"""N18: a local reset lineage is ONE conversation in the listing, the gateway delete and the
legacy store delete (creation id S0 owns policy/FIFO; the transcript moved to reset child S1)."""
from types import SimpleNamespace

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def reset_lineage(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session
    from hermes_state import SessionDB

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    authority._schedule = lambda ref: None
    owner = Principal('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit',
                                                        'session:control'}), 'native')
    ref = create_local_session(authority, owner, {'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path),
                                                  'model': 'fixture', 'toolsets': []})
    authority.db.append_message(ref.session_id, 'user', 'before reset')
    row = authority.db.get_session(ref.session_id)
    reset = await mutate_session(authority, owner, ref, dict(
        session_id=ref.session_id, request_id='reset', expected_revision=row['runtime_revision'],
        expected_generation=row['runtime_generation'], operation='reset', payload={}))
    authority.db.append_message(reset['target_session_id'], 'user', 'after reset')
    yield SimpleNamespace(authority=authority, owner=owner, s0=ref.session_id, s1=reset['target_session_id'])
    store._db.close()


@pytest.mark.asyncio
async def test_reset_lineage_lists_once_and_deleting_the_listed_row_deletes_it(reset_lineage):
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import mutate_session
    t = reset_lineage
    rows = t.authority.db.list_sessions_rich(order_by_last_active=True, limit=50)
    assert [r['id'] for r in rows] == [t.s1], 'listed as two conversations'
    assert {t.s0, t.s1} <= set(rows[0]['_lineage_ids'])
    # Pages are filtered before LIMIT/OFFSET: the superseded segment never fills a later page.
    assert t.authority.db.list_sessions_rich(limit=1, offset=1) == []
    row = t.authority.db.get_session(t.s1)
    result = await mutate_session(t.authority, t.owner, SessionRef('default', t.s1), dict(
        session_id=t.s1, request_id='delete-current', expected_revision=row['runtime_revision'],
        expected_generation=row['runtime_generation'], operation='delete', payload={}))
    assert set(result['deleted_ids']) == {t.s0, t.s1}
    assert t.authority.db.get_session(t.s0) is None and t.s0 not in t.authority.sessions


@pytest.mark.asyncio
async def test_legacy_delete_of_either_segment_retires_the_whole_lineage(reset_lineage):
    from hermes_state_local import POLICY_PREFIX
    t = reset_lineage
    db = t.authority.db
    assert db.delete_session(t.s0, exclude_active_write_guards=True)
    assert db.get_session(t.s1) is None, 'the current segment survived without its owner'
    with db._read_ctx() as conn:
        assert conn.execute('SELECT 1 FROM state_meta WHERE key=?', (POLICY_PREFIX + t.s0,)).fetchone() is None
