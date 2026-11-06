"""Gateway repoints preserve inherited continuation and exact moved sets (#149132)."""
import json
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import SessionLineageError
from hermes_state_local import POLICY_PREFIX


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    profile = tmp_path / 'profiles/science'
    profile.mkdir(parents=True)
    local, sibling = SessionDB(tmp_path / 'state.db'), SessionDB(profile / 'state.db')
    try:
        yield local, sibling
    finally:
        local.close()
        sibling.close()


def _rows(db):
    return {table: [tuple(row) for row in db._conn.execute(f'SELECT * FROM {table} ORDER BY 1')]
            for table in ('sessions', 'messages', 'state_meta', 'gateway_routing')}


def _foreign(db, sid):
    db._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name) VALUES (?,'cli',1,'science')", (sid,))
    db._conn.commit()


def _branch(db, marker):
    for sid, parent, reason, metadata in (
        ('outside', None, 'compression', None),
        ('branch', 'outside', 'compression', {marker: 'outside'}),
        ('middle', 'branch', 'compression', {marker: 'outside'}),
        ('tip', 'middle', 'ws_orphan_reap', {marker: 'outside'}),
    ):
        db._conn.execute("INSERT INTO sessions(id,source,started_at,parent_session_id,end_reason,ended_at,model_config) "
                         "VALUES (?,'cli',1,?,?,2,?)", (sid, parent, reason, json.dumps(metadata) if metadata else None))
    db._conn.commit()


def _peer(db, *, ancestors=True):
    db.record_gateway_session_peer('tip', source='telegram', session_key='agent:main:telegram:dm:fixture',
                                   include_compression_ancestors=ancestors)


@pytest.mark.parametrize('marker', ['_branched_from', '_reset_from', '_delegate_from'])
@pytest.mark.parametrize('contested', [None, 'branch', 'middle', 'tip', 'outside'])
def test_peer_routes_complete_continuation_without_crossing_its_original_boundary(stores, marker, contested):
    local, sibling = stores
    _branch(local, marker)
    assert local.get_compression_lineage('tip') == ['branch', 'middle', 'tip']
    if contested:
        _foreign(sibling, contested)
    before = _rows(local), _rows(sibling)
    outside = dict(local.get_session('outside'))
    if contested in ('branch', 'middle', 'tip'):
        with pytest.raises(SessionLineageError):
            _peer(local)
        assert (_rows(local), _rows(sibling)) == before
    else:
        _peer(local)
        assert {row[0] for row in local._conn.execute("SELECT id FROM sessions WHERE source='telegram'")} == {'branch', 'middle', 'tip'}
        assert local.get_session('outside') == outside
        assert _rows(sibling) == before[1]


@pytest.mark.parametrize('marker', ['_branched_from', '_reset_from', '_delegate_from'])
def test_ordinary_peer_refresh_still_updates_only_the_supplied_segment(stores, marker):
    local, sibling = stores
    _branch(local, marker)
    _foreign(sibling, 'branch')
    kept = {sid: dict(local.get_session(sid)) for sid in ('outside', 'branch', 'middle')}
    before_sibling = _rows(sibling)
    _peer(local, ancestors=False)
    assert {sid: local.get_session(sid) for sid in kept} == kept
    assert local.get_session('tip')['source'] == 'telegram'
    assert _rows(sibling) == before_sibling


@pytest.mark.parametrize('contested', ['root', 'owner'])
def test_source_change_does_not_hand_off_receipts_of_unselected_ancestors(stores, contested):
    local, sibling = stores
    local._conn.execute("INSERT INTO sessions(id,source,started_at,end_reason,ended_at,session_key) "
                         "VALUES ('root','telegram',1,'compression',2,'agent:main:telegram:dm:old')")
    local._conn.execute("INSERT INTO sessions(id,source,started_at,parent_session_id) VALUES ('tip','tool',3,'root')")
    local._conn.execute("INSERT INTO sessions(id,source,started_at,end_reason,ended_at,title) "
                         "VALUES ('owner','cli',0,'session_reset',1,'Keep old title')")
    local._conn.execute('INSERT INTO state_meta(key,value) VALUES (?,?)', (POLICY_PREFIX + 'owner', json.dumps({
        'entry': {'session_id': 'root'}, 'route': 'agent:main:local:dm:owner', 'lineage': ['owner', 'root']})))
    local._conn.commit()
    _foreign(sibling, contested)
    kept = {sid: dict(local.get_session(sid)) for sid in ('root', 'owner')}
    before_receipts = [tuple(row) for row in local._conn.execute('SELECT * FROM state_meta ORDER BY 1')]
    before_sibling = _rows(sibling)
    _peer(local)
    assert local.get_session('tip')['source'] == 'telegram'
    assert {sid: local.get_session(sid) for sid in kept} == kept
    assert [tuple(row) for row in local._conn.execute('SELECT * FROM state_meta ORDER BY 1')] == before_receipts
    assert _rows(sibling) == before_sibling
