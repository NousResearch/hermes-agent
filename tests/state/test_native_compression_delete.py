"""Idle native conversation deletion includes its physical compression history."""
import json

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as runtime


def compressed(db, parent, child):
    assert db.try_acquire_compression_lock(parent, 'fixture-metadata')
    try:
        db.publish_compression_child(parent_session_id=parent, child_session_id=child,
            source='telegram', messages=[{'role': 'user', 'content': f'private history for {child}'}],
            compression_lock_holder='fixture-metadata')
    finally:
        db.release_compression_lock(parent, 'fixture-metadata')


def delete(db, epoch, sid):
    row = db.get_session(sid)
    return runtime.mutate_runtime_session(db, epoch=epoch, principal_id='fixture-owner',
        session_id=sid, request_id='delete-once', operation='delete', payload={},
        expected_revision=row['runtime_revision'], expected_generation=row['runtime_generation'])


@pytest.mark.parametrize('depth', [1, 2])
def test_delete_removes_native_continuations_and_their_current_route(tmp_path, depth):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = runtime.begin_runtime_epoch(db, instance_id='fixture-owner')
        db.create_session('root', source='telegram')
        db.append_message('root', 'user', 'old private history')
        chain = ['root']
        for index in range(depth):
            child = f'continuation-{index}'
            compressed(db, chain[-1], child)
            chain.append(child)
        db._execute_write(lambda conn: conn.execute('INSERT INTO gateway_routing VALUES(?,?,?,?)',
            ('', 'native-route', json.dumps({'session_id': chain[-1]}), 1.0)))
        receipt = delete(db, epoch, 'root')
        assert set(receipt['deleted_ids']) == set(chain)
        for sid in chain:
            assert db.get_session(sid) is None
            assert db.get_messages(sid) == []
        assert not db._read_all('SELECT * FROM gateway_routing')


def test_independent_branches_and_noncompression_children_remain(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = runtime.begin_runtime_epoch(db, instance_id='fixture-owner')
        db.create_session('root', source='telegram')
        compressed(db, 'root', 'tip')
        db.create_session('branch', source='gui', parent_session_id='root',
            model_config={'_branched_from': 'root'})
        db.create_session('independent-child', source='gui', parent_session_id='tip')
        for sid in ('branch', 'independent-child'):
            db.append_message(sid, 'user', 'keep this independent history')
            assert db.get_compression_lineage(sid)[0] == sid
        receipt = delete(db, epoch, 'root')
        assert set(receipt['deleted_ids']) == {'root', 'tip'}
        for sid in ('branch', 'independent-child'):
            assert db.get_session(sid)['parent_session_id'] is None
            assert db.get_messages(sid)[0]['content'] == 'keep this independent history'


def test_queued_continuation_blocks_the_whole_delete_without_cancellation(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = runtime.begin_runtime_epoch(db, instance_id='fixture-owner')
        db.create_session('root', source='telegram')
        compressed(db, 'root', 'tip')
        runtime.admit_session_input(db, epoch=epoch, principal_id='fixture-owner',
            session_id='tip', request_id='queued-only', payload={'text': 'not executed'})
        with pytest.raises(runtime.RuntimeStoreError, match='session_busy'):
            delete(db, epoch, 'root')
        assert db.get_session('root') is not None
        assert db.get_session('tip') is not None


def test_noncompression_parent_still_preserves_ordinary_children(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = runtime.begin_runtime_epoch(db, instance_id='fixture-owner')
        db.create_session('root', source='telegram')
        db.create_session('separate', source='gui', parent_session_id='root')
        db.append_message('separate', 'user', 'independent data')
        assert delete(db, epoch, 'root')['deleted_ids'] == ['root']
        assert db.get_session('separate')['parent_session_id'] is None
        assert db.get_messages('separate')[0]['content'] == 'independent data'


def test_delegate_continuations_are_retired(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = runtime.begin_runtime_epoch(db, instance_id='fixture-owner')
        db.create_session('root', source='telegram')
        compressed(db, 'root', 'tip')
        db.create_session('delegate', source='cli', parent_session_id='tip',
            model_config={'_delegate_from': 'tip'})
        compressed(db, 'delegate', 'delegate-tip')
        receipt = delete(db, epoch, 'root')
        assert set(receipt['deleted_ids']) == {'root', 'tip', 'delegate', 'delegate-tip'}


def test_ambiguous_continuation_ownership_refuses_deletion(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = runtime.begin_runtime_epoch(db, instance_id='fixture-owner')
        db.create_session('root', source='telegram')
        compressed(db, 'root', 'tip')
        db.create_session('other-child', source='telegram', parent_session_id='root')
        assert db.get_compression_lineage('other-child')[0] == 'other-child'
        with pytest.raises(runtime.RuntimeStoreError, match='admission_conflict'):
            delete(db, epoch, 'root')
        assert all(db.get_session(sid) is not None for sid in ('root', 'tip', 'other-child'))


def test_compressed_fork_keeps_its_own_continuation_not_its_ancestor(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = runtime.begin_runtime_epoch(db, instance_id='fixture-owner')
        db.create_session('ancestor', source='gui')
        db.create_session('root', source='gui', parent_session_id='ancestor',
            model_config={'_branched_from': 'ancestor'})
        compressed(db, 'root', 'tip')
        assert db.get_compression_lineage('tip')[0] == 'root'
        assert set(delete(db, epoch, 'root')['deleted_ids']) == {'root', 'tip'}
        assert db.get_session('ancestor') is not None
