"""Deletion respects inherited lineage and queued continuation metadata."""
import json

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as runtime
from tests.state.test_native_compression_delete import compressed, delete


@pytest.mark.parametrize('marker', ['_branched_from', '_delegate_from'])
def test_actual_inherited_marker_stays_bound_to_original_parent(tmp_path, marker):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = runtime.begin_runtime_epoch(db, instance_id='review')
        db.create_session('ancestor', source='telegram')
        compressed(db, 'ancestor', 'ancestor-tip')
        cfg = {marker: 'ancestor'}
        db.create_session('fork', source='cli', parent_session_id='ancestor', model_config=cfg)
        assert db.try_acquire_compression_lock('fork', 'review-metadata')
        try:
            db.publish_compression_child(parent_session_id='fork', child_session_id='fork-tip',
                source='cli', model_config=cfg, messages=[{'role': 'user', 'content': 'fixture'}],
                compression_lock_holder='review-metadata')
        finally:
            db.release_compression_lock('fork', 'review-metadata')
        assert json.loads(db.get_session('fork-tip')['model_config'])[marker] == 'ancestor'
        assert db.get_compression_lineage('fork-tip') == ['fork', 'fork-tip']
        assert set(delete(db, epoch, 'fork')['deleted_ids']) == {'fork', 'fork-tip'}
        assert db.get_session('ancestor') and db.get_session('ancestor-tip')


@pytest.mark.parametrize('topology', ['inherited-fork', 'unmarked-delegate-tip'])
def test_queued_omitted_tip_prevents_delete_and_preserves_all_metadata(tmp_path, topology):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = runtime.begin_runtime_epoch(db, instance_id='review')
        if topology == 'inherited-fork':
            db.create_session('ancestor', source='telegram')
            cfg = {'_branched_from': 'ancestor'}
            db.create_session('root', source='telegram', parent_session_id='ancestor', model_config=cfg)
            assert db.try_acquire_compression_lock('root', 'review-metadata')
            try:
                db.publish_compression_child(parent_session_id='root', child_session_id='tip',
                    source='telegram', model_config=cfg,
                    messages=[{'role': 'user', 'content': 'private continuation'}],
                    compression_lock_holder='review-metadata')
            finally:
                db.release_compression_lock('root', 'review-metadata')
        else:
            db.create_session('root', source='telegram')
            db.create_session('delegate', source='cli', parent_session_id='root',
                model_config={'_delegate_from': 'root'})
            compressed(db, 'delegate', 'tip')
        db.save_gateway_routing_entry('review-route', json.dumps({'session_id': 'tip'}))
        runtime.admit_session_input(db, epoch=epoch, principal_id='fixture-owner',
            session_id='tip', request_id='queued-only', payload={'text': 'never executed'})
        before = {
            table: [dict(row) for row in db._read_all(f'SELECT * FROM {table}')]
            for table in ('sessions', 'messages', 'gateway_routing', 'session_admissions', 'state_meta')
        }
        receipt = None
        error = None
        try:
            receipt = delete(db, epoch, 'root')
        except runtime.RuntimeStoreError as exc:
            error = str(exc)
        assert error == 'session_busy', {
            'unexpected_receipt': receipt,
            'tip_parent': db.get_session('tip')['parent_session_id'],
            'admissions': [dict(row) for row in db._read_all(
                'SELECT target_session_id,status FROM session_admissions')],
            'routes': db.load_gateway_routing_entries(),
        }
        for table, rows in before.items():
            assert [dict(row) for row in db._read_all(f'SELECT * FROM {table}')] == rows
