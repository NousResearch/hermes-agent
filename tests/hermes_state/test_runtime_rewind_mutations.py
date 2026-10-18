"""Rewind fences the generation and preserves archived history."""
import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt


def test_rewind_has_existing_semantics_with_atomic_retry(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.append_messages_batch('s', [{'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'reply'}, {'role': 'user', 'content': 'again'}])
        target = db.get_messages('s')[-1]['id']
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='rewind',
            expected_revision=0, expected_generation=0, operation='rewind', payload={'target_message_id': target})
        receipt = rt.mutate_runtime_session(db, **args)
        assert receipt['rewound_count'] == 1 and receipt['target_message']['content'] == 'again'
        assert [m['content'] for m in db.get_messages('s')] == ['first', 'reply']
        assert db.message_count('s') == 3
        assert db.get_session('s')['runtime_generation'] == 1
        assert rt.mutate_runtime_session(db, **args) == receipt
        assert db.get_session('s')['rewind_count'] == 1


def test_rewind_refuses_queued_work(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        db.append_message('s', role='user', content='keep')
        target = db.get_messages('s')[0]['id']
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={'text': 'queued'})
        with pytest.raises(rt.RuntimeStoreError, match='session_busy'):
            rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='s', request_id='rewind',
                expected_revision=0, expected_generation=0, operation='rewind', payload={'target_message_id': target})
        assert db.get_messages('s')[0]['content'] == 'keep'
