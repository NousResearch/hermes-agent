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


def _compacted_session(db):
    """Q1..A2 archived by in-place compaction (still listed with include_compacted), then Q3/A3."""
    db.create_session('s', source='test')
    db.append_messages_batch('s', [{'role': 'user', 'content': 'Q1'}, {'role': 'assistant', 'content': 'A1'},
                                   {'role': 'user', 'content': 'Q2'}, {'role': 'assistant', 'content': 'A2'}])
    db.archive_and_compact('s', [{'role': 'user', 'content': '[CONTEXT SUMMARY] Q1..A2'},
                                 {'role': 'assistant', 'content': 'ok, continuing'}])
    db.append_messages_batch('s', [{'role': 'user', 'content': 'Q3'}, {'role': 'assistant', 'content': 'A3'}])
    shown = {m['content']: m['id'] for m in db.get_messages('s', include_compacted=True)}
    return shown, rt.begin_runtime_epoch(db, instance_id='owner')


def _rewind_to(db, epoch, target, request_id='rewind'):
    return rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='s', request_id=request_id,
        expected_revision=0, expected_generation=0, operation='rewind', payload={'target_message_id': target})


def test_rewind_refuses_a_row_compaction_archived(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        shown, epoch = _compacted_session(db)
        live = ['[CONTEXT SUMMARY] Q1..A2', 'ok, continuing', 'Q3', 'A3']
        assert [m['content'] for m in db.get_messages('s')] == live
        with pytest.raises(rt.RuntimeStoreError, match='invalid_params') as refused:
            _rewind_to(db, epoch, shown['Q2'])
        assert refused.value.field == 'target_message_id'
        assert [m['content'] for m in db.get_messages('s')] == live
        assert db.get_session('s')['runtime_generation'] == 0
        # The live turn after the compaction still rewinds normally.
        receipt = _rewind_to(db, epoch, shown['Q3'], request_id='live')
        assert receipt['rewound_count'] == 2
        assert [m['content'] for m in db.get_messages('s')] == live[:2]


@pytest.mark.parametrize('target', ['missing', 'assistant'])
def test_rewind_refuses_a_target_that_is_not_a_live_user_row(tmp_path, target):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        shown, epoch = _compacted_session(db)
        with pytest.raises(rt.RuntimeStoreError, match='invalid_params'):
            _rewind_to(db, epoch, 99999 if target == 'missing' else shown['A3'])
        assert [m['content'] for m in db.get_messages('s')][-2:] == ['Q3', 'A3']
