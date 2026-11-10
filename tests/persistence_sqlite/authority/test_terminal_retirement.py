"""Deletion removes history but retains exact, scoped terminal evidence."""
import sqlite3
from types import SimpleNamespace

import pytest
from gateway.session_results import retain_result, admission_result
from persistence.authority.vocabulary import admission_fingerprint
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import (
    admit_session_input, claim_session_input, get_session_admission, recover_session_inputs, settle_session_input,
)
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.authority._runtime_retirement import delete_in_transaction, retired_session
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX, retire_terminal_receipts
from persistence.sqlite.conversation._store.sessions import create_session, delete_session, get_session
from persistence.sqlite.conversation._store.usage import update_token_counts
from persistence.sqlite.workers._executions import register_worker_execution
from persistence.sqlite.workers._receipt import mutate_worker_execution

MEDIA_ROOT = "/nonexistent/native-inputs"


def _mutate(holder, **args):
    return mutate_runtime_session(holder, media_root=MEDIA_ROOT, **args)[0]


def test_used_history_retirement_is_atomic_and_exact(backend, store):
    holder = store.holder()
    create_session(holder, 'used', 'api_server')
    append_message(holder, 'used', 'user', 'physical history')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    args = dict(principal_id='api', session_id='used', request_id='once', payload={'text': 'input'})
    admit_session_input(holder, epoch=epoch, **args)
    row = claim_session_input(holder, epoch=epoch, session_id='used')
    register_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='used',
        generation=row['generation'], kind='compute', adoption_secret='owned-proof')
    worker_append = {'messages': [{'role': 'assistant', 'content': 'worker history'}]}
    worker_result = mutate_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='used',
        generation=row['generation'], sequence=1, operation='transcript.append', payload=worker_append)
    result = {'result': {'final_response': 'exact reply', 'messages': []}, 'usage': {'input_tokens': 7}}
    retain_result(store, epoch=epoch, row=row, result=result)
    snap = get_session(holder, 'used')
    delete = dict(principal_id='human', session_id='used', request_id='delete', operation='delete', payload={},
                  expected_revision=snap['runtime_revision'], expected_generation=snap['runtime_generation'])
    holder._execute_write(lambda c: c.execute("CREATE TRIGGER deny_delete_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.v1.%' BEGIN SELECT RAISE(ABORT,'receipt failed'); END"))
    with pytest.raises(sqlite3.IntegrityError, match='receipt failed'):
        _mutate(holder, epoch=epoch, **delete)
    assert get_session(holder, 'used') is not None
    assert get_session_admission(holder, admission_id=row['admission_id'])['status'] == 'terminal'
    holder._execute_write(lambda c: c.execute('DROP TRIGGER deny_delete_receipt'))
    receipt = _mutate(holder, epoch=epoch, **delete)
    backend.reopen()
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id='restart')
    recover_session_inputs(holder, epoch=epoch)
    assert get_session(holder, 'used') is None and get_messages(holder, 'used') == []
    with holder._read_ctx() as c:
        assert not c.execute('PRAGMA foreign_key_check').fetchall()
        for table in ('session_admissions', 'worker_executions', 'worker_receipts'):
            assert c.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0] == 0
    assert _mutate(holder, epoch=epoch, **delete) == receipt
    replay, _ = admit_session_input(holder, epoch=epoch, **args)
    assert replay['admission_id'] == row['admission_id'] and replay['status'] == 'terminal'
    assert replay['payload'] == {}, 'input history must not be archived in the identity tombstone'
    # Settlement stores this turn's suffix, flagged so no reader re-derives a boundary from it.
    assert admission_result(store, row['admission_id']) == {
        'result': {**result['result'], '_messages_are_turn_suffix': True}, 'usage': result['usage']}
    assert worker_result['count'] == 1
    for change in ({'payload': {'text': 'changed'}}, {'principal_id': 'foreign'}, {'request_id': 'fresh'}):
        with pytest.raises(AuthorityRefusal):
            admit_session_input(holder, epoch=epoch, **(args | change))
    with pytest.raises(AuthorityRefusal):
        mutate_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='used',
            generation=row['generation'], sequence=2, operation='transcript.append', payload=worker_append)


def test_nonterminal_obligations_prevent_any_retirement(store):
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    for status in ('queued', 'unknown', 'started', 'worker'):
        create_session(holder, status, 'api_server')
        if status == 'worker':
            register_worker_execution(holder, epoch=epoch, execution_id=status, session_id=status,
                generation=0, kind='compute', adoption_secret='proof')
        else:
            admit_session_input(holder, epoch=epoch, principal_id='api', session_id=status,
                request_id=status, payload={'text': status})
            if status != 'queued':
                claim_session_input(holder, epoch=epoch, session_id=status)
            if status == 'unknown':
                epoch = begin_runtime_epoch(holder, instance_id='next')
                recover_session_inputs(holder, epoch=epoch)
        reason = 'unknown_execution' if status == 'unknown' else 'session_busy'
        with pytest.raises(AuthorityRefusal, match=reason):
            holder._execute_write(lambda c: delete_in_transaction(holder, c, status, {}, media_root=MEDIA_ROOT))
        assert get_session(holder, status) is not None
        with holder._read_ctx() as c:
            assert not c.execute("SELECT 1 FROM state_meta WHERE key LIKE 'gateway.retired_session.v1.%'").fetchall()


def test_late_accounting_backfill_cannot_resurrect_a_retired_session(store):
    """A delayed background-review usage callback lands after delete committed: the
    retirement fence must refuse the missing-row backfill instead of recreating the
    session beside its durable tombstone."""
    holder = store.holder()
    create_session(holder, 'retired', 'cli')
    append_message(holder, 'retired', 'user', 'history')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    snap = get_session(holder, 'retired')
    delete = dict(principal_id='human', session_id='retired', request_id='delete', operation='delete',
                  payload={}, expected_revision=snap['runtime_revision'],
                  expected_generation=snap['runtime_generation'])
    receipt = _mutate(holder, epoch=epoch, **delete)
    # The real delayed callback: a background-review fork reporting into its parent's store.
    from agent.background_review import _record_review_usage_to_parent
    parent = SimpleNamespace(_session_db=holder, persistence=store, session_id='retired')
    _record_review_usage_to_parent(parent, {'model': 'm', 'provider': 'p', 'base_url': None,
                                            'api_calls': 1, 'input_tokens': 3, 'output_tokens': 1})
    for late in (lambda: update_token_counts(holder, 'retired', input_tokens=5, output_tokens=2, model='m'),
                 lambda: create_session(holder, 'retired', 'unknown')):
        with pytest.raises(AuthorityRefusal, match='not_found'):
            late()
    assert get_session(holder, 'retired') is None, 'late accounting resurrected a deleted session'
    with holder._read_ctx() as c:
        assert c.execute("SELECT COUNT(*) FROM session_model_usage WHERE session_id='retired'").fetchone()[0] == 0
    assert _mutate(holder, epoch=epoch, **delete) == receipt
    # Live sessions keep the legacy missing-row backfill.
    update_token_counts(holder, 'fresh', input_tokens=1, output_tokens=1, model='m')
    assert get_session(holder, 'fresh')['source'] == 'unknown'


def test_deletion_tombstone_keeps_only_the_closing_worker_result(store):
    """Full worker results (history reads) must not outlive the user's delete;
    only the closing receipt stays replayable, earlier digests still detect conflicts."""
    holder = store.holder()
    create_session(holder, 'gone', 'api_server')
    append_message(holder, 'gone', 'user', 'SECRET_HISTORY_LINE')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    register_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='gone',
        generation=0, kind='compute', adoption_secret='proof')
    history = mutate_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='gone',
        generation=0, sequence=1, operation='compression.history', payload={
            'target': 'gone', 'include_ancestors': False, 'include_inactive': False,
            'repair_alternation': False, 'include_row_ids': False, 'include_compacted': False})
    assert 'SECRET_HISTORY_LINE' in str(history)
    mutate_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='gone',
        generation=0, sequence=2, operation='execution.finish', payload={})
    holder._execute_write(lambda c: retire_terminal_receipts(c, ['gone'], media_root=MEDIA_ROOT))
    with holder._read_ctx() as c:
        tombstones = ''.join(v for (v,) in c.execute("SELECT value FROM state_meta WHERE key LIKE 'gateway.terminal_worker.v1.%'"))
    assert 'SECRET_HISTORY_LINE' not in tombstones


def test_deleting_a_session_drops_transcript_copies_from_its_terminal_results(store):
    """Result blobs carry the cumulative ``messages`` of every turn; deletion keeps each
    exact retry's outcome and usage but no copy of the deleted history."""
    holder = store.holder()
    create_session(holder, 'gone', 'api_server')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    history, rows = [], []
    for turn in range(3):
        admit_session_input(holder, epoch=epoch, principal_id='api', session_id='gone',
                            request_id=f'r{turn}', payload={'text': f'SECRET_HISTORY q{turn}'})
        row = claim_session_input(holder, epoch=epoch, session_id='gone')
        append_message(holder, 'gone', 'user', f'SECRET_HISTORY q{turn}')
        history += [{'role': 'user', 'content': f'SECRET_HISTORY q{turn}'}, {'role': 'assistant', 'content': f'a{turn}'}]
        retain_result(store, epoch=epoch, row=row, result={'result': {
            'final_response': f'a{turn}', 'messages': list(history), 'last_reasoning': 'SECRET_HISTORY why',
            'completed': True, 'api_calls': 1, 'input_tokens': 5}, 'usage': {'input_tokens': 5, 'output_tokens': 2}})
        rows.append(row)
        # Each live receipt holds only its own turn's output, never the cumulative history.
        assert admission_result(store, row['admission_id'])['result']['messages'] == [
            {'role': 'assistant', 'content': f'a{turn}'}]
    delete_session(holder, 'gone', media_root=MEDIA_ROOT)
    with holder._read_ctx() as c:
        blobs = ''.join(v for (v,) in c.execute("SELECT value FROM state_meta WHERE key LIKE 'gateway.admission.result.v1.%'"))
    assert 'SECRET_HISTORY' not in blobs, 'deleted history survives in terminal result blobs'
    replay, _ = admit_session_input(holder, epoch=epoch, principal_id='api', session_id='gone', request_id='r2',
                                    payload={'text': 'SECRET_HISTORY q2'})
    assert replay['admission_id'] == rows[2]['admission_id'] and replay['status'] == 'terminal'
    assert admission_result(store, replay['admission_id']) == {'result': {
        'final_response': 'a2', 'messages': [], 'completed': True, 'api_calls': 1, 'input_tokens': 5,
        '_messages_are_turn_suffix': True},
        'usage': {'input_tokens': 5, 'output_tokens': 2}}


@pytest.mark.parametrize('stray', [
    "INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('','other','[]',0)",
    "INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('','other','{bad',0)",
    "INSERT INTO state_meta(key,value) VALUES('gateway.local_policy.v1:other','[]')",
    "INSERT INTO state_meta(key,value) VALUES('gateway.local_policy.v1:other','{\"entry\": []}')",
])
@pytest.mark.parametrize('delete', ['canonical', 'legacy', 'prune'])
def test_a_malformed_unrelated_route_or_policy_row_cannot_abort_a_delete(store, stray, delete):
    """Deletion fences decode every routing entry and creation receipt to
    find the target's; one unrelated malformed row must not roll back an unrelated delete."""
    import time
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    create_session(holder, 'target', 'api_server')
    holder._execute_write(lambda c: c.execute(stray))
    if delete == 'canonical':
        snap = get_session(holder, 'target')
        _mutate(holder, epoch=epoch, principal_id='human', session_id='target', request_id='d',
                operation='delete', payload={}, expected_revision=snap['runtime_revision'],
                expected_generation=snap['runtime_generation'])
    elif delete == 'legacy':
        assert delete_session(holder, 'target', media_root=MEDIA_ROOT)
    else:
        old = time.time() - 100 * 86400
        holder._execute_write(lambda c: c.execute('UPDATE sessions SET started_at=?, ended_at=? WHERE id=?',
                                                  (old, old, 'target')))
        assert store.maintenance.prune_sessions(older_than_days=30, media_root=MEDIA_ROOT).count == 1
    assert get_session(holder, 'target') is None
    assert retired_session(holder, 'target')
    with holder._read_ctx() as c:  # the unrelated row is left exactly as it was
        assert c.execute("SELECT (SELECT COUNT(*) FROM gateway_routing) + (SELECT COUNT(*) FROM state_meta "
                         "WHERE key GLOB 'gateway.local_policy.v1:*')").fetchone()[0] == 1


def test_deleted_history_is_not_recoverable_from_worker_or_mutation_receipts(store):
    """A failed worker is terminalized by settlement without an execution.finish
    receipt, so its last receipt may be a history read; a rewind/compress receipt copies the
    rewound message / summary. Neither may replay the deleted transcript after the delete."""
    import json
    holder = store.holder()
    create_session(holder, 'gone', 'cli')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    append_message(holder, 'gone', 'user', 'first')
    append_message(holder, 'gone', 'assistant', 'a1')
    rewound = append_message(holder, 'gone', 'user', 'SECRET_REWOUND')
    append_message(holder, 'gone', 'assistant', 'a2')
    snap = get_session(holder, 'gone')
    rewind = dict(principal_id='human', session_id='gone', request_id='rw', operation='rewind',
                  payload={'target_message_id': rewound}, expected_revision=snap['runtime_revision'],
                  expected_generation=snap['runtime_generation'])
    receipt = _mutate(holder, epoch=epoch, **rewind)
    assert 'SECRET_REWOUND' in json.dumps(receipt)
    append_message(holder, 'gone', 'user', 'SECRET_HISTORY_LINE')
    admit_session_input(holder, epoch=epoch, principal_id='api', session_id='gone', request_id='r',
                        payload={'text': 'x'})
    row = claim_session_input(holder, epoch=epoch, session_id='gone')
    register_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='gone',
        generation=row['generation'], kind='compute', adoption_secret='proof')
    history = {'target': 'gone', 'include_ancestors': False, 'include_inactive': False,
               'repair_alternation': False, 'include_row_ids': False, 'include_compacted': False}
    mutate_worker_execution(holder, epoch=epoch, execution_id='worker', session_id='gone',
        generation=row['generation'], sequence=1, operation='compression.history', payload=history)
    settle_session_input(holder, epoch=epoch, admission_id=row['admission_id'],
                         generation=row['generation'], outcome='failed')
    snap = get_session(holder, 'gone')
    _mutate(holder, epoch=epoch, principal_id='human', session_id='gone', request_id='d',
            operation='delete', payload={}, expected_revision=snap['runtime_revision'],
            expected_generation=snap['runtime_generation'])
    with holder._read_ctx() as c:
        blobs = ''.join(v for (v,) in c.execute('SELECT value FROM state_meta'))
    assert 'SECRET_REWOUND' not in blobs and 'SECRET_HISTORY_LINE' not in blobs
    replay = _mutate(holder, epoch=epoch, **rewind)
    assert replay == {**receipt, 'target_message': None}


def test_mutation_receipts_naming_a_deleted_physical_target_drop_its_text(store, tmp_path):
    """A local owner's rewind receipt is keyed by the logical id but copies the
    rewound turn from the physical reset child. Deleting that child (``hermes sessions delete`` on
    the dashboard row, which removes the whole conversation) strips it and the user-set title."""
    import json
    from tests.persistence_sqlite.authority.test_target_advance_fence import _local_session
    holder = store.holder()

    def args(sid, operation, request_id, payload):
        snap = get_session(holder, sid)
        return dict(principal_id='human', session_id=sid, request_id=request_id, operation=operation,
                    payload=payload, expected_revision=snap['runtime_revision'],
                    expected_generation=snap['runtime_generation'])

    epoch = begin_runtime_epoch(holder, instance_id='owner')
    sid = _local_session(store, epoch, tmp_path)
    child = _mutate(holder, epoch=epoch, **args(sid, 'reset', 'reset', {}))['target_session_id']
    append_message(holder, child, 'user', 'first')
    append_message(holder, child, 'assistant', 'a1')
    target = append_message(holder, child, 'user', 'SECRET_CHILD_TURN')
    append_message(holder, child, 'assistant', 'a2')
    rewind = args(sid, 'rewind', 'rw', {'target_message_id': target})
    assert 'SECRET_CHILD_TURN' in json.dumps(_mutate(holder, epoch=epoch, **rewind))
    rename = args(sid, 'rename', 'rn', {'title': 'SECRET_TITLE'})
    _mutate(holder, epoch=epoch, **rename)
    # The listed row is the reset child; deleting it deletes the whole conversation.
    assert delete_session(holder, child, media_root=MEDIA_ROOT) and get_session(holder, sid) is None

    def blobs():
        with holder._read_ctx() as c:
            return ''.join(v for (v,) in c.execute('SELECT value FROM state_meta'))
    assert 'SECRET_CHILD_TURN' not in blobs() and 'SECRET_TITLE' not in blobs()
    replay = _mutate(holder, epoch=epoch, **rewind)
    assert replay['target_message'] is None and replay['rewound_count'] == 2
    assert _mutate(holder, epoch=epoch, **rename)['title'] is None


def test_retirement_drops_the_turn_user_row_copy(store):
    """The settled user row as the model saw it is a transcript copy: deletion strips it."""
    holder = store.holder()
    create_session(holder, 'gone', 'api_server')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    admit_session_input(holder, epoch=epoch, principal_id='api', session_id='gone', request_id='r1',
                        payload={'text': 'secret ask'})
    row = claim_session_input(holder, epoch=epoch, session_id='gone')
    user = {'role': 'user', 'content': 'secret ask', 'api_content': 'memory: secret ask'}
    retain_result(store, epoch=epoch, row=row, result={'result': {
        'final_response': 'reply', 'current_turn_user_idx': 0,
        'messages': [user, {'role': 'assistant', 'content': 'reply'}]}})
    assert admission_result(store, row['admission_id'])['result']['_turn_user_row'] == user
    snap = get_session(holder, 'gone')
    _mutate(holder, epoch=epoch, principal_id='human', session_id='gone', request_id='del',
            operation='delete', payload={}, expected_revision=snap['runtime_revision'],
            expected_generation=snap['runtime_generation'])
    retired = admission_result(store, row['admission_id'])['result']
    assert retired['final_response'] == 'reply' and '_turn_user_row' not in retired


def test_tombstones_past_the_retry_horizon_are_reclaimed_by_the_prune_pass(store, monkeypatch):
    """Deletion tombstones keep byte-exact retry replies for the 7-day horizon, then the maintenance
    prune pass reclaims every row they hold but the deletion fence itself (B2-06)."""
    import time
    now = time.time()
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id='owner')

    def settle_and_delete(sid, at):
        create_session(holder, sid, 'api_server')
        args = dict(principal_id='api', session_id=sid, request_id='r', payload={'text': 'ask ' + sid})
        admit_session_input(holder, epoch=epoch, **args)
        row = claim_session_input(holder, epoch=epoch, session_id=sid)
        retain_result(store, epoch=epoch, row=row, result={'result': {'final_response': 'reply ' + sid, 'messages': []}})
        snap = get_session(holder, sid)
        monkeypatch.setattr(time, 'time', lambda: at)  # the delete happened *at*
        _mutate(holder, epoch=epoch, principal_id='human', session_id=sid, request_id='del',
                operation='delete', payload={}, expected_revision=snap['runtime_revision'],
                expected_generation=snap['runtime_generation'])
        monkeypatch.undo()
        return args, row['admission_id']

    old, old_id = settle_and_delete('old', now - 8 * 86400)
    young, young_id = settle_and_delete('young', now - 6 * 86400)

    def rows_naming(sid, admission_id):
        from persistence.sqlite.conversation._store.terminal import ADMISSION_PREFIX, RESULT_PREFIX, identity_key
        keys = (ADMISSION_PREFIX + admission_id, RESULT_PREFIX + admission_id,
                identity_key('api', sid, 'r'), RETIRED_PREFIX + sid)
        return [k for k in keys if store.state_meta.get(k) is not None]
    assert len(rows_naming('old', old_id)) == 4
    store.maintenance.prune_sessions(older_than_days=90, media_root=MEDIA_ROOT)
    assert len(rows_naming('young', young_id)) == 4
    assert rows_naming('old', old_id) == [RETIRED_PREFIX + 'old'], \
        'tombstones past the horizon were not reclaimed (the deletion fence stays)'
    with pytest.raises(AuthorityRefusal, match='not_found'):
        admit_session_input(holder, epoch=epoch, **old)
    replay, _ = admit_session_input(holder, epoch=epoch, **young)
    assert replay['admission_id'] == young_id and replay['status'] == 'terminal'
    assert admission_result(store, young_id)['result']['final_response'] == 'reply young'
