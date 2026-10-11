"""Deletion removes history but retains exact, scoped terminal evidence."""
import sqlite3
from types import SimpleNamespace

import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt
from gateway.session_results import retain_result, admission_result


def test_used_history_retirement_is_atomic_and_exact(tmp_path, monkeypatch):
    import hermes_state_mutation_retirement as retirement
    import hermes_state_mutations
    monkeypatch.setattr(hermes_state_mutations, '_delete',
        getattr(retirement, 'delete_in_transaction', hermes_state_mutations._delete))
    path = tmp_path / 'state.db'
    with SessionDB(path) as db:
        db.create_session('used', source='api_server')
        db.append_message('used', 'user', 'physical history')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        args = dict(principal_id='api', session_id='used', request_id='once', payload={'text': 'input'})
        rt.admit_session_input(db, epoch=epoch, **args)
        row = rt.claim_session_input(db, epoch=epoch, session_id='used')
        rt.register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='used',
            generation=row['generation'], kind='compute', adoption_secret='owned-proof')
        worker_append = {'messages': [{'role': 'assistant', 'content': 'worker history'}]}
        worker_result = rt.mutate_worker_execution(db, epoch=epoch, execution_id='worker', session_id='used',
            generation=row['generation'], sequence=1, operation='transcript.append', payload=worker_append)
        result = {'result': {'final_response': 'exact reply', 'messages': []}, 'usage': {'input_tokens': 7}}
        retain_result(db, epoch=epoch, row=row, result=result)
        snap = db.get_session('used')
        delete = dict(principal_id='human', session_id='used', request_id='delete', operation='delete', payload={},
                      expected_revision=snap['runtime_revision'], expected_generation=snap['runtime_generation'])
        db._execute_write(lambda c: c.execute("CREATE TRIGGER deny_delete_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.v1.%' BEGIN SELECT RAISE(ABORT,'receipt failed'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='receipt failed'):
            rt.mutate_runtime_session(db, epoch=epoch, **delete)
        assert db.get_session('used') is not None
        assert rt.get_session_admission(db, admission_id=row['admission_id'])['status'] == 'terminal'
        db._execute_write(lambda c: c.execute('DROP TRIGGER deny_delete_receipt'))
        receipt = rt.mutate_runtime_session(db, epoch=epoch, **delete)
    with SessionDB(path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='restart')
        rt.recover_session_inputs(db, epoch=epoch)
        assert db.get_session('used') is None and db.get_messages('used') == []
        with db._read_ctx() as c:
            assert not c.execute('PRAGMA foreign_key_check').fetchall()
            for table in ('session_admissions', 'worker_executions', 'worker_receipts'):
                assert c.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0] == 0
        assert rt.mutate_runtime_session(db, epoch=epoch, **delete) == receipt
        replay = rt.admit_session_input(db, epoch=epoch, **args)
        assert replay['admission_id'] == row['admission_id'] and replay['status'] == 'terminal'
        assert replay['payload'] == {}, 'input history must not be archived in the identity tombstone'
        # Settlement stores this turn's suffix, flagged so no reader re-derives a boundary from it.
        assert admission_result(db, row['admission_id']) == {
            'result': {**result['result'], '_messages_are_turn_suffix': True}, 'usage': result['usage']}
        from hermes_state_terminal import terminal_worker_receipt
        worker_args = dict(execution_id='worker', session_id='used', generation=row['generation'], sequence=1,
            adoption_secret='owned-proof', payload_digest=rt.admission_fingerprint(canonical_target='used',
                payload={'operation': 'transcript.append', 'payload': worker_append}))
        # The worker was terminalized by settlement, never by an execution.finish receipt: its
        # last result is not a closing receipt, so it is redacted like every earlier one (W5).
        assert worker_result['count'] == 1
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            terminal_worker_receipt(db, **worker_args)
        with pytest.raises(rt.RuntimeStoreError, match='permission_denied'):
            terminal_worker_receipt(db, **(worker_args | {'adoption_secret': 'foreign'}))
        for change in ({'payload': {'text': 'changed'}}, {'principal_id': 'foreign'}, {'request_id': 'fresh'}):
            with pytest.raises(rt.RuntimeStoreError):
                rt.admit_session_input(db, epoch=epoch, **(args | change))
        with pytest.raises(rt.RuntimeStoreError):
            rt.mutate_worker_execution(db, epoch=epoch, execution_id='worker', session_id='used',
                generation=row['generation'], sequence=2, operation='transcript.append', payload=worker_append)


def test_nonterminal_obligations_prevent_any_retirement(tmp_path):
    import hermes_state_mutation_retirement as retirement
    import hermes_state_mutations
    delete_in_transaction = getattr(retirement, 'delete_in_transaction', hermes_state_mutations._delete)
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        for status in ('queued', 'unknown', 'started', 'worker'):
            db.create_session(status, source='api_server')
            if status == 'worker':
                rt.register_worker_execution(db, epoch=epoch, execution_id=status, session_id=status,
                    generation=0, kind='compute', adoption_secret='proof')
            else:
                rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id=status,
                    request_id=status, payload={'text': status})
                if status != 'queued':
                    rt.claim_session_input(db, epoch=epoch, session_id=status)
                if status == 'unknown':
                    epoch = rt.begin_runtime_epoch(db, instance_id='next')
                    rt.recover_session_inputs(db, epoch=epoch)
            reason = 'unknown_execution' if status == 'unknown' else 'session_busy'
            with pytest.raises(rt.RuntimeStoreError, match=reason):
                db._execute_write(lambda c: delete_in_transaction(db, c, status, {}))
            assert db.get_session(status) is not None
            with db._read_ctx() as c:
                assert not c.execute("SELECT 1 FROM state_meta WHERE key LIKE 'gateway.retired_session.v1.%'").fetchall()


def test_late_accounting_backfill_cannot_resurrect_a_retired_session(tmp_path):
    """A delayed background-review usage callback lands after delete committed: the
    retirement fence must refuse the missing-row backfill instead of recreating the
    session beside its durable tombstone."""
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('retired', source='cli')
        db.append_message('retired', 'user', 'history')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        snap = db.get_session('retired')
        delete = dict(principal_id='human', session_id='retired', request_id='delete', operation='delete',
                      payload={}, expected_revision=snap['runtime_revision'],
                      expected_generation=snap['runtime_generation'])
        receipt = rt.mutate_runtime_session(db, epoch=epoch, **delete)
        # The real delayed callback: a background-review fork reporting into its parent.
        from agent.background_review import _record_review_usage_to_parent
        parent = SimpleNamespace(_session_db=db, session_id='retired')
        _record_review_usage_to_parent(parent, {'model': 'm', 'provider': 'p', 'base_url': None,
                                                'api_calls': 1, 'input_tokens': 3, 'output_tokens': 1})
        for late in (lambda: db.update_token_counts('retired', input_tokens=5, output_tokens=2, model='m'),
                     lambda: db.ensure_session('retired', source='unknown')):
            with pytest.raises(rt.RuntimeStoreError, match='not_found'):
                late()
        assert db.get_session('retired') is None, 'late accounting resurrected a deleted session'
        with db._read_ctx() as c:
            assert c.execute("SELECT COUNT(*) FROM session_model_usage WHERE session_id='retired'").fetchone()[0] == 0
        assert rt.mutate_runtime_session(db, epoch=epoch, **delete) == receipt
        # Live sessions keep the legacy missing-row backfill.
        db.update_token_counts('fresh', input_tokens=1, output_tokens=1, model='m')
        assert db.get_session('fresh')['source'] == 'unknown'


def test_deletion_tombstone_keeps_only_the_closing_worker_result(tmp_path):
    """Full worker results (history reads) must not outlive the user's delete;
    only the closing receipt stays replayable, earlier digests still detect conflicts."""
    import hermes_state_mutation_retirement as retirement
    from hermes_state_terminal import terminal_worker_receipt
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('gone', source='api_server')
        db.append_message('gone', 'user', 'SECRET_HISTORY_LINE')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        rt.register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='gone',
            generation=0, kind='compute', adoption_secret='proof')
        history = rt.mutate_worker_execution(db, epoch=epoch, execution_id='worker', session_id='gone',
            generation=0, sequence=1, operation='compression.history', payload={
                'target': 'gone', 'include_ancestors': False, 'include_inactive': False,
                'repair_alternation': False, 'include_row_ids': False, 'include_compacted': False})
        assert 'SECRET_HISTORY_LINE' in str(history)
        closing = rt.mutate_worker_execution(db, epoch=epoch, execution_id='worker', session_id='gone',
            generation=0, sequence=2, operation='execution.finish', payload={})
        db._execute_write(lambda c: retirement.retire_terminal_receipts(c, ['gone']))
        with db._read_ctx() as c:
            tombstones = ''.join(v for (v,) in c.execute("SELECT value FROM state_meta WHERE key LIKE 'gateway.terminal_worker.v1.%'"))
        assert 'SECRET_HISTORY_LINE' not in tombstones
        digest = lambda op, payload: rt.admission_fingerprint(canonical_target='gone', payload={'operation': op, 'payload': payload})
        args = dict(execution_id='worker', session_id='gone', generation=0, adoption_secret='proof')
        assert terminal_worker_receipt(db, sequence=2, payload_digest=digest('execution.finish', {}), **args) == closing
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            terminal_worker_receipt(db, sequence=1, payload_digest=digest('execution.finish', {}), **args)


def test_deleting_a_session_drops_transcript_copies_from_its_terminal_results(tmp_path):
    """R11: result blobs carry the cumulative ``messages`` of every turn; deletion keeps each
    exact retry's outcome and usage but no copy of the deleted history."""
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('gone', source='api_server')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        history, rows = [], []
        for turn in range(3):
            rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='gone',
                                   request_id=f'r{turn}', payload={'text': f'SECRET_HISTORY q{turn}'})
            row = rt.claim_session_input(db, epoch=epoch, session_id='gone')
            db.append_message('gone', 'user', f'SECRET_HISTORY q{turn}')
            history += [{'role': 'user', 'content': f'SECRET_HISTORY q{turn}'}, {'role': 'assistant', 'content': f'a{turn}'}]
            retain_result(db, epoch=epoch, row=row, result={'result': {
                'final_response': f'a{turn}', 'messages': list(history), 'last_reasoning': 'SECRET_HISTORY why',
                'completed': True, 'api_calls': 1, 'input_tokens': 5}, 'usage': {'input_tokens': 5, 'output_tokens': 2}})
            rows.append(row)
            # Each live receipt holds only its own turn's output, never the cumulative history.
            assert admission_result(db, row['admission_id'])['result']['messages'] == [
                {'role': 'assistant', 'content': f'a{turn}'}]
        db.delete_session('gone')
        with db._read_ctx() as c:
            blobs = ''.join(v for (v,) in c.execute("SELECT value FROM state_meta WHERE key LIKE 'gateway.admission.result.v1.%'"))
        assert 'SECRET_HISTORY' not in blobs, 'deleted history survives in terminal result blobs'
        replay = rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='gone', request_id='r2',
                                        payload={'text': 'SECRET_HISTORY q2'})
        assert replay['admission_id'] == rows[2]['admission_id'] and replay['status'] == 'terminal'
        assert admission_result(db, replay['admission_id']) == {'result': {
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
def test_a_malformed_unrelated_route_or_policy_row_cannot_abort_a_delete(tmp_path, stray, delete):
    """W15 / routing-array: deletion fences decode every routing entry and creation receipt to
    find the target's; one unrelated malformed row must not roll back an unrelated delete."""
    import time
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.create_session('target', source='api_server')
        db._execute_write(lambda c: c.execute(stray))
        if delete == 'canonical':
            snap = db.get_session('target')
            rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='target', request_id='d',
                operation='delete', payload={}, expected_revision=snap['runtime_revision'],
                expected_generation=snap['runtime_generation'])
        elif delete == 'legacy':
            assert db.delete_session('target')
        else:
            old = time.time() - 100 * 86400
            db._execute_write(lambda c: c.execute('UPDATE sessions SET started_at=?, ended_at=? WHERE id=?',
                                                  (old, old, 'target')))
            assert db.prune_sessions(older_than_days=30) == 1
        assert db.get_session('target') is None
        from hermes_state_mutation_retirement import retired_session
        assert retired_session(db, 'target')
        with db._read_ctx() as c:  # the unrelated row is left exactly as it was
            assert c.execute("SELECT (SELECT COUNT(*) FROM gateway_routing) + (SELECT COUNT(*) FROM state_meta "
                             "WHERE key GLOB 'gateway.local_policy.v1:*')").fetchone()[0] == 1


def test_deleted_history_is_not_recoverable_from_worker_or_mutation_receipts(tmp_path):
    """W5 / pastels 2.3: a failed worker is terminalized by settlement without an execution.finish
    receipt, so its last receipt may be a history read; a rewind/compress receipt copies the
    rewound message / summary. Neither may replay the deleted transcript after the delete."""
    import json
    from hermes_state_terminal import terminal_worker_receipt
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('gone', source='cli')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.append_message('gone', 'user', 'first')
        db.append_message('gone', 'assistant', 'a1')
        rewound = db.append_message('gone', 'user', 'SECRET_REWOUND')
        db.append_message('gone', 'assistant', 'a2')
        snap = db.get_session('gone')
        rewind = dict(principal_id='human', session_id='gone', request_id='rw', operation='rewind',
                      payload={'target_message_id': rewound}, expected_revision=snap['runtime_revision'],
                      expected_generation=snap['runtime_generation'])
        receipt = rt.mutate_runtime_session(db, epoch=epoch, **rewind)
        assert 'SECRET_REWOUND' in json.dumps(receipt)
        db.append_message('gone', 'user', 'SECRET_HISTORY_LINE')
        rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='gone', request_id='r',
                               payload={'text': 'x'})
        row = rt.claim_session_input(db, epoch=epoch, session_id='gone')
        rt.register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='gone',
            generation=row['generation'], kind='compute', adoption_secret='proof')
        history = {'target': 'gone', 'include_ancestors': False, 'include_inactive': False,
                   'repair_alternation': False, 'include_row_ids': False, 'include_compacted': False}
        rt.mutate_worker_execution(db, epoch=epoch, execution_id='worker', session_id='gone',
            generation=row['generation'], sequence=1, operation='compression.history', payload=history)
        rt.settle_session_input(db, epoch=epoch, admission_id=row['admission_id'],
                                generation=row['generation'], outcome='failed')
        snap = db.get_session('gone')
        rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='gone', request_id='d',
            operation='delete', payload={}, expected_revision=snap['runtime_revision'],
            expected_generation=snap['runtime_generation'])
        with db._read_ctx() as c:
            blobs = ''.join(v for (v,) in c.execute('SELECT value FROM state_meta'))
        assert 'SECRET_REWOUND' not in blobs and 'SECRET_HISTORY_LINE' not in blobs
        replay = rt.mutate_runtime_session(db, epoch=epoch, **rewind)
        assert replay == {**receipt, 'target_message': None}
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            terminal_worker_receipt(db, execution_id='worker', session_id='gone', generation=row['generation'],
                sequence=1, adoption_secret='proof', payload_digest=rt.admission_fingerprint(
                    canonical_target='gone', payload={'operation': 'compression.history', 'payload': history}))


def test_mutation_receipts_naming_a_deleted_physical_target_drop_its_text(tmp_path):
    """D7 / pastels M07: a local owner's rewind receipt is keyed by the logical id but copies the
    rewound turn from the physical reset child. Deleting that child (``hermes sessions delete`` on
    the dashboard row, which removes the whole conversation) strips it and the user-set title."""
    import json
    from tests.hermes_state.test_target_advance_fence import _local_session

    def args(db, sid, operation, request_id, payload):
        snap = db.get_session(sid)
        return dict(principal_id='human', session_id=sid, request_id=request_id, operation=operation,
                    payload=payload, expected_revision=snap['runtime_revision'],
                    expected_generation=snap['runtime_generation'])

    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        sid = _local_session(db, epoch, tmp_path)
        child = rt.mutate_runtime_session(db, epoch=epoch, **args(db, sid, 'reset', 'reset', {}))['target_session_id']
        db.append_message(child, 'user', 'first')
        db.append_message(child, 'assistant', 'a1')
        target = db.append_message(child, 'user', 'SECRET_CHILD_TURN')
        db.append_message(child, 'assistant', 'a2')
        rewind = args(db, sid, 'rewind', 'rw', {'target_message_id': target})
        assert 'SECRET_CHILD_TURN' in json.dumps(rt.mutate_runtime_session(db, epoch=epoch, **rewind))
        rename = args(db, sid, 'rename', 'rn', {'title': 'SECRET_TITLE'})
        rt.mutate_runtime_session(db, epoch=epoch, **rename)
        # The listed row is the reset child; deleting it deletes the whole conversation.
        assert db.delete_session(child) and db.get_session(sid) is None

        def blobs():
            with db._read_ctx() as c:
                return ''.join(v for (v,) in c.execute('SELECT value FROM state_meta'))
        assert 'SECRET_CHILD_TURN' not in blobs() and 'SECRET_TITLE' not in blobs()
        replay = rt.mutate_runtime_session(db, epoch=epoch, **rewind)
        assert replay['target_message'] is None and replay['rewound_count'] == 2
        assert rt.mutate_runtime_session(db, epoch=epoch, **rename)['title'] is None
