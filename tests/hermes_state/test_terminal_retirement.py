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
        worker_result = rt.persist_worker_message(db, epoch=epoch, execution_id='worker', session_id='used',
            generation=row['generation'], sequence=1, role='assistant', content='worker history')
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
        assert admission_result(db, row['admission_id']) == result
        from hermes_state_terminal import terminal_worker_receipt
        worker_args = dict(execution_id='worker', session_id='used', generation=row['generation'], sequence=1,
            adoption_secret='owned-proof', payload_digest=rt.admission_fingerprint(canonical_target='used',
                payload={'operation': 'append_text', 'role': 'assistant', 'content': 'worker history'}))
        assert terminal_worker_receipt(db, **worker_args) == worker_result
        with pytest.raises(rt.RuntimeStoreError, match='permission_denied'):
            terminal_worker_receipt(db, **(worker_args | {'adoption_secret': 'foreign'}))
        for change in ({'payload': {'text': 'changed'}}, {'principal_id': 'foreign'}, {'request_id': 'fresh'}):
            with pytest.raises(rt.RuntimeStoreError):
                rt.admit_session_input(db, epoch=epoch, **(args | change))
        with pytest.raises(rt.RuntimeStoreError):
            rt.persist_worker_message(db, epoch=epoch, execution_id='worker', session_id='used',
                generation=row['generation'], sequence=2, role='assistant', content='late')


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
