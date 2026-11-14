"""A settled worker execution and a settled turn keep a retry cache, not history.

`worker_receipts` is keyed by execution and `gateway.admission.result.v1.*` by admission, so
pruning sessions only reclaimed the rows its own retirement fence writes. These rows outlived
every prune and refilled: a store that prunes sessions still grew (measured 149,564 receipts /
48.5 MB on one profile). The reclaim is bounded rather than open-ended, and an exact retry still
resolves inside the horizon.
"""
import json
import time
from contextlib import closing

import pytest

from hermes_state import SessionDB
from hermes_state_local import POLICY_PREFIX
from hermes_state_runtime import (
    RuntimeStoreError, admit_session_input, begin_runtime_epoch, claim_session_input,
    mutate_worker_execution, register_worker_execution,
)
from hermes_state_terminal import RESULT_PREFIX
from gateway.session_results import admission_result, retain_result

# The documented bounds (hermes_state_runtime_workers.TERMINAL_RECEIPT_KEEP and
# hermes_state_terminal.ADMISSION_RESULT_KEEP). Kept literal here on purpose: these are the
# retention contract the card states, and moving them must be a deliberate change to it.
KEPT_RECEIPTS = 32
KEPT_RESULTS = 16


def _age(db, session_id, at):
    """Move a session (and its transcript) out of the retention window."""
    db._conn.execute("UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?", (at, at, session_id))
    db._conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (at, session_id))
    db._conn.commit()


def _execution(db, *, epoch, session_id, execution_id, receipts, finish, generation=0):
    register_worker_execution(db, epoch=epoch, execution_id=execution_id, session_id=session_id,
                              generation=generation, kind='kanban', adoption_secret='private')
    for sequence in range(1, receipts + 1):
        mutate_worker_execution(db, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                generation=generation, sequence=sequence, operation='transcript.append',
                                payload={'messages': [{'role': 'assistant', 'content': f'{execution_id} {sequence}'}]})
    if finish:
        mutate_worker_execution(db, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                generation=generation, sequence=receipts + 1, operation='execution.finish', payload={})


def _receipt_count(db, execution_id):
    return db._read_one('SELECT COUNT(*) FROM worker_receipts WHERE execution_id=?', (execution_id,))[0]


def _settled_turn(db, *, epoch, session_id, request_id, reply):
    admit_session_input(db, epoch=epoch, principal_id='api', session_id=session_id,
                        request_id=request_id, payload={'text': reply})
    row = claim_session_input(db, epoch=epoch, session_id=session_id)
    retain_result(db, epoch=epoch, row=row,
                  result={'result': {'final_response': reply, 'messages': []}, 'usage': {}})
    return row


def test_prune_caps_terminal_receipts_and_keeps_the_closing_one_replayable(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        for session_id in ('closed', 'live', 'idle'):
            db.create_session(session_id, source='cli')
        db.end_session('idle', 'done')
        _age(db, 'idle', time.time() - 2 * 86400)
        epoch = begin_runtime_epoch(db, instance_id='owner')
        receipts = KEPT_RECEIPTS * 2
        _execution(db, epoch=epoch, session_id='closed', execution_id='closed-worker',
                   receipts=receipts, finish=True)
        _execution(db, epoch=epoch, session_id='live', execution_id='live-worker',
                   receipts=receipts, finish=False)

        assert db.prune_sessions(older_than_days=1) == 1  # the pass that runs the reclaim

        # A live execution is in flight: every receipt is still a retry someone can send.
        assert _receipt_count(db, 'live-worker') == receipts
        # A terminal one keeps only its newest receipts, and the closing one still replays byte-exact.
        assert _receipt_count(db, 'closed-worker') == KEPT_RECEIPTS
        last_receipt = receipts + 1
        assert db._read_one('SELECT COUNT(*) FROM worker_receipts WHERE execution_id=? AND sequence=?',
                            ('closed-worker', last_receipt))[0] == 1
        assert mutate_worker_execution(db, epoch=epoch, execution_id='closed-worker', session_id='closed',
                                       generation=0, sequence=last_receipt, operation='execution.finish',
                                       payload={}) == {'status': 'terminal'}
        # A publication older than the retained window has no retry left: it is refused, not re-run.
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            mutate_worker_execution(db, epoch=epoch, execution_id='closed-worker', session_id='closed',
                                    generation=0, sequence=1, operation='transcript.append',
                                    payload={'messages': [{'role': 'assistant', 'content': 'closed-worker 1'}]})


def test_pruned_session_reclaims_its_receipts_and_keeps_the_retry_window_answer(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('gone', source='api_server')
        db.append_message('gone', 'user', 'hello')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        settled = _settled_turn(db, epoch=epoch, session_id='gone', request_id='once', reply='answer')
        admission_id = settled['admission_id']
        _execution(db, epoch=epoch, session_id='gone', execution_id='worker', receipts=5, finish=True,
                   generation=settled['generation'])
        db.end_session('gone', 'done')
        _age(db, 'gone', time.time() - 2 * 86400)

        assert db.prune_sessions(older_than_days=1) == 1

        assert db.get_session('gone') is None
        assert _receipt_count(db, 'worker') == 0, 'a pruned session kept its worker receipts'
        # The deletion fence still serves its exact retry inside the horizon: retirement keeps the
        # compacted answer, and the reclaim must not take it before that horizon closes.
        assert admission_result(db, admission_id)['result']['final_response'] == 'answer'

        from hermes_state_mutation_retirement import prune_retired_tombstones
        db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
        assert db._read_one('SELECT COUNT(*) FROM state_meta WHERE key=?', (RESULT_PREFIX + admission_id,))[0] == 0


def test_settled_results_are_capped_per_session_and_expire_with_it(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('chat', source='api_server')
        db.append_message('chat', 'user', 'start')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        turns = KEPT_RESULTS + 4
        admitted = [_settled_turn(db, epoch=epoch, session_id='chat', request_id=f'turn-{i}',
                                  reply=f'reply {i}')['admission_id'] for i in range(turns)]

        db.prune_sessions(older_than_days=1)  # nothing to prune: the cache is still bounded

        kept = [admission_id for admission_id in admitted if admission_result(db, admission_id) is not None]
        assert kept == admitted[-KEPT_RESULTS:]
        assert admission_result(db, kept[-1])['result']['final_response'] == f'reply {turns - 1}'

        # Past the retry horizon there is no retry to answer, even while the session row stays.
        _age(db, 'chat', time.time() - 8 * 86400)
        db.prune_sessions(older_than_days=1)
        assert [admission_id for admission_id in admitted if admission_result(db, admission_id) is not None] == []


def test_a_local_policy_receipt_goes_with_its_session_and_stranded_rows_are_reclaimed(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        def policy(session_id):
            return (POLICY_PREFIX + session_id,
                    json.dumps({'session_id': session_id, 'route': session_id, 'principal_id': 'user',
                                'request_id': 'req', 'profile_id': 'p', 'policy': {},
                                'entry': {'session_id': session_id}}))

        db.create_session('local', source='desktop')
        for session_id in ('local', 'stranded'):
            db._execute_write(lambda conn, row=policy(session_id): conn.execute(
                'INSERT INTO state_meta(key,value) VALUES(?,?)', row))

        db.delete_session('local', exclude_active_write_guards=True)
        assert db._read_one('SELECT COUNT(*) FROM state_meta WHERE key=?', (POLICY_PREFIX + 'local',))[0] == 0, \
            'a deleted session left its creation receipt behind'

        # A row an older release stranded (no session, no target) has no reader left: the pass reclaims it.
        db.prune_sessions(older_than_days=1)
        assert db._read_one('SELECT COUNT(*) FROM state_meta WHERE key=?', (POLICY_PREFIX + 'stranded',))[0] == 0
