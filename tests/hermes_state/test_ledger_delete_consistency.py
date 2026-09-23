"""Legacy delete/prune paths, local reset and adopted-worker settlement stay consistent
with the ``session_admissions`` / ``worker_executions`` ledger (ON DELETE RESTRICT).

Reported by @GodsBoy and @ahrazzle on #106742; fix shape by @dialapus.
"""
import logging
import time
from contextlib import closing
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def _old_ended(db, sid, source='cli'):
    db.create_session(sid, source=source)
    stamp = time.time() - 100 * 86400
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET ended_at=?, started_at=? WHERE id=?", (stamp, stamp, sid)))


def _settled_admission(db, epoch, sid, request_id='done'):
    accepted = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                                      request_id=request_id, payload={'text': 'x'})
    claim = rt.claim_session_input(db, epoch=epoch, session_id=sid)
    rt.settle_session_input(db, epoch=epoch, admission_id=accepted['admission_id'],
                            generation=claim['generation'], outcome='completed')
    return accepted['admission_id']


def _started_admission(db, epoch, sid, request_id='live'):
    rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                           request_id=request_id, payload={'text': 'x'})
    return rt.claim_session_input(db, epoch=epoch, session_id=sid)


def _ledger_rows(db, sid):
    return db._read_one('SELECT (SELECT COUNT(*) FROM session_admissions WHERE target_session_id=?)'
                        ' + (SELECT COUNT(*) FROM worker_executions WHERE session_id=?)', (sid, sid))[0]


@pytest.mark.parametrize('delete', [
    lambda db, sid: db.delete_session(sid),
    lambda db, sid: db.delete_sessions([sid]),
    lambda db, sid: db.delete_session_if_empty(sid),
])
def test_legacy_delete_retires_terminal_ledger_rows_and_refuses_live_work(tmp_path, delete):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        _old_ended(db, 'settled')
        admission_id = _settled_admission(db, epoch, 'settled')
        _old_ended(db, 'busy')
        _started_admission(db, epoch, 'busy')
        assert delete(db, 'settled')
        assert db.get_session('settled') is None and _ledger_rows(db, 'settled') == 0
        # The terminal admission survives as a tombstone (by id), not as history.
        tombstone = rt.get_session_admission(db, admission_id=admission_id)
        assert tombstone['status'] == 'terminal' and tombstone['payload'] == {}
        with pytest.raises(rt.RuntimeStoreError, match='session_busy'):
            delete(db, 'busy')
        assert db.get_session('busy') is not None and _ledger_rows(db, 'busy') == 1
        with db._read_ctx() as c:
            assert not c.execute('PRAGMA foreign_key_check').fetchall()


@pytest.mark.parametrize('delete', [
    lambda db, sid: db.delete_session(sid),
    lambda db, sid: db.delete_sessions([sid]),
    lambda db, sid: db.delete_session_if_empty(sid),
    lambda db, sid: db.delete_empty_sessions(),
    lambda db, sid: db.prune_sessions(older_than_days=30),
    lambda db, sid: db.prune_empty_ghost_sessions(),
], ids=['single', 'bulk', 'if_empty', 'empty_sweep', 'prune', 'ghost_prune'])
def test_legacy_delete_publishes_the_full_retirement_fence(tmp_path, delete):
    """Same contract as the canonical mutate(delete): the exact retry of a settled request
    still returns its terminal receipt, and a late accounting backfill cannot recreate the
    row (which would let the same request be admitted a second time)."""
    from hermes_state_mutation_retirement import retired_session
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        _old_ended(db, 's', source='tui')
        db.save_gateway_routing_entry('route:s', '{"session_id": "s"}')
        request = dict(epoch=epoch, principal_id='human', session_id='s', request_id='once',
                       payload={'text': 'x'})
        admission_id = _settled_admission(db, epoch, 's', 'once')
        assert delete(db, 's')
        assert db.get_session('s') is None and retired_session(db, 's')
        assert db.load_gateway_routing_entries() == {}
        retry = rt.admit_session_input(db, **request)
        assert retry['status'] == 'terminal' and retry['admission_id'] == admission_id
        with pytest.raises(rt.RuntimeStoreError, match='not_found'):
            db.update_token_counts('s', input_tokens=1, output_tokens=1, model='m')
        assert db.get_session('s') is None


def test_legacy_delete_fences_cascaded_delegate_children(tmp_path):
    from hermes_state_mutation_retirement import retired_session
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.create_session('parent', source='cli')
        db.create_session('child', source='cli', parent_session_id='parent',
                          model_config={'_delegate_from': 'parent'})
        admission_id = _settled_admission(db, epoch, 'child', 'sub')
        assert db.delete_session('parent')
        assert db.get_session('child') is None and retired_session(db, 'child')
        retry = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='child',
                                       request_id='sub', payload={'text': 'x'})
        assert retry['status'] == 'terminal' and retry['admission_id'] == admission_id
        with pytest.raises(rt.RuntimeStoreError, match='not_found'):
            db.ensure_session('child', source='unknown')


def test_sweeps_skip_sessions_with_live_work_and_retire_the_rest(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        for sid, source in (('ghost_settled', 'tui'), ('ghost_busy', 'tui'),
                            ('old_settled', 'cli'), ('old_busy', 'cli'), ('old_plain', 'cli')):
            _old_ended(db, sid, source)
        _settled_admission(db, epoch, 'ghost_settled')
        _started_admission(db, epoch, 'ghost_busy')
        _settled_admission(db, epoch, 'old_settled')
        _started_admission(db, epoch, 'old_busy')

        assert db.prune_empty_ghost_sessions() == 1
        assert db.get_session('ghost_settled') is None
        assert db.get_session('ghost_busy') is not None, 'a live session must not abort the whole prune'
        assert _ledger_rows(db, 'ghost_settled') == 0 and _ledger_rows(db, 'ghost_busy') == 1

        assert db.prune_sessions(older_than_days=30) == 2
        assert db.get_session('old_settled') is None and db.get_session('old_plain') is None
        assert db.get_session('old_busy') is not None
        assert _ledger_rows(db, 'old_busy') == 1

        # The dashboard's empty-session sweep is the same class of bulk sweep.
        assert db.delete_empty_sessions() == 0
        assert db.get_session('ghost_busy') is not None and db.get_session('old_busy') is not None
        with db._read_ctx() as c:
            assert not c.execute('PRAGMA foreign_key_check').fetchall()


def test_startup_repair_latches_after_a_refusal_and_warns_once(monkeypatch, caplog, tmp_path):
    import cli
    import hermes_cli.config
    import hermes_constants

    monkeypatch.setattr(hermes_cli.config, 'load_config', lambda: {'sessions': {'auto_prune': False}})
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: tmp_path)
    meta = {}
    session_db = MagicMock()
    session_db.get_meta.side_effect = meta.get
    session_db.set_meta.side_effect = meta.__setitem__
    session_db.prune_empty_ghost_sessions.side_effect = rt.RuntimeStoreError('session_busy')
    session_db.finalize_orphaned_compression_sessions.return_value = 0

    with caplog.at_level(logging.DEBUG, logger='cli'):
        cli._run_state_db_auto_maintenance(session_db)
        cli._run_state_db_auto_maintenance(session_db)
    refusals = [r for r in caplog.records if 'Ghost session prune skipped' in r.getMessage()]
    assert [r.levelno for r in refusals] == [logging.WARNING], 'refusal logged once, at warning'
    assert session_db.prune_empty_ghost_sessions.call_count == 1, 'latched after the refusal'
    assert meta.get('ghost_session_prune_v1')


def _local_session(db, epoch):
    """Commit a local (CLI-owned) logical session; returns ``(session_id, reset_entry_dict)``."""
    from hermes_state_local import commit_local_session
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_local_recovery import local_identity
    from gateway.session_policy import build_policy
    sid = local_identity('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': '/', 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(db, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'r', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    reset = SessionEntry(entry.session_key, 'child', now, now, origin=source,
                         platform=Platform.LOCAL, is_fresh_reset=True)
    return sid, reset.to_dict()


def test_local_reset_refuses_over_started_admission_then_fences_generation(tmp_path):
    from hermes_state_local_lineage import reset_local_target
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        sid, reset = _local_session(db, epoch)
        started = _started_admission(db, epoch, sid)
        before = db.get_session(sid)
        with pytest.raises(rt.RuntimeStoreError, match='session_busy'):
            reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset)
        assert db.get_session('child') is None and db.get_session(sid) == before
        rt.settle_session_input(db, epoch=epoch, admission_id=started['admission_id'],
                                generation=started['generation'], outcome='completed')
        reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset)
        after = db.get_session(sid)
        assert after['end_reason'] == 'session_reset' and db.get_session('child') is not None
        assert after['runtime_generation'] == before['runtime_generation'] + 1
        # A worker registered against the pre-reset generation is fenced out.
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.register_worker_execution(db, epoch=epoch, execution_id='stale', session_id=sid,
                                         generation=before['runtime_generation'], kind='compute',
                                         adoption_secret='private-fixture')


@pytest.mark.parametrize('adopted', [False, True], ids=['registered', 'running'])
def test_local_reset_refuses_over_live_compute_worker_but_not_a_queued_follower(tmp_path, adopted):
    """A registered/running worker is executing even though no admission is 'started' (an
    idle-registered worker has none; an adopted one may finish through execution.finish). Reset
    over it would bump the generation, strand the worker's persists as stale_generation and
    leave the queued follower unclaimable. A queued follower alone must still allow reset."""
    from hermes_state_local_lineage import reset_local_target
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        sid, reset = _local_session(db, epoch)
        scope = dict(execution_id='worker', session_id=sid, generation=0)
        rt.register_worker_execution(db, epoch=epoch, **scope, kind='compute',
                                     adoption_secret='private', require_idle=True)
        if adopted:
            rt.adopt_worker_execution(db, epoch=epoch, **scope, adoption_secret='private')
        follower = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                                          request_id='follower', payload={'text': 'next'})
        before = db.get_session(sid)
        with pytest.raises(rt.RuntimeStoreError, match='session_busy'):
            reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset)
        assert db.get_session(sid) == before and db.get_session('child') is None
        rt.persist_worker_message(db, epoch=epoch, **scope, sequence=1, role='assistant', content='ok')
        rt.finish_worker_execution(db, epoch=epoch, **scope)
        # Worker terminal, follower still queued: /reset is exactly what the follower waits on.
        reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset)
        assert db.get_session(sid)['runtime_generation'] == before['runtime_generation'] + 1
        claimed = rt.claim_session_input(db, epoch=epoch, session_id=sid)
        assert claimed is not None and claimed['admission_id'] == follower['admission_id']


def test_adopted_worker_finish_settles_linked_admission_and_frees_follower(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        first = _started_admission(db, epoch, 's', 'first')
        assignment = dict(execution_id='compute', session_id='s', generation=first['generation'])
        rt.register_worker_execution(db, epoch=epoch, **assignment, kind='compute', adoption_secret='private')
        follower = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s',
                                          request_id='second', payload={})
        epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=epoch)
        rt.adopt_worker_execution(db, epoch=epoch, **assignment, adoption_secret='private')
        assert rt.get_session_admission(db, admission_id=first['admission_id'])['status'] == 'started'
        assert rt.claim_session_input(db, epoch=epoch, session_id='s') is None
        rt.persist_worker_message(db, epoch=epoch, **assignment, sequence=1, role='assistant', content='done')
        rt.finish_worker_execution(db, epoch=epoch, **assignment)
        settled = rt.get_session_admission(db, admission_id=first['admission_id'])
        assert settled['status'] == 'terminal' and settled['outcome'] == 'completed'
        claimed = rt.claim_session_input(db, epoch=epoch, session_id='s')
        assert claimed is not None and claimed['admission_id'] == follower['admission_id']


def test_discarding_an_unadmitted_row_leaves_no_fence_but_an_admitted_row_is_fenced(tmp_path):
    """Seed-copy compensation removes a row nobody was admitted against WITHOUT the retirement
    marker (the lazy first-prompt path must be able to recreate the id); once any receipt
    exists the same call takes the fenced delete so the id cannot be resurrected."""
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    with closing(SessionDB(db_path=tmp_path / 'state.db')) as db:
        db.create_session('fresh', source='desktop')
        assert db.discard_unadmitted_session('fresh') is True
        assert db.get_session('fresh') is None
        assert db.get_meta(RETIRED_PREFIX + 'fresh') is None
        db.create_session('fresh', source='desktop')  # lazy recreation still allowed
        assert db.get_session('fresh') is not None

        db.create_session('used', source='desktop')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        _settled_admission(db, epoch, 'used')
        assert db.discard_unadmitted_session('used') is True
        assert db.get_meta(RETIRED_PREFIX + 'used') is not None
