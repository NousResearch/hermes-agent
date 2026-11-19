"""The SQLite backend's direct delete/prune paths, local reset and adopted-worker settlement stay consistent
with the ``session_admissions`` / ``worker_executions`` ledger (ON DELETE RESTRICT).

Reported by @GodsBoy and @ahrazzle on #106742; fix shape by @dialapus.
"""
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import (
    admit_session_input, claim_session_input, get_session_admission, list_session_admissions, settle_session_input,
)
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import (
    create_session, delete_empty_sessions, delete_session, delete_session_if_empty, delete_sessions,
    discard_unadmitted_session, get_session,
)
from persistence.sqlite.conversation._store.usage import update_token_counts
from persistence.sqlite.workers._executions import adopt_worker_execution, register_worker_execution
from persistence.sqlite.workers._receipt import mutate_worker_execution
from tests.contracts._scopes import MEDIA_ROOT


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def _old_ended(db, sid, source='cli'):
    create_session(db, sid, source=source)
    stamp = time.time() - 100 * 86400
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET ended_at=?, started_at=? WHERE id=?", (stamp, stamp, sid)))


def _settled_admission(db, epoch, sid, request_id='done'):
    accepted, _ = admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                                      request_id=request_id, payload={'text': 'x'})
    claim = claim_session_input(db, epoch=epoch, session_id=sid)
    settle_session_input(db, epoch=epoch, admission_id=accepted['admission_id'],
                         generation=claim['generation'], outcome='completed')
    return accepted['admission_id']


def _started_admission(db, epoch, sid, request_id='live'):
    admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                        request_id=request_id, payload={'text': 'x'})
    return claim_session_input(db, epoch=epoch, session_id=sid)


def _ledger_rows(db, sid):
    return db._read_one('SELECT (SELECT COUNT(*) FROM session_admissions WHERE target_session_id=?)'
                        ' + (SELECT COUNT(*) FROM worker_executions WHERE session_id=?)', (sid, sid))[0]


@pytest.mark.parametrize('delete', [
    lambda db, store, sid: delete_session(db, sid, media_root=MEDIA_ROOT),
    lambda db, store, sid: delete_sessions(db, [sid], media_root=MEDIA_ROOT)[0],
    lambda db, store, sid: delete_session_if_empty(db, sid, media_root=MEDIA_ROOT),
])
def test_legacy_delete_retires_terminal_ledger_rows_and_refuses_live_work(db, store, delete):
    epoch = begin_runtime_epoch(db, instance_id='owner')
    _old_ended(db, 'settled')
    admission_id = _settled_admission(db, epoch, 'settled')
    _old_ended(db, 'busy')
    _started_admission(db, epoch, 'busy')
    assert delete(db, store, 'settled')
    assert get_session(db, 'settled') is None and _ledger_rows(db, 'settled') == 0
    # The terminal admission survives as a tombstone (by id), not as history.
    tombstone = get_session_admission(db, admission_id=admission_id)
    assert tombstone['status'] == 'terminal' and tombstone['payload'] == {}
    with pytest.raises(AuthorityRefusal, match='session_busy'):
        delete(db, store, 'busy')
    assert get_session(db, 'busy') is not None and _ledger_rows(db, 'busy') == 1
    with db._read_ctx() as c:
        assert not c.execute('PRAGMA foreign_key_check').fetchall()


@pytest.mark.parametrize('delete', [
    lambda db, store, sid: delete_session(db, sid, media_root=MEDIA_ROOT),
    lambda db, store, sid: delete_sessions(db, [sid], media_root=MEDIA_ROOT)[0],
    lambda db, store, sid: delete_session_if_empty(db, sid, media_root=MEDIA_ROOT),
    lambda db, store, sid: delete_empty_sessions(db, media_root=MEDIA_ROOT),
    lambda db, store, sid: store.maintenance.prune_sessions(older_than_days=30, media_root=MEDIA_ROOT).count,
    lambda db, store, sid: store.maintenance.prune_empty_ghost_sessions(media_root=MEDIA_ROOT),
], ids=['single', 'bulk', 'if_empty', 'empty_sweep', 'prune', 'ghost_prune'])
def test_legacy_delete_publishes_the_full_retirement_fence(db, store, delete):
    """Same contract as the canonical mutate(delete): the exact retry of a settled request
    still returns its terminal receipt, and a late accounting backfill cannot recreate the
    row (which would let the same request be admitted a second time)."""
    from persistence.sqlite.authority._runtime_retirement import retired_session
    epoch = begin_runtime_epoch(db, instance_id='owner')
    _old_ended(db, 's', source='tui')
    store.routing.upsert_routing_entry('route:s', '{"session_id": "s"}')
    request = dict(epoch=epoch, principal_id='human', session_id='s', request_id='once',
                   payload={'text': 'x'})
    admission_id = _settled_admission(db, epoch, 's', 'once')
    assert delete(db, store, 's')
    assert get_session(db, 's') is None and retired_session(db, 's')
    assert store.routing.load_routing_entries() == {}
    retry, _ = admit_session_input(db, **request)
    assert retry['status'] == 'terminal' and retry['admission_id'] == admission_id
    with pytest.raises(AuthorityRefusal, match='not_found'):
        update_token_counts(db, 's', input_tokens=1, output_tokens=1, model='m')
    assert get_session(db, 's') is None


def test_legacy_delete_fences_cascaded_delegate_children(db):
    from persistence.sqlite.authority._runtime_retirement import retired_session
    epoch = begin_runtime_epoch(db, instance_id='owner')
    create_session(db, 'parent', source='cli')
    create_session(db, 'child', source='cli', parent_session_id='parent',
                   model_config={'_delegate_from': 'parent'})
    admission_id = _settled_admission(db, epoch, 'child', 'sub')
    assert delete_session(db, 'parent', media_root=MEDIA_ROOT)
    assert get_session(db, 'child') is None and retired_session(db, 'child')
    retry, _ = admit_session_input(db, epoch=epoch, principal_id='human', session_id='child',
                                   request_id='sub', payload={'text': 'x'})
    assert retry['status'] == 'terminal' and retry['admission_id'] == admission_id
    with pytest.raises(AuthorityRefusal, match='not_found'):
        create_session(db, 'child', source='unknown')


def test_sweeps_skip_sessions_with_live_work_and_retire_the_rest(db, store):
    epoch = begin_runtime_epoch(db, instance_id='owner')
    for sid, source in (('ghost_settled', 'tui'), ('ghost_busy', 'tui'),
                        ('old_settled', 'cli'), ('old_busy', 'cli'), ('old_plain', 'cli')):
        _old_ended(db, sid, source)
    _settled_admission(db, epoch, 'ghost_settled')
    _started_admission(db, epoch, 'ghost_busy')
    _settled_admission(db, epoch, 'old_settled')
    _started_admission(db, epoch, 'old_busy')

    assert len(store.maintenance.prune_empty_ghost_sessions(media_root=MEDIA_ROOT)) == 1
    assert get_session(db, 'ghost_settled') is None
    assert get_session(db, 'ghost_busy') is not None, 'a live session must not abort the whole prune'
    assert _ledger_rows(db, 'ghost_settled') == 0 and _ledger_rows(db, 'ghost_busy') == 1

    assert store.maintenance.prune_sessions(older_than_days=30, media_root=MEDIA_ROOT).count == 2
    assert get_session(db, 'old_settled') is None and get_session(db, 'old_plain') is None
    assert get_session(db, 'old_busy') is not None
    assert _ledger_rows(db, 'old_busy') == 1

    # The dashboard's empty-session sweep is the same class of bulk sweep.
    assert len(delete_empty_sessions(db, media_root=MEDIA_ROOT)) == 0
    assert get_session(db, 'ghost_busy') is not None and get_session(db, 'old_busy') is not None
    with db._read_ctx() as c:
        assert not c.execute('PRAGMA foreign_key_check').fetchall()


def _local_session(db, epoch):
    """Commit a local (CLI-owned) logical session; returns ``(session_id, reset_entry_dict)``."""
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_policy import build_policy, created_surface
    from persistence.authority.bindings import local_session_id
    from persistence.sqlite.authority._local import commit_local_session
    sid = local_session_id('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': str(Path(db.db_path).parent), 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(db, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'r', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)},
        client_request=None, created_source=created_surface(policy.source))
    reset = SessionEntry(entry.session_key, 'child', now, now, origin=source,
                         platform=Platform.LOCAL, is_fresh_reset=True)
    return sid, reset.to_dict()


def test_local_reset_refuses_over_started_admission_then_fences_generation(db):
    from persistence.sqlite.authority._local import reset_local_target
    epoch = begin_runtime_epoch(db, instance_id='owner')
    sid, reset = _local_session(db, epoch)
    started = _started_admission(db, epoch, sid)
    before = get_session(db, sid)
    with pytest.raises(AuthorityRefusal, match='session_busy'):
        reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset)
    assert get_session(db, 'child') is None and get_session(db, sid) == before
    settle_session_input(db, epoch=epoch, admission_id=started['admission_id'],
                         generation=started['generation'], outcome='completed')
    reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset)
    after = get_session(db, sid)
    assert after['end_reason'] == 'session_reset' and get_session(db, 'child') is not None
    assert after['runtime_generation'] == before['runtime_generation'] + 1
    # A worker registered against the pre-reset generation is fenced out.
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        register_worker_execution(db, epoch=epoch, execution_id='stale', session_id=sid,
                                  generation=before['runtime_generation'], kind='compute',
                                  adoption_secret='private-fixture')


@pytest.mark.parametrize('adopted', [False, True], ids=['registered', 'running'])
def test_local_reset_refuses_over_live_compute_worker_but_not_a_queued_follower(db, adopted):
    """A registered/running worker is executing even though no admission is 'started' (an
    idle-registered worker has none; an adopted one may finish through execution.finish). Reset
    over it would bump the generation, strand the worker's persists as stale_generation and
    leave the queued follower unclaimable. A queued follower alone must still allow reset."""
    from persistence.sqlite.authority._local import reset_local_target
    epoch = begin_runtime_epoch(db, instance_id='owner')
    sid, reset = _local_session(db, epoch)
    scope = dict(execution_id='worker', session_id=sid, generation=0)
    register_worker_execution(db, epoch=epoch, **scope, kind='compute',
                              adoption_secret='private', require_idle=True)
    if adopted:
        adopt_worker_execution(db, epoch=epoch, **scope, adoption_secret='private')
    follower, _ = admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                                      request_id='follower', payload={'text': 'next'})
    before = get_session(db, sid)
    with pytest.raises(AuthorityRefusal, match='session_busy'):
        reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset)
    assert get_session(db, sid) == before and get_session(db, 'child') is None
    mutate_worker_execution(db, epoch=epoch, **scope, sequence=1, operation='transcript.append',
                            payload={'messages': [{'role': 'assistant', 'content': 'ok'}]})
    mutate_worker_execution(db, epoch=epoch, **scope, sequence=2, operation='execution.finish', payload={})
    # Worker terminal, follower still queued: /reset is exactly what the follower waits on.
    reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset)
    assert get_session(db, sid)['runtime_generation'] == before['runtime_generation'] + 1
    claimed = claim_session_input(db, epoch=epoch, session_id=sid)
    assert claimed is not None and claimed['admission_id'] == follower['admission_id']


def test_discarding_an_unadmitted_row_leaves_no_fence_but_an_admitted_row_is_fenced(db):
    """Seed-copy compensation removes a row nobody was admitted against WITHOUT the retirement
    marker (the lazy first-prompt path must be able to recreate the id); once any receipt
    exists the same call takes the fenced delete so the id cannot be resurrected."""
    from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX
    create_session(db, 'fresh', source='desktop')
    assert 'fresh' in discard_unadmitted_session(db, 'fresh', media_root=MEDIA_ROOT)
    assert get_session(db, 'fresh') is None
    assert db.get_meta(RETIRED_PREFIX + 'fresh') is None
    create_session(db, 'fresh', source='desktop')  # lazy recreation still allowed
    assert get_session(db, 'fresh') is not None

    create_session(db, 'used', source='desktop')
    epoch = begin_runtime_epoch(db, instance_id='owner')
    _settled_admission(db, epoch, 'used')
    assert 'used' in discard_unadmitted_session(db, 'used', media_root=MEDIA_ROOT)
    assert db.get_meta(RETIRED_PREFIX + 'used') is not None


def test_local_reset_stamps_ended_at_on_the_same_float_clock_as_started_at(db):
    """``ended_at`` uses the ``time.time()`` clock of ``started_at`` and every sibling end path:
    SQLite's integer-second clock truncates, so a reset in the same second as the creation
    recorded a session that ended before it started."""
    from persistence.sqlite.authority._local import reset_local_target
    epoch = begin_runtime_epoch(db, instance_id='owner')
    sid, reset = _local_session(db, epoch)
    before = time.time()
    reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset)
    parent = get_session(db, sid)
    assert parent['ended_at'] >= max(parent['started_at'], before)


def test_pending_and_idle_probes_do_not_scan_finished_history(db, monkeypatch):
    """pastels minor: ``status!='terminal'`` cannot bound the (session, status) indexes, so every
    pending/idle probe walked the session's whole finished ledger. Probes must stay O(live rows)."""
    from contextlib import nullcontext
    from persistence.sqlite.authority._local import end_idle_local_session
    from persistence.sqlite.authority._runtime_guards import require_idle
    from persistence.sqlite.conversation._store.retirement import _LIVE_LEDGER_SQL
    create_session(db, 's', source='api_server')
    epoch = begin_runtime_epoch(db, instance_id='owner')
    def history(conn):
        conn.executemany(
            "INSERT INTO session_admissions(admission_id,request_id,principal_id,target_session_id,lineage_json,"
            "payload_json,payload_digest,intent,status,outcome,owner_epoch,generation) "
            "VALUES(?,?,'p','s','[]','{}','d','queue','terminal','completed',1,1)",
            [(f'a{i}', f'r{i}') for i in range(20000)])
        conn.executemany("INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,"
                         "status,adoption_digest) VALUES(?,'s','compute',1,1,'terminal','d')",
                         [(f'w{i}',) for i in range(20000)])
    db._execute_write(history)
    monkeypatch.setattr(db, '_read_ctx', lambda: nullcontext(db._conn))
    ticks = []
    db._conn.set_progress_handler(lambda: ticks.append(1) and 0, 1000)
    probes = {
        'pending': lambda: list_session_admissions(db, session_id='s'),
        'require_idle': lambda: db._execute_write(lambda c: require_idle(db, c, ['s'])),
        'prune_live_check': lambda: db._execute_write(lambda c: c.execute(_LIVE_LEDGER_SQL, ('s', 's')).fetchone()),
        'end_idle': lambda: end_idle_local_session(db, epoch=epoch, session_id='s', target_id='s', reason='idle'),
    }
    for name, probe in probes.items():
        ticks.clear()
        probe()
        # A scan of 40k finished rows is hundreds of thousands of VM steps; a bounded probe is a few.
        assert len(ticks) < 20, f'{name} scanned finished history ({len(ticks)}k VM steps)'
    db._conn.set_progress_handler(None, 0)
