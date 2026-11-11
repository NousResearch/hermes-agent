"""#47 Phase 2 — cross-session ownership gate on fresh Kanban admissions.

The gate lives in ``gateway/session_kanban.py::run_task``'s ``admit_claim``
closure. Every test runs against a private ``tmp_path`` board (never the
production DB) and drives ``run_task`` end to end with the session layer
mocked, so the real SQLite admission transaction is exercised.

Groups:
  (a) fresh admission claims the ledger and marks owner_admitted
  (b) a live foreign owner refuses admission and releases the card claim
  (c) the same session re-claiming its own scope is idempotent and admitted
  (d) HERMES_KANBAN_OWNERSHIP_ENFORCE=0 skips the ledger entirely
  (e) an underivable scope refuses as ownership_unavailable
  (f) an OwnershipError inside the claim refuses as ownership_txn_error
  (g) the breaker parks a twice-failed card 'blocked', not 'ready'
  (h) the pre-existing owner_admitted marker branch is unchanged
"""
import asyncio
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from hermes_state_runtime import RuntimeStoreError

# R1 (#47 Phase 2): import the recovery module at COLLECTION time, before any
# test patches ``hermes_state_local.local_receipt`` (see test_kanban_finite_admission).
import gateway.session_local_recovery as _slr

BOARD = 'owned'
OTHER = 'other-session'
OWNER = 'owner-session'
BODY = 'source_repo: Andrew288369/ANDREW-GRP\nsource_issue: 47'
SCOPE = 'gh:andrew288369/andrew-grp#47'


def _prepare(tmp_path, monkeypatch, *, body=BODY, consecutive_failures=0):
    """Private board + a claimed (running) task. Returns (workspace, task)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env',
                        lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir(exist_ok=True)
    kb.create_board(BOARD)
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        task_id = kb.create_task(conn, title='Owned', assignee='default', workspace_kind='dir',
                                 workspace_path=str(workspace), body=body)
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
        if consecutive_failures:
            conn.execute('UPDATE tasks SET consecutive_failures=? WHERE id=?',
                         (consecutive_failures, task_id))
            task = kb.get_task(conn, task_id)
    return workspace, task


def _run_task(tmp_path, task, monkeypatch, *, session_id=OWNER, enforce='1'):
    """Wire run_task with a mocked session layer; returns (coroutine, params, calls)."""
    from gateway.session_contract import Principal
    from gateway.session_kanban import run_task
    if enforce is None:
        monkeypatch.delenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', raising=False)
    else:
        monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', enforce)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'},
              'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    monkeypatch.setattr('gateway.session_local.create_local_session',
                        lambda *a, **k: SimpleNamespace(session_id=session_id, profile_id=str(tmp_path)))
    monkeypatch.setattr('gateway.session_kanban.asdict', lambda receipt: {})
    calls = []

    async def submit(actor, submission, **kwargs):
        calls.append(('submit', submission))
        return SimpleNamespace(admission_id='a')

    async def resume(ref, params):
        calls.append(('resume', ref))

    # The session layer is mocked: ``get_session`` must resolve a REAL session
    # (the gate's ``verify_session`` checks exactly that), and ``local_receipt``
    # must not return a saved claim or the retry branch would replace the fresh
    # admission under test.
    monkeypatch.setattr('hermes_state_local.local_receipt', lambda db, sid: None)
    monkeypatch.setattr(_slr, 'local_receipt', lambda db, sid: None)
    db = SimpleNamespace(get_session=lambda sid: True, db_path=str(tmp_path / 'state.db'))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, submit=submit)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    params = dict(board=BOARD, task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    return run_task, connection, params, calls


def _pre_own(tmp_path, scope, session_id):
    """Seed the ledger with a live foreign/same owner on the private board."""
    from cross_session_ownership import claim_ownership, init_schema
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        init_schema(conn)
        return claim_ownership(conn, scope, session_id, ttl_seconds=900)


def _seed_marker(tmp_path, task, *, session_id):
    """Insert an owner_admitted marker for this attempt (the retry branch)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    request_id = 'kanban:' + json.dumps([BOARD, task.id, task.current_run_id], separators=(',', ':'))
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        kb._append_event(conn, task.id, 'owner_admitted',
                         {'session_id': session_id, 'request_id': request_id},
                         run_id=task.current_run_id)


def _board_rows(tmp_path, task_id):
    """Read the post-run card + ledger state."""
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        task = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        events = [dict(r) for r in conn.execute(
            'SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id', (task_id,))]
        # The ledger table is created lazily (init_schema) only when the gate runs;
        # enforce-off / missing-scope / marker-branch cases never create it.
        has_ownership = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_ownership'").fetchone()
        owners = [dict(r) for r in conn.execute('SELECT * FROM task_ownership')] if has_ownership else []
        run = conn.execute('SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1',
                           (task_id,)).fetchone()
    return task, events, owners, run


# (a) ----------------------------------------------------------------------
def test_a_fresh_admission_claims_ownership_and_marks_owner(tmp_path, monkeypatch):
    _workspace, task = _prepare(tmp_path, monkeypatch)
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    result = asyncio.run(run_task(connection, params))
    assert result['session_id'] == OWNER
    assert [c[0] for c in calls] == ['resume', 'submit']
    card, events, owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'running' and card['current_run_id'] == task.current_run_id
    kinds = [e['kind'] for e in events]
    assert kinds.count('owner_admitted') == 1 and 'ownership_refused' not in kinds
    assert len(owners) == 1 and owners[0]['owner_session'] == OWNER
    assert owners[0]['state'] == 'ACTIVE' and owners[0]['scope_key'] == SCOPE


# (b) ----------------------------------------------------------------------
def test_b_owned_by_other_live_session_is_refused_and_released(tmp_path, monkeypatch):
    _workspace, task = _prepare(tmp_path, monkeypatch)
    assert _pre_own(tmp_path, SCOPE, OTHER)['ok'] is True
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, events, owners, run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'ready'
    assert card['claim_lock'] is None and card['claim_expires'] is None
    assert card['worker_pid'] is None and card['current_run_id'] is None
    assert card['last_failure_error'] == 'ownership_refused'
    assert run['status'] == 'refused' and run['ended_at'] is not None
    refused = [e for e in events if e['kind'] == 'ownership_refused']
    assert len(refused) == 1
    assert json.loads(refused[0]['payload'])['reason'] == 'owned_by_other_session'
    assert [e['kind'] for e in events].count('owner_admitted') == 0
    # The ledger was not stolen: the foreign owner is untouched.
    assert len(owners) == 1 and owners[0]['owner_session'] == OTHER
    assert owners[0]['state'] == 'ACTIVE'


# (c) ----------------------------------------------------------------------
def test_c_same_session_reclaim_is_idempotent_and_admitted(tmp_path, monkeypatch):
    _workspace, task = _prepare(tmp_path, monkeypatch)
    assert _pre_own(tmp_path, SCOPE, OWNER)['ok'] is True
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    result = asyncio.run(run_task(connection, params))
    assert result['session_id'] == OWNER
    assert [c[0] for c in calls] == ['resume', 'submit']
    card, events, owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'running'
    kinds = [e['kind'] for e in events]
    assert kinds.count('owner_admitted') == 1 and 'ownership_refused' not in kinds
    assert len(owners) == 1 and owners[0]['owner_session'] == OWNER
    assert owners[0]['state'] == 'ACTIVE'


# (d) ----------------------------------------------------------------------
def test_d_enforce_off_skips_the_ledger(tmp_path, monkeypatch):
    _workspace, task = _prepare(tmp_path, monkeypatch)
    assert _pre_own(tmp_path, SCOPE, OTHER)['ok'] is True
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch, enforce='0')
    result = asyncio.run(run_task(connection, params))
    assert result['session_id'] == OWNER
    assert [c[0] for c in calls] == ['resume', 'submit']
    card, events, owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'running'
    kinds = [e['kind'] for e in events]
    assert kinds.count('owner_admitted') == 1 and 'ownership_refused' not in kinds
    # The gate never ran: the foreign owner row is exactly as seeded.
    assert len(owners) == 1 and owners[0]['owner_session'] == OTHER


# (e) ----------------------------------------------------------------------
def test_e_missing_scope_is_refused_unavailable(tmp_path, monkeypatch):
    _workspace, task = _prepare(tmp_path, monkeypatch)
    monkeypatch.setattr('cross_session_ownership.canonical_key_from_kanban', lambda task, **k: None)
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, events, owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'ready' and card['claim_lock'] is None
    refused = [e for e in events if e['kind'] == 'ownership_refused']
    assert len(refused) == 1
    assert json.loads(refused[0]['payload'])['reason'] == 'ownership_unavailable'
    assert owners == []


# (f) ----------------------------------------------------------------------
def test_f_ownership_txn_error_is_refused(tmp_path, monkeypatch):
    _workspace, task = _prepare(tmp_path, monkeypatch)

    def boom(*a, **k):
        from cross_session_ownership import OwnershipError
        raise OwnershipError('ledger unavailable')

    monkeypatch.setattr('cross_session_ownership.claim_ownership', boom)
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, events, owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'ready' and card['claim_lock'] is None
    refused = [e for e in events if e['kind'] == 'ownership_refused']
    assert len(refused) == 1
    assert json.loads(refused[0]['payload'])['reason'] == 'ownership_txn_error'
    assert owners == []


# (g) ----------------------------------------------------------------------
def test_g_breaker_parks_twice_failed_card_blocked(tmp_path, monkeypatch):
    _workspace, task = _prepare(tmp_path, monkeypatch, consecutive_failures=2)
    assert _pre_own(tmp_path, SCOPE, OTHER)['ok'] is True
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, events, _owners, run = _board_rows(tmp_path, task.id)
    # Breaker: >=2 consecutive failures parks the card instead of re-queueing it.
    assert card['status'] == 'blocked'
    assert card['claim_lock'] is None and card['current_run_id'] is None
    assert run['status'] == 'refused'
    assert [e['kind'] for e in events].count('owner_admitted') == 0


# (h) ----------------------------------------------------------------------
def test_h_existing_marker_branch_is_unchanged(tmp_path, monkeypatch):
    _workspace, task = _prepare(tmp_path, monkeypatch)
    assert _pre_own(tmp_path, SCOPE, OTHER)['ok'] is True
    # Same-session marker: the retry branch admits without consulting the ledger.
    _seed_marker(tmp_path, task, session_id=OWNER)
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    result = asyncio.run(run_task(connection, params))
    assert result['session_id'] == OWNER
    assert [c[0] for c in calls] == ['resume', 'submit']
    card, events, owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'running'
    assert [e['kind'] for e in events].count('owner_admitted') == 1
    assert len(owners) == 1 and owners[0]['owner_session'] == OTHER


def test_h_existing_marker_for_another_session_is_stale(tmp_path, monkeypatch):
    _workspace, task = _prepare(tmp_path, monkeypatch)
    _seed_marker(tmp_path, task, session_id=OTHER)
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, events, owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'running'  # untouched: the marker branch raised
    assert [e['kind'] for e in events].count('owner_admitted') == 1
    assert owners == []


# (i) P1-A — canonical_key_from_kanban raising must not escape the admission txn
def test_i_canonical_key_ownership_error_releases_card(tmp_path, monkeypatch):
    """A whitespace-bearing source_repo makes validate_scope_key raise OwnershipError
    inside canonical_key_from_kanban. Before the fix the exception escaped
    kb.write_txn, rolled the admission back, and left the card 'running' with a
    stale claim_lock forever. Now it fails closed: refused + released in-txn."""
    bad_body = 'source_repo: Andrew288369 /ANDREW-GRP\nsource_issue: 47'
    _workspace, task = _prepare(tmp_path, monkeypatch, body=bad_body)
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, events, owners, run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'ready'
    assert card['claim_lock'] is None and card['claim_expires'] is None
    assert card['worker_pid'] is None and card['current_run_id'] is None
    assert card['last_failure_error'] == 'ownership_refused'
    assert run['status'] == 'refused' and run['ended_at'] is not None
    refused = [e for e in events if e['kind'] == 'ownership_refused']
    assert len(refused) == 1
    assert json.loads(refused[0]['payload'])['reason'] == 'ownership_txn_error'
    assert [e['kind'] for e in events].count('owner_admitted') == 0
    assert owners == []


# (j) P1-A — a corrupt consecutive_failures counter fails closed to 'blocked'
def test_j_corrupt_failure_counter_fails_closed_blocked(tmp_path, monkeypatch):
    """int('not-a-number') raises ValueError inside the refusal path. The defensive
    release helper must park the card 'blocked' (never re-queue it) and must not
    let the exception escape the admission transaction."""
    _workspace, task = _prepare(tmp_path, monkeypatch)
    assert _pre_own(tmp_path, SCOPE, OTHER)['ok'] is True
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.execute("UPDATE tasks SET consecutive_failures=? WHERE id=?",
                     ('not-a-number', task.id))
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, events, _owners, run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'blocked'
    assert card['claim_lock'] is None and card['current_run_id'] is None
    assert run['status'] == 'refused'
    refused = [e for e in events if e['kind'] == 'ownership_refused']
    assert len(refused) == 1
    assert json.loads(refused[0]['payload'])['reason'] == 'owned_by_other_session'
    assert [e['kind'] for e in events].count('owner_admitted') == 0


# (k) P1-A — the defensive release helper still releases when the row is missing
def test_k_release_helper_missing_row_still_releases_by_task_id(tmp_path, monkeypatch):
    from gateway.session_kanban import _release_refused_claim
    from hermes_cli.kanban_db_connect import connect
    _workspace, task = _prepare(tmp_path, monkeypatch)
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        _release_refused_claim(conn, task, None, refused='ownership_txn_error',
                               scope=None, scope_reason=None)
    card, events, _owners, run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'ready'
    assert card['claim_lock'] is None and card['claim_expires'] is None
    assert card['worker_pid'] is None and card['current_run_id'] is None
    assert run['status'] == 'refused'
    assert [e['kind'] for e in events].count('ownership_refused') == 1
    assert json.loads(next(e for e in events if e['kind'] == 'ownership_refused')['payload'])['reason'] \
        == 'ownership_txn_error'


# (l) C1 — the refusal's durable half commits even when bookkeeping raises
#
# ``sqlite3.Connection`` is immutable: ``monkeypatch.setattr(sqlite3.Connection,
# 'execute', ...)`` raises ``TypeError: cannot set 'execute' attribute of
# immutable type``. The seam is the module-level ``kb._end_run`` /
# ``kb._append_event`` attributes, which ``_release_refused_claim`` looks up on
# ``kb`` at call time — patching those exercises the real SQL UPDATE in the same
# transaction.
def test_l_end_run_raise_still_commits_the_refusal(tmp_path, monkeypatch):
    """``_end_run`` raising must not roll back the release.

    The UPDATE in ``_release_refused_claim`` is the refusal's durable half;
    closing the run is bookkeeping. Before C1 the exception escaped into the
    caller's fail-closed handler, which re-ran the helper with the
    pre-increment row and defeated the breaker. The card must be released and
    the counter incremented even though the run close failed.
    """
    from hermes_cli import kanban_db as kb
    _workspace, task = _prepare(tmp_path, monkeypatch)
    assert _pre_own(tmp_path, SCOPE, OTHER)['ok'] is True

    def boom(*a, **k):
        raise RuntimeError('end_run exploded')

    monkeypatch.setattr(kb, '_end_run', boom)
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, _events, _owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'ready'
    assert card['claim_lock'] is None and card['claim_expires'] is None
    assert card['worker_pid'] is None
    assert card['consecutive_failures'] == 1
    assert card['last_failure_error'] == 'ownership_refused'
    # The claim is really gone: the card can be claimed again (not stuck).
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        assert kb.claim_task(conn, task.id) is not None


def test_l_append_event_raise_still_releases_the_claim(tmp_path, monkeypatch):
    """``_append_event`` raising must not roll back the release either."""
    from hermes_cli import kanban_db as kb
    _workspace, task = _prepare(tmp_path, monkeypatch)
    assert _pre_own(tmp_path, SCOPE, OTHER)['ok'] is True
    real_append_event = kb._append_event

    def boom(*a, **k):
        raise RuntimeError('append_event exploded')

    monkeypatch.setattr(kb, '_append_event', boom)
    run_task, connection, params, calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    card, events, _owners, run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'ready'
    assert card['claim_lock'] is None and card['claim_expires'] is None
    assert card['worker_pid'] is None and card['current_run_id'] is None
    assert card['consecutive_failures'] == 1
    assert card['last_failure_error'] == 'ownership_refused'
    # The run close succeeded (it runs before the event append).
    assert run['status'] == 'refused' and run['ended_at'] is not None
    # The event append failed, so no ownership_refused marker was written.
    assert [e['kind'] for e in events].count('ownership_refused') == 0
    # Restore the real append: the re-claim's own bookkeeping needs it.
    monkeypatch.setattr(kb, '_append_event', real_append_event)
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        assert kb.claim_task(conn, task.id) is not None


def test_l_bookkeeping_raise_never_defeats_the_breaker(tmp_path, monkeypatch):
    """Both bookkeeping calls raising: the counter still increments, so a
    second refusal parks the card instead of re-queueing it."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    _workspace, task = _prepare(tmp_path, monkeypatch)
    assert _pre_own(tmp_path, SCOPE, OTHER)['ok'] is True
    real_end_run, real_append_event = kb._end_run, kb._append_event

    def boom(*a, **k):
        raise RuntimeError('bookkeeping exploded')

    monkeypatch.setattr(kb, '_end_run', boom)
    monkeypatch.setattr(kb, '_append_event', boom)
    run_task, connection, params, _calls = _run_task(tmp_path, task, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    card, _events, _owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'ready'
    assert card['claim_lock'] is None
    assert card['consecutive_failures'] == 1
    # Restore the real bookkeeping: the re-claim and the second refusal must run
    # normally so the breaker (not the patch) decides the outcome.
    monkeypatch.setattr(kb, '_end_run', real_end_run)
    monkeypatch.setattr(kb, '_append_event', real_append_event)
    # Re-claim and refuse again: the breaker reads 1 + 1 = 2 and parks it.
    with closing(connect(board=BOARD)) as conn:
        reclaimed = kb.claim_task(conn, task.id)
    assert reclaimed is not None
    run_task, connection, params, _calls = _run_task(tmp_path, reclaimed, monkeypatch)
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(connection, params))
    card, _events, _owners, _run = _board_rows(tmp_path, task.id)
    assert card['status'] == 'blocked'
    assert card['consecutive_failures'] == 2
