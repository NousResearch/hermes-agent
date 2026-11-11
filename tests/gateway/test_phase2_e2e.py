"""#47 Phase 2 — C6 END-TO-END tests on the PRODUCTION path.

Unlike the unit/ledger suites, every scenario here drives the REAL
``gateway/session_kanban.py::run_task`` admission gate and the REAL
``gateway/ownership_gate.py`` pre/post tool-call hooks against a private
``tmp_path`` board. The production DB is never opened at all.

Scenarios (L0 C6 matrix)
------------------------
  E1  Two ``run_task`` admissions race ONE card -> exactly 1 winner, 0 double
      execute, one ``owner_admitted`` marker, one ACTIVE ledger owner.
  E2  Cross-mode conflict through the real gate, both directions:
      a Direct hook claim blocks the Kanban card, and a Kanban admission
      blocks the Direct hook.
  E3  A batch whose top-level goal is FREE but whose ``tasks[]`` hides a
      conflicting issue is blocked by the real hook (no nested bypass), and
      the free scope is NOT claimed (all-or-nothing).
  E4  The real worker heartbeat keeps an admitted claim alive past its TTL;
      the real completion paths release it so the next session can claim.
  E5  Two Direct (Delegate Profile) sessions dispatching the SAME logical work
      are fenced by the real hook layer: the first claims the scope, the second
      is blocked and claims nothing, a success post keeps the fence, and the
      failure-release path frees the scope so the second session is admitted.

Production safety: every write lands in ``tmp_path``. The ownership ledger is
the private board DB itself — the same file the admission gate, the Direct
hook and the worker heartbeat all resolve — so a cross-mode collision is a
real collision, not two independent fakes.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from cross_session_ownership import (
    STATE_ACTIVE,
    STATE_RELEASED,
    VERDICT_FREE,
    VERDICT_OWNED,
    VERDICT_STALE,
    claim_ownership,
    init_schema,
    inspect_ownership,
    list_ownership,
)
from gateway import ownership_gate as gate
from gateway import session_kanban
from hermes_state_runtime import RuntimeStoreError

# R1 (#47 Phase 2): import the recovery module at COLLECTION time, before any
# test patches ``hermes_state_local.local_receipt`` (see test_kanban_finite_admission).
import gateway.session_local_recovery as _slr

BOARD = 'e2e'
REPO_OWNER = 'Andrew288369'
REPO_NAME = 'ANDREW-GRP'
BODY = 'source_repo: Andrew288369/ANDREW-GRP\nsource_issue: 33'
SCOPE = 'gh:andrew288369/andrew-grp#33'
SCOPE_39 = 'gh:andrew288369/andrew-grp#39'
SCOPE_47 = 'gh:andrew288369/andrew-grp#47'


# ---------------------------------------------------------------------------
# board / connection helpers
# ---------------------------------------------------------------------------
def _prepare_board(tmp_path, monkeypatch, *, body=BODY, consecutive_failures=0):
    """Private board + a claimed (running) card. Returns ``(workspace, task, path)``.

    The ledger schema is pre-created so a concurrent admission race never has
    to issue DDL inside the board's write lock.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    # The Direct hook verifies the claimed session_id against the AMBIENT
    # session identity; inherit the runner's own HERMES_SESSION_ID and every
    # positive dispatch case would fail closed for the wrong reason.
    monkeypatch.delenv('HERMES_SESSION_ID', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env',
                        lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir(exist_ok=True)
    kb.create_board(BOARD)
    path = kb.kanban_db_path(board=BOARD)
    # Pin the ownership ledger to THIS board DB: the Direct hook resolves the
    # ledger through ``kb.kanban_db_path()`` (no board arg), so the hook and
    # the admission gate must land on the same file.
    monkeypatch.setenv('HERMES_KANBAN_DB', str(path))
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        init_schema(conn)
        task_id = kb.create_task(conn, title='E2E owned', assignee='default',
                                 workspace_kind='dir', workspace_path=str(workspace), body=body)
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
        if consecutive_failures:
            conn.execute('UPDATE tasks SET consecutive_failures=? WHERE id=?',
                         (consecutive_failures, task_id))
            task = kb.get_task(conn, task_id)
    return workspace, task, path


def _connection(tmp_path, monkeypatch, *, subject, calls):
    """A ``run_task`` connection whose session layer is mocked per subject."""
    from gateway.session_contract import Principal

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
    actor = Principal(subject, str(tmp_path), frozenset({'session:create'}), 'transport')
    return SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)


def _wire_session_layer(monkeypatch, tmp_path, session_by_subject):
    """Mock only the session-creation boundary; the admission path stays real."""
    monkeypatch.setattr('gateway.run._load_gateway_config',
                        lambda: {'model': {'default': 'loop-model', 'provider': 'custom'},
                                 'platform_toolsets': {'cli': ['terminal']}})
    monkeypatch.setattr('gateway.session_kanban.asdict', lambda receipt: {})

    def _create_session(authority, actor, meta, **kwargs):
        return SimpleNamespace(session_id=session_by_subject[actor.subject],
                               profile_id=str(tmp_path))

    monkeypatch.setattr('gateway.session_local.create_local_session', _create_session)


def _params(task):
    return dict(board=BOARD, task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)


def _context(path, task):
    """The worker context ``bind_worker_context`` would set for this attempt."""
    return {'db': str(path), 'task_id': task.id, 'run_id': task.current_run_id,
            'claim_lock': task.claim_lock,
            'skills': list(getattr(task, 'skills', None) or ()),
            'context': '', 'goal_mode': bool(getattr(task, 'goal_mode', False)),
            'goal_text': '\n\n'.join(p for p in (task.title, task.body) if p),
            'goal_max_turns': getattr(task, 'goal_max_turns', None) or 0}


# ---------------------------------------------------------------------------
# read helpers
# ---------------------------------------------------------------------------
def _open(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _inspect(path, scope):
    with closing(_open(path)) as conn:
        return inspect_ownership(conn, scope)


def _claim(path, scope, session_id):
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        return claim_ownership(conn, scope, session_id, ttl_seconds=900)
    finally:
        conn.close()


def _board_rows(path, task_id):
    with closing(_open(path)) as conn:
        task = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        events = [dict(r) for r in conn.execute(
            'SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id', (task_id,))]
        owners = [dict(r) for r in conn.execute('SELECT * FROM task_ownership')]
        run = conn.execute('SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1',
                           (task_id,)).fetchone()
    return task, events, owners, run


def _shorten_leases(path, context, *, ttl):
    """Simulate a short-TTL claim: both leases expire ``ttl`` seconds from now."""
    from hermes_cli.kanban_db_connect import connect
    expires = int(time.time()) + ttl
    with closing(connect(board=BOARD)) as conn:
        conn.execute('UPDATE tasks SET claim_expires=? WHERE id=?', (expires, context['task_id']))
        conn.execute('UPDATE task_runs SET claim_expires=? WHERE id=?', (expires, context['run_id']))
        conn.execute('UPDATE task_ownership SET claim_expires=? WHERE scope_key=?',
                     (expires, SCOPE))


def _expire_leases(path, context):
    """Simulate a worker that outlived its TTL with no beat at all."""
    from hermes_cli.kanban_db_connect import connect
    past = int(time.time()) - 1
    with closing(connect(board=BOARD)) as conn:
        conn.execute('UPDATE tasks SET claim_expires=? WHERE id=?', (past, context['task_id']))
        conn.execute('UPDATE task_runs SET claim_expires=? WHERE id=?', (past, context['run_id']))
        conn.execute('UPDATE task_ownership SET claim_expires=? WHERE scope_key=?',
                     (past, SCOPE))


def _write_worker_bound(path, context):
    """The marker ``bind_worker_context`` writes when a worker binds a run."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        with kb.write_txn(conn):
            kb._append_event(conn, context['task_id'], 'worker_bound',
                             {'pid': os.getpid(), 'claim_lock': context['claim_lock']},
                             run_id=context['run_id'])


class _FakeAgent:
    """Minimal agent for the real ``run_worker_turns`` turn loop."""

    def __init__(self, session_id, results):
        self.session_id = session_id
        self.results = list(results)
        self.calls = []
        self._session_db = self

    def get_messages_as_conversation(self, _sid):
        return [{'role': 'user', 'content': 'from-store'}]

    def run_conversation(self, message, conversation_history=None, turn_author=None, task_id=None):
        self.calls.append(message)
        return self.results.pop(0)


# ---------------------------------------------------------------------------
# E1 — two admissions race one card
# ---------------------------------------------------------------------------
def test_e1_two_admissions_race_exactly_one_winner(tmp_path, monkeypatch):
    _workspace, task, path = _prepare_board(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', '1')
    calls = []
    conn_a = _connection(tmp_path, monkeypatch, subject='owner-A', calls=calls)
    conn_b = _connection(tmp_path, monkeypatch, subject='owner-B', calls=calls)
    _wire_session_layer(monkeypatch, tmp_path, {'owner-A': 'session-A', 'owner-B': 'session-B'})
    params = _params(task)

    # Force a genuine race: both admission threads meet BEFORE either takes the
    # board write lock, so the loser cannot merely observe a committed marker.
    from hermes_cli.kanban_db_connect import connect_closing as real_connect_closing
    barrier = threading.Barrier(2)

    @contextlib.contextmanager
    def _synced_connect_closing(db_path=None, *, board=None):
        barrier.wait(timeout=20)
        with real_connect_closing(db_path, board=board) as conn:
            yield conn

    monkeypatch.setattr('hermes_cli.kanban_db_connect.connect_closing', _synced_connect_closing)

    from gateway.session_kanban import run_task

    async def _race():
        return await asyncio.gather(run_task(conn_a, params), run_task(conn_b, params),
                                    return_exceptions=True)

    results = asyncio.run(_race())

    winners = [r for r in results if isinstance(r, dict)]
    losers = [r for r in results if isinstance(r, RuntimeStoreError)]
    assert len(winners) == 1, results
    assert len(losers) == 1, results
    # The loser is refused by the admission gate, never dispatched.
    assert losers[0].args[0] == 'stale_kanban_claim', losers[0].args
    # Exactly one worker reached resume/submit: no double execution.
    assert [c[0] for c in calls] == ['resume', 'submit'], calls
    winner_session = winners[0]['session_id']
    assert winner_session in {'session-A', 'session-B'}

    card, events, owners, _run = _board_rows(path, task.id)
    assert card['status'] == 'running'
    assert card['claim_lock'] == task.claim_lock
    assert card['current_run_id'] == task.current_run_id
    assert [e['kind'] for e in events].count('owner_admitted') == 1
    marker = next(e for e in events if e['kind'] == 'owner_admitted')
    assert json.loads(marker['payload'])['session_id'] == winner_session
    # One ACTIVE ledger owner, and it is the winner.
    assert len(owners) == 1 and owners[0]['owner_session'] == winner_session
    assert owners[0]['state'] == STATE_ACTIVE
    assert owners[0]['scope_key'] == SCOPE


# ---------------------------------------------------------------------------
# E2 — cross-mode conflict through the real gate (both directions)
# ---------------------------------------------------------------------------
def _direct_args(goal, *, context='', tasks=None):
    batch = {'goal': goal, 'context': context}
    if tasks is not None:
        batch['tasks'] = tasks
    return {'batch': batch}


def test_e2a_direct_hook_claim_blocks_kanban_card(tmp_path, monkeypatch):
    _workspace, task, path = _prepare_board(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', '1')
    monkeypatch.setenv('HERMES_OWNERSHIP_DIRECT_ENFORCE', '1')
    gate.reset_state()

    # The REAL Direct hook claims the issue scope for session-D.
    pre = gate.pre_tool_call(
        tool_name='delegate_task',
        args=_direct_args(f'Work {REPO_OWNER}/{REPO_NAME}#33 per directive'),
        session_id='session-D', tool_call_id='call-e2a')
    assert pre is None, pre
    assert _inspect(path, SCOPE)['owner'] == 'session-D'

    # The Kanban card for the same logical work is refused by the REAL gate.
    calls = []
    conn = _connection(tmp_path, monkeypatch, subject='owner-K', calls=calls)
    _wire_session_layer(monkeypatch, tmp_path, {'owner-K': 'session-K'})
    from gateway.session_kanban import run_task
    with pytest.raises(RuntimeStoreError, match='ownership_refused'):
        asyncio.run(run_task(conn, _params(task)))
    assert calls == [], 'a refused admission must not reach resume/submit'

    card, events, owners, _run = _board_rows(path, task.id)
    assert card['status'] == 'ready'
    assert card['claim_lock'] is None and card['current_run_id'] is None
    assert [e['kind'] for e in events].count('owner_admitted') == 0
    # The Direct owner was not disturbed.
    assert len(owners) == 1 and owners[0]['owner_session'] == 'session-D'
    assert owners[0]['state'] == STATE_ACTIVE


def test_e2b_kanban_admission_blocks_direct_hook(tmp_path, monkeypatch):
    _workspace, task, path = _prepare_board(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', '1')
    monkeypatch.setenv('HERMES_OWNERSHIP_DIRECT_ENFORCE', '1')
    gate.reset_state()

    # The REAL admission claims the scope for the Kanban session.
    calls = []
    conn = _connection(tmp_path, monkeypatch, subject='owner-K', calls=calls)
    _wire_session_layer(monkeypatch, tmp_path, {'owner-K': 'session-K'})
    from gateway.session_kanban import run_task
    result = asyncio.run(run_task(conn, _params(task)))
    assert result['session_id'] == 'session-K'
    assert _inspect(path, SCOPE)['owner'] == 'session-K'

    # The REAL Direct hook for the same issue is blocked.
    pre = gate.pre_tool_call(
        tool_name='delegate_task',
        args=_direct_args(f'Work {REPO_OWNER}/{REPO_NAME}#33 per directive'),
        session_id='session-D', tool_call_id='call-e2b')
    assert isinstance(pre, dict) and pre.get('action') == 'block', pre
    assert SCOPE in pre['message']
    # The Kanban owner is untouched; the blocked call recorded nothing.
    assert _inspect(path, SCOPE)['owner'] == 'session-K'
    assert gate.claimed_scopes('call-e2b') == []
    # A post-hook failure release for the blocked call is a no-op.
    gate.post_tool_call(tool_name='delegate_task',
                        args=_direct_args(f'Work {REPO_OWNER}/{REPO_NAME}#33'),
                        session_id='session-D', tool_call_id='call-e2b',
                        status='error', error_type='tool_error')
    assert _inspect(path, SCOPE)['owner'] == 'session-K'


# ---------------------------------------------------------------------------
# E3 — batch tasks[] cannot bypass the real hook
# ---------------------------------------------------------------------------
def test_e3_batch_nested_goal_cannot_bypass_the_hook(tmp_path, monkeypatch):
    _workspace, _task, path = _prepare_board(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_OWNERSHIP_DIRECT_ENFORCE', '1')
    gate.reset_state()

    # Seed a live foreign owner on #33 through the REAL hook.
    assert gate.pre_tool_call(
        tool_name='delegate_task',
        args=_direct_args(f'Work {REPO_OWNER}/{REPO_NAME}#33'),
        session_id='session-other', tool_call_id='call-e3-seed') is None

    # (i) top-level goal is FREE (#47); the conflict (#33) hides in tasks[].
    batch_nested = _direct_args(
        f'Work {REPO_OWNER}/{REPO_NAME}#47 per directive',
        tasks=[
            {'goal': f'Review {REPO_OWNER}/{REPO_NAME}#39', 'context': ''},
            {'goal': f'Implement {REPO_OWNER}/{REPO_NAME}#33', 'context': ''},
        ],
    )
    pre = gate.pre_tool_call(tool_name='delegate_task', args=batch_nested,
                             session_id='session-batch', tool_call_id='call-e3-nested')
    assert isinstance(pre, dict) and pre.get('action') == 'block', pre
    assert SCOPE in pre['message']
    # all-or-nothing: neither the free #47 nor the free #39 was claimed.
    assert _inspect(path, SCOPE_47)['verdict'] == VERDICT_FREE
    assert _inspect(path, SCOPE_39)['verdict'] == VERDICT_FREE
    assert gate.claimed_scopes('call-e3-nested') == []

    # (ii) top-level goal conflicts (#33); tasks[] carries a free #39.
    batch_toplevel = _direct_args(
        f'Work {REPO_OWNER}/{REPO_NAME}#33 per directive',
        tasks=[{'goal': f'Also cover {REPO_OWNER}/{REPO_NAME}#39', 'context': ''}],
    )
    pre2 = gate.pre_tool_call(tool_name='delegate_task', args=batch_toplevel,
                              session_id='session-batch', tool_call_id='call-e3-top')
    assert isinstance(pre2, dict) and pre2.get('action') == 'block', pre2
    assert SCOPE in pre2['message']
    assert _inspect(path, SCOPE_39)['verdict'] == VERDICT_FREE

    # The foreign owner is untouched and no partial claim leaked.
    with closing(_open(path)) as conn:
        rows = list_ownership(conn, active_only=True)
    assert len(rows) == 1 and rows[0]['owner_session'] == 'session-other'
    assert rows[0]['scope_key'] == SCOPE


# ---------------------------------------------------------------------------
# E4 — heartbeat keeps the claim alive; completion releases it
# ---------------------------------------------------------------------------
def test_e4a_heartbeat_keeps_admitted_claim_alive_past_ttl(tmp_path, monkeypatch):
    _workspace, task, path = _prepare_board(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', '1')
    calls = []
    conn = _connection(tmp_path, monkeypatch, subject='owner-H', calls=calls)
    _wire_session_layer(monkeypatch, tmp_path, {'owner-H': 'session-H'})
    from gateway.session_kanban import run_task
    result = asyncio.run(run_task(conn, _params(task)))
    assert result['session_id'] == 'session-H'
    context = _context(path, task)

    # Shrink both leases to a 2 s TTL (the dispatcher's real TTL is 900 s).
    _shorten_leases(path, context, ttl=2)
    # The REAL per-turn heartbeat thread beats both leases.
    stop = session_kanban._start_claim_heartbeat(context, interval_seconds=0.05, ttl_seconds=2)
    try:
        time.sleep(2.5)  # well past the 2 s TTL, with no tool activity at all
        ins = _inspect(path, SCOPE)
        assert ins['verdict'] == VERDICT_OWNED, ins
        assert ins['owner'] == 'session-H'
        assert ins['seconds_to_expiry'] >= 1, ins
        # A second session cannot take over a beating worker.
        assert _claim(path, SCOPE, 'session-intruder')['ok'] is False
    finally:
        stop.set()


def test_e4b_without_heartbeat_the_claim_goes_stale(tmp_path, monkeypatch):
    """Control: the same short TTL with NO heartbeat is lost (the bug C3 fixes)."""
    _workspace, task, path = _prepare_board(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', '1')
    calls = []
    conn = _connection(tmp_path, monkeypatch, subject='owner-H', calls=calls)
    _wire_session_layer(monkeypatch, tmp_path, {'owner-H': 'session-H'})
    from gateway.session_kanban import run_task
    asyncio.run(run_task(conn, _params(task)))
    context = _context(path, task)
    _expire_leases(path, context)
    ins = _inspect(path, SCOPE)
    assert ins['verdict'] == VERDICT_STALE, ins
    # A second session takes over the stale claim.
    assert _claim(path, SCOPE, 'session-intruder')['ok'] is True


def test_e4c_release_on_completion_frees_the_scope(tmp_path, monkeypatch):
    _workspace, task, path = _prepare_board(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', '1')
    calls = []
    conn = _connection(tmp_path, monkeypatch, subject='owner-R', calls=calls)
    _wire_session_layer(monkeypatch, tmp_path, {'owner-R': 'session-R'})
    from gateway.session_kanban import run_task
    result = asyncio.run(run_task(conn, _params(task)))
    assert result['session_id'] == 'session-R'
    context = _context(path, task)

    # The worker binds to the attempt (the real bind writes this marker), then
    # the REAL completion path runs.
    _write_worker_bound(path, context)
    session_kanban._record_worker_result(context, 0, 'done')

    ins = _inspect(path, SCOPE)
    assert ins['verdict'] == VERDICT_FREE, ins
    assert ins['state'] == STATE_RELEASED, ins
    # The next session can now claim the freed scope.
    assert _claim(path, SCOPE, 'session-next')['ok'] is True


def test_e4d_full_worker_turn_releases_the_scope(tmp_path, monkeypatch):
    """The REAL ``run_worker_turns`` loop (agent turn -> finally -> release)."""
    _workspace, task, path = _prepare_board(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', '1')
    calls = []
    conn = _connection(tmp_path, monkeypatch, subject='owner-T', calls=calls)
    _wire_session_layer(monkeypatch, tmp_path, {'owner-T': 'session-T'})
    from gateway.session_kanban import run_task
    result = asyncio.run(run_task(conn, _params(task)))
    assert result['session_id'] == 'session-T'
    context = _context(path, task)
    _write_worker_bound(path, context)

    monkeypatch.setattr('agent.skill_commands.build_preloaded_skills_prompt',
                        lambda *a, **k: ('', [], []))
    frame = {'text': 'work the card', 'policy': {'kanban_json': json.dumps(context)}}
    agent = _FakeAgent('session-T', [{'completed': True, 'final_response': 'done'}])
    turned = session_kanban.run_worker_turns(agent, frame, [])

    assert turned['final_response'] == 'done'
    assert len(agent.calls) == 1
    ins = _inspect(path, SCOPE)
    assert ins['verdict'] == VERDICT_FREE, ins
    assert ins['state'] == STATE_RELEASED, ins
    assert _claim(path, SCOPE, 'session-next')['ok'] is True


# ---------------------------------------------------------------------------
# E5 — Direct <-> Direct: two Delegate Profile sessions, same logical work
# ---------------------------------------------------------------------------
def test_e5_direct_to_direct_same_work_is_fenced(tmp_path, monkeypatch):
    """Two Direct (Delegate Profile) sessions dispatching the SAME logical work
    are fenced by the REAL hook layer on the production ledger: D1 claims the
    scope, D2 is blocked and claims nothing, D2's failed post cannot release
    D1's claim, D1's success post keeps the fence, and D1's failure-release
    frees the scope so D2 is admitted on retry (a dynamic fence, no deadlock).
    """
    _workspace, _task, path = _prepare_board(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_OWNERSHIP_DIRECT_ENFORCE', '1')
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_ENFORCE', '1')
    gate.reset_state()

    goal = f'Work {REPO_OWNER}/{REPO_NAME}#33 per directive'

    # (i) Direct session D1 is admitted: the REAL hook claims #33 for it.
    pre_d1 = gate.pre_tool_call(tool_name='delegate_task', args=_direct_args(goal),
                                session_id='session-D1', tool_call_id='call-e5-d1')
    assert pre_d1 is None, pre_d1
    ins = _inspect(path, SCOPE)
    assert ins['verdict'] == VERDICT_OWNED and ins['owner'] == 'session-D1', ins
    assert ins['state'] == STATE_ACTIVE, ins
    assert gate.claimed_scopes('call-e5-d1') == [SCOPE]

    # (ii) Direct session D2 dispatches the SAME logical work -> blocked.
    pre_d2 = gate.pre_tool_call(tool_name='delegate_task', args=_direct_args(goal),
                                session_id='session-D2', tool_call_id='call-e5-d2')
    assert isinstance(pre_d2, dict) and pre_d2.get('action') == 'block', pre_d2
    assert SCOPE in pre_d2['message'], pre_d2
    # The blocked dispatch claimed nothing; D1's claim is undisturbed.
    assert gate.claimed_scopes('call-e5-d2') == []
    ins = _inspect(path, SCOPE)
    assert ins['verdict'] == VERDICT_OWNED and ins['owner'] == 'session-D1', ins
    with closing(_open(path)) as conn:
        rows = list_ownership(conn, active_only=True)
    assert len(rows) == 1 and rows[0]['owner_session'] == 'session-D1', rows
    assert rows[0]['scope_key'] == SCOPE, rows

    # (iii) D2's failed post is a no-op: a blocked call must not release D1.
    gate.post_tool_call(tool_name='delegate_task', args=_direct_args(goal),
                        session_id='session-D2', tool_call_id='call-e5-d2',
                        status='error', error_type='tool_error')
    assert _inspect(path, SCOPE)['owner'] == 'session-D1'

    # (iv) D1 succeeds: the fence is KEPT after the call returns.
    gate.post_tool_call(tool_name='delegate_task', args=_direct_args(goal),
                        session_id='session-D1', tool_call_id='call-e5-d1', status='ok')
    assert gate.claimed_scopes('call-e5-d1') == [SCOPE]
    pre_d2b = gate.pre_tool_call(tool_name='delegate_task', args=_direct_args(goal),
                                 session_id='session-D2', tool_call_id='call-e5-d2b')
    assert isinstance(pre_d2b, dict) and pre_d2b.get('action') == 'block', pre_d2b
    assert gate.claimed_scopes('call-e5-d2b') == []

    # (v) D1 fails: the REAL release path frees the scope -> D2 is admitted.
    gate.post_tool_call(tool_name='delegate_task', args=_direct_args(goal),
                        session_id='session-D1', tool_call_id='call-e5-d1',
                        status='error', error_type='tool_error')
    ins = _inspect(path, SCOPE)
    assert ins['verdict'] == VERDICT_FREE, ins
    assert ins['state'] == STATE_RELEASED, ins
    pre_d2c = gate.pre_tool_call(tool_name='delegate_task', args=_direct_args(goal),
                                 session_id='session-D2', tool_call_id='call-e5-d2c')
    assert pre_d2c is None, pre_d2c
    ins = _inspect(path, SCOPE)
    assert ins['verdict'] == VERDICT_OWNED and ins['owner'] == 'session-D2', ins
