"""#47 Phase 2 C4 — a finished or lost attempt RELEASES its cross-session claim.

The ledger claim is what keeps two live sessions off the same logical work
item; it must END when the attempt ends. Without a release the row stays
ACTIVE until its 900 s TTL lapses, so the next session (or a Direct dispatch of
the same issue) is locked out for up to fifteen minutes after the first worker
already finished.

Every test runs against a private ``tmp_path`` board with a REAL ownership
ledger and drives the real ``gateway/session_kanban.py`` completion paths
(never the production DB):

  (a) a completed worker result releases the claim
  (b) a failed worker result releases the claim
  (c) a recorded worker loss releases the claim
  (d) a release against a claim that is already gone never raises and never
      mutates a foreign owner / a missing row
  (e) a broken ledger cannot fail the worker's own exit booking
"""
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

BOARD = 'release'
OWNER = 'owner-session'
OTHER = 'other-session'
BODY = 'source_repo: Andrew288369/ANDREW-GRP\nsource_issue: 47'
SCOPE = 'gh:andrew288369/andrew-grp#47'


def _claim(tmp_path, monkeypatch, *, ledger_owner=OWNER, marker_session=OWNER,
           session_env=None, with_claim=True):
    """Private board: a running card, its worker_bound + owner_admitted markers,
    and (optionally) a live ledger claim owned by ``ledger_owner``.

    ``marker_session`` / ``session_env`` are the identities the release helper
    will try; they are independent of ``ledger_owner`` so a foreign-owner case
    can be seeded. Returns ``(path, context)``.
    """
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    if session_env is None:
        monkeypatch.delenv('HERMES_SESSION_ID', raising=False)
    else:
        monkeypatch.setenv('HERMES_SESSION_ID', session_env)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env',
                        lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir(exist_ok=True)
    kb.create_board(BOARD)
    path = kb.kanban_db_path(board=BOARD)
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        task_id = kb.create_task(conn, title='Release me', assignee='default',
                                 workspace_kind='dir', workspace_path=str(workspace), body=BODY)
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
        assert task is not None
        context = {'db': str(path), 'task_id': task_id,
                   'run_id': task.current_run_id, 'claim_lock': task.claim_lock}
        from cross_session_ownership import claim_ownership, init_schema
        init_schema(conn)
        if with_claim:
            assert claim_ownership(conn, SCOPE, ledger_owner, ttl_seconds=900)['ok'] is True
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, 'worker_bound',
                             {'pid': os.getpid(), 'claim_lock': task.claim_lock},
                             run_id=task.current_run_id)
            kb._append_event(conn, task_id, 'owner_admitted',
                             {'session_id': marker_session, 'scope': SCOPE,
                              'request_id': 'kanban:release'},
                             run_id=task.current_run_id)
    return path, context


def _readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _ledger(path):
    """``(row|None, events)`` for SCOPE on a read-only connection."""
    with closing(_readonly(path)) as conn:
        row = conn.execute('SELECT * FROM task_ownership WHERE scope_key=?', (SCOPE,)).fetchone()
        events = [dict(r) for r in conn.execute(
            'SELECT event, actor FROM task_ownership_events WHERE scope_key=? ORDER BY id',
            (SCOPE,))]
    return (dict(row) if row is not None else None), events


# (a) ----------------------------------------------------------------------
def test_completed_worker_result_releases_the_claim(tmp_path, monkeypatch):
    path, context = _claim(tmp_path, monkeypatch)
    session_kanban._record_worker_result(context, 0, 'done')
    row, events = _ledger(path)
    assert row is not None
    assert row['owner_session'] == OWNER
    assert row['state'] == 'RELEASED'
    # The seed claim itself logs 'claimed'; the release appends 'released'.
    assert [e['event'] for e in events] == ['claimed', 'released']
    assert events[-1]['actor'] == OWNER


# (b) ----------------------------------------------------------------------
def test_failed_worker_result_releases_the_claim(tmp_path, monkeypatch):
    path, context = _claim(tmp_path, monkeypatch)
    session_kanban._record_worker_result(context, 1, 'boom')
    row, events = _ledger(path)
    assert row['state'] == 'RELEASED'
    # The seed claim itself logs 'claimed'; the release appends 'released'.
    assert [e['event'] for e in events] == ['claimed', 'released']


def test_worker_result_releases_via_the_ambient_session_candidate(tmp_path, monkeypatch):
    """The admission marker may name a different session than the worker's own:
    HERMES_SESSION_ID is a first-class candidate, not just the marker."""
    path, context = _claim(tmp_path, monkeypatch, marker_session=OTHER, session_env=OWNER)
    session_kanban._record_worker_result(context, 0, 'done')
    row, _events = _ledger(path)
    assert row['state'] == 'RELEASED'


# (c) ----------------------------------------------------------------------
def test_recorded_worker_loss_releases_the_claim(tmp_path, monkeypatch):
    path, context = _claim(tmp_path, monkeypatch)
    policy = SimpleNamespace(kanban_json=json.dumps(context))
    session_kanban.record_worker_loss(policy, 'managed_worker_failed', pid=1234, bound=False)
    row, events = _ledger(path)
    assert row['state'] == 'RELEASED'
    # The seed claim itself logs 'claimed'; the release appends 'released'.
    assert [e['event'] for e in events] == ['claimed', 'released']


# (d) ----------------------------------------------------------------------
def test_release_never_touches_a_foreign_owner(tmp_path, monkeypatch):
    """The claim is already gone from this attempt's point of view: a foreign
    session owns it. The release is a no-op, not an error, and the foreign
    owner is untouched."""
    path, context = _claim(tmp_path, monkeypatch, ledger_owner=OTHER, marker_session=OWNER,
                           session_env=OWNER)
    session_kanban._record_worker_result(context, 0, 'done')  # must not raise
    row, events = _ledger(path)
    assert row['owner_session'] == OTHER
    assert row['state'] == 'ACTIVE'
    # Only the seed claim exists; the no-op release added no 'released' event
    # and the foreign owner is untouched.
    assert [e['event'] for e in events] == ['claimed']
    assert events[0]['actor'] == OTHER


def test_release_is_a_no_op_when_no_claim_exists(tmp_path, monkeypatch):
    """``no_owner``: the ledger row is already gone (released/never claimed)."""
    path, context = _claim(tmp_path, monkeypatch, with_claim=False)
    assert session_kanban._release_ownership_claim(
        _readonly_conn(path), context, reason='probe') == [
        {'ok': False, 'reason': 'no_owner', 'scope': SCOPE}]
    row, events = _ledger(path)
    assert row is None and events == []


def _readonly_conn(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# (e) ----------------------------------------------------------------------
def test_a_broken_ledger_does_not_fail_the_worker_exit_booking(tmp_path, monkeypatch):
    """Release is best-effort: a raising ledger must not fail (or roll back) the
    worker's own ``worker_result`` booking."""
    path, context = _claim(tmp_path, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError('ledger on fire')

    monkeypatch.setattr('cross_session_ownership.release_ownership', boom)
    session_kanban._record_worker_result(context, 0, 'done')  # must not raise
    booked = session_kanban.worker_result(path, context)
    assert booked.get('exit_code') == 0 and booked.get('claim_lock') == context['claim_lock']