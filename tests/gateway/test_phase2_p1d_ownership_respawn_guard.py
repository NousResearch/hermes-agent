"""#47 Phase 2 P1-D — an ownership refusal must be SPACED, never parked.

The admission gate (``gateway/session_kanban.py``) releases a refused card with
``last_failure_error='ownership_refused'``. Without the dispatcher-side half of
the fix that text is caught by the quota/auth blocker pattern, which parks the
card forever even though the ledger still owns the scope — retrying is futile
AND forbidden. P1-D pins the three moving parts:

  (a) INCREMENT — ``_release_refused_claim`` bumps ``consecutive_failures``
      SQL-side, so the breaker (which reads old + 1) eventually stops the
      dispatcher from re-spawning a card the gate keeps refusing;
  (b) BREAKER — a card already at >=1 failure parks ``blocked``, not ``ready``;
  (c) REGEX — ``_RESPAWN_BLOCKER_RE`` matches the ``ownership_refused`` /
      ``ownership refusal`` signature (and stays away from benign prose);
  (d) COOLDOWN — ``check_respawn_guard`` returns ``ownership_refusal_cooldown``
      for a fresh refusal, ``None`` once the cooldown elapses or is disabled.

Every test runs against a private ``tmp_path`` board; the production DB is never
touched.
"""
import sqlite3
import time
from contextlib import closing

import pytest

BOARD = 'p1d'
SCOPE = 'gh:andrew288369/andrew-grp#47'


def _prepare(tmp_path, monkeypatch, *, consecutive_failures=0, max_retries=None):
    """Private board + a claimed (running) card. Returns the claimed task row."""
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
        task_id = kb.create_task(conn, title='P1-D', assignee='default', workspace_kind='dir',
                                 workspace_path=str(workspace),
                                 body='source_repo: Andrew288369/ANDREW-GRP\nsource_issue: 47',
                                 max_retries=max_retries)
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
        if consecutive_failures:
            conn.execute('UPDATE tasks SET consecutive_failures=? WHERE id=?',
                         (consecutive_failures, task_id))
    return task


def _refuse(tmp_path, task):
    """Book a REAL refusal through the production helper, in one write txn."""
    from gateway.session_kanban import _release_refused_claim
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        with kb.write_txn(conn):
            row = conn.execute('SELECT * FROM tasks WHERE id=?', (task.id,)).fetchone()
            _release_refused_claim(conn, task, row, refused='owned_by_other_session',
                                   scope=SCOPE, scope_reason=None)


def _card(tmp_path, task_id):
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()


def _latest_run(tmp_path, task_id):
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            'SELECT * FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1',
            (task_id,)).fetchone()


# (a) ----------------------------------------------------------------------
def test_a_refusal_increments_the_failure_counter(tmp_path, monkeypatch):
    task = _prepare(tmp_path, monkeypatch)
    _refuse(tmp_path, task)
    card, run = _card(tmp_path, task.id), _latest_run(tmp_path, task.id)
    # 0 -> 1, written SQL-side so a concurrent writer cannot lose the step.
    assert card['consecutive_failures'] == 1
    assert card['status'] == 'ready'
    assert card['claim_lock'] is None and card['current_run_id'] is None
    assert card['last_failure_error'] == 'ownership_refused'
    assert run['outcome'] == 'refused' and run['error'] == 'ownership_refused'
    assert run['ended_at'] is not None


# (b) ----------------------------------------------------------------------
def test_b_card_already_at_one_failure_parks_blocked(tmp_path, monkeypatch):
    task = _prepare(tmp_path, monkeypatch, consecutive_failures=1)
    _refuse(tmp_path, task)
    card = _card(tmp_path, task.id)
    # old (1) + 1 = 2 >= 2 -> the breaker parks the card instead of re-queueing it.
    assert card['consecutive_failures'] == 2
    assert card['status'] == 'blocked'


# (c) ----------------------------------------------------------------------
@pytest.mark.parametrize('text', [
    'ownership_refused',
    'ownership refusal',
    'OWNERSHIP-REFUSAL',
    'ownership  refusal',
])
def test_c_blocker_regex_matches_the_refusal_signature(text):
    from hermes_cli.kanban_db_dispatch import _RESPAWN_BLOCKER_RE
    assert _RESPAWN_BLOCKER_RE.search(text), text


@pytest.mark.parametrize('text', [
    'ownership model is healthy',
    'the worker refused a coffee',   # bare "refused" is not the refusal signature
    'card completed successfully',
])
def test_c2_blocker_regex_ignores_benign_prose(text):
    from hermes_cli.kanban_db_dispatch import _RESPAWN_BLOCKER_RE
    assert not _RESPAWN_BLOCKER_RE.search(text), text


# (d) ----------------------------------------------------------------------
def test_d_fresh_refusal_is_spaced_not_parked(tmp_path, monkeypatch):
    from hermes_cli.kanban_db_dispatch import check_respawn_guard
    from hermes_cli.kanban_db_connect import connect
    task = _prepare(tmp_path, monkeypatch)
    _refuse(tmp_path, task)
    with closing(connect(board=BOARD)) as conn:
        assert check_respawn_guard(conn, task.id) == 'ownership_refusal_cooldown'


def test_d2_elapsed_cooldown_lets_the_card_retry(tmp_path, monkeypatch):
    from hermes_cli.kanban_db_dispatch import check_respawn_guard
    from hermes_cli.kanban_db_connect import connect
    task = _prepare(tmp_path, monkeypatch)
    _refuse(tmp_path, task)
    run = _latest_run(tmp_path, task.id)
    with closing(connect(board=BOARD)) as conn:
        conn.execute('UPDATE task_runs SET ended_at=? WHERE id=?',
                     (int(time.time()) - 400, run['id']))
        assert check_respawn_guard(conn, task.id) is None


def test_d3_disabled_cooldown_respawns_on_the_next_tick(tmp_path, monkeypatch):
    from hermes_cli.kanban_db_dispatch import check_respawn_guard
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_REFUSAL_COOLDOWN_SECONDS', '0')
    task = _prepare(tmp_path, monkeypatch)
    _refuse(tmp_path, task)
    with closing(connect(board=BOARD)) as conn:
        assert check_respawn_guard(conn, task.id) is None


def test_d4_env_override_is_honoured(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    assert kb._resolve_ownership_refusal_cooldown_seconds() == 300
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_REFUSAL_COOLDOWN_SECONDS', '42')
    assert kb._resolve_ownership_refusal_cooldown_seconds() == 42
    # Unparseable falls back to the default (never to 0 / never raises).
    monkeypatch.setenv('HERMES_KANBAN_OWNERSHIP_REFUSAL_COOLDOWN_SECONDS', 'not-an-int')
    assert kb._resolve_ownership_refusal_cooldown_seconds() == 300


# --- C3: an ownership refusal is STICKY across recompute_ready --------------
#
# ``sqlite3.Connection`` is immutable, so ``execute`` cannot be monkeypatched;
# these tests drive the real ``_release_refused_claim`` and ``recompute_ready``
# against a private ``tmp_path`` board instead.
def _status(tmp_path, task_id):
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute(
            'SELECT status, consecutive_failures, max_retries FROM tasks WHERE id=?',
            (task_id,)).fetchone())


def _sticky(tmp_path, task_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        return kb._has_sticky_block(conn, task_id)


def _recompute(tmp_path, *, failure_limit=None):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        return kb.recompute_ready(conn, failure_limit=failure_limit)


def _unblock(tmp_path, task_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        return kb.unblock_task(conn, task_id)


def test_e_sticky_refusal_survives_recompute_ready(tmp_path, monkeypatch):
    """A twice-refused card stays ``blocked``: ``recompute_ready`` must not
    promote it back to ``ready`` in the same tick."""
    task = _prepare(tmp_path, monkeypatch)
    _refuse(tmp_path, task)
    _refuse(tmp_path, task)
    st = _status(tmp_path, task.id)
    assert st['status'] == 'blocked' and st['consecutive_failures'] == 2
    assert _sticky(tmp_path, task.id) is True
    assert _recompute(tmp_path) == 0
    assert _status(tmp_path, task.id)['status'] == 'blocked'


def test_e2_sticky_ignores_a_raised_max_retries(tmp_path, monkeypatch):
    """The sticky marker, not the counter, holds the card.

    With ``max_retries=5`` the counter is 2 < 5, so without C3
    ``recompute_ready`` would promote the card back to ``ready`` and the
    dispatcher would respawn it every tick against a ledger-owned scope.
    """
    task = _prepare(tmp_path, monkeypatch, max_retries=5)
    _refuse(tmp_path, task)
    _refuse(tmp_path, task)
    st = _status(tmp_path, task.id)
    assert st['status'] == 'blocked' and st['consecutive_failures'] == 2
    assert _sticky(tmp_path, task.id) is True
    assert _recompute(tmp_path) == 0
    assert _status(tmp_path, task.id)['status'] == 'blocked'


def test_e3_explicit_unblocked_clears_the_sticky_marker(tmp_path, monkeypatch):
    """An operator's ``unblock`` is the escape hatch: the newest ``unblocked``
    event clears the sticky marker, so a re-parked card is promoted again."""
    task = _prepare(tmp_path, monkeypatch, max_retries=5)
    _refuse(tmp_path, task)
    _refuse(tmp_path, task)
    assert _sticky(tmp_path, task.id) is True
    assert _unblock(tmp_path, task.id) is True
    assert _sticky(tmp_path, task.id) is False
    # Re-park the card WITHOUT a fresh refusal: the unblocked marker must let
    # recompute_ready release it (the unblock reset the counter to 0).
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board=BOARD)) as conn:
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task.id,))
    assert _recompute(tmp_path) == 1
    assert _status(tmp_path, task.id)['status'] == 'ready'


def test_e4_refusal_after_unblocked_is_sticky_again(tmp_path, monkeypatch):
    """The newest ``ownership_refused`` AFTER the newest ``unblocked`` re-arms
    the sticky marker — a fresh refusal is not cleared by an older release."""
    task = _prepare(tmp_path, monkeypatch, max_retries=5)
    _refuse(tmp_path, task)
    _refuse(tmp_path, task)
    assert _unblock(tmp_path, task.id) is True
    assert _sticky(tmp_path, task.id) is False
    _refuse(tmp_path, task)
    assert _sticky(tmp_path, task.id) is True
