"""Recovery triage must retain the original work item, not create a new graph."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli.kanban_db_graph import decompose_triage_task


CHILDREN = [{"title": f"replacement {i}", "assignee": "other"} for i in range(4)]


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with kbc.connect_closing() as conn:
        yield conn


def _work(conn, *, triage=False):
    return kb.create_task(
        conn, title="original work", body="Keep the reviewed source and test evidence.",
        assignee="worker", tenant="project", priority=3, triage=triage,
    )


def _repeat_block(conn, tid, *, with_run=True):
    for attempt in range(kb.BLOCK_RECURRENCE_LIMIT):
        if attempt:
            assert kb.unblock_task(conn, tid)
        if with_run:
            assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.block_task(
            conn, tid, reason="source and tests already verified" if with_run else None,
            kind="capability",
        )
    assert kb.get_task(conn, tid).status == "triage"
    assert any(e.kind == "block_loop_detected" for e in kb.list_events(conn, tid))
    assert bool(kb.list_runs(conn, tid)) == with_run


def _snapshot(conn):
    return {
        table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("tasks", "task_links", "task_runs", "task_events", "task_comments")
    }


@pytest.mark.parametrize("with_run", [True, False])
def test_native_fanout_refuses_repeat_block_triage(board, with_run):
    tid = _work(board)
    _repeat_block(board, tid, with_run=with_run)
    before = _snapshot(board)

    children = decompose_triage_task(
        board, tid, root_assignee="other", children=CHILDREN, author="decomposer",
    )

    assert children is None, "recovery triage must not create four replacement cards"
    assert _snapshot(board) == before


@pytest.mark.parametrize("with_run", [True, False])
def test_recovery_does_not_call_aux_or_overwrite_original(board, monkeypatch, with_run):
    tid = _work(board)
    _repeat_block(board, tid, with_run=with_run)
    before = _snapshot(board)
    aux = Mock(return_value=(json.dumps({"fanout": True, "tasks": CHILDREN}), ""))
    monkeypatch.setattr(decomp, "_call_aux", aux)

    outcome = decomp.decompose_task(tid)

    aux.assert_not_called()
    assert not outcome.ok
    assert _snapshot(board) == before


def test_auto_tick_selects_fresh_intake_not_recovery(board, monkeypatch):
    from types import SimpleNamespace
    from gateway import kanban_watchers_dispatcher as dispatcher

    recovery = _work(board)
    _repeat_block(board, recovery)
    never_claimed = _work(board)
    _repeat_block(board, never_claimed, with_run=False)
    fresh = _work(board, triage=True)
    recovery_before = kb.get_task(board, recovery)
    aux = Mock(return_value=(json.dumps({"fanout": True, "tasks": CHILDREN}), ""))
    monkeypatch.setattr(decomp, "_call_aux", aux)
    monkeypatch.setattr(dispatcher, "_board_slugs", lambda kb: [kb.DEFAULT_BOARD])
    settings = dispatcher._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    tick = dispatcher._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD=kb.DEFAULT_BOARD), settings)

    assert decomp.list_triage_ids() == [fresh]
    assert tick.auto_decompose_tick(1) == 1
    assert aux.call_count == 1
    assert aux.call_args.args[1] == fresh
    assert kb.get_task(board, recovery) == recovery_before
    assert kb.get_task(board, never_claimed).status == "triage"
    assert len(kb.list_tasks(board)) == 7  # three originals and four legitimate children
    assert tick.auto_decompose_tick(1) == 0
    assert aux.call_count == 1


@pytest.mark.parametrize("fanout", [True, False])
@pytest.mark.parametrize("with_run", [True, False])
def test_history_changed_during_aux_is_rechecked_before_any_write(board, monkeypatch, fanout, with_run):
    tid = _work(board, triage=True)
    after_history = {}

    def reply_after_recovery(*args, **kwargs):
        # Another connection accepts and attempts the original card while aux
        # is in flight, then the real block-loop lifecycle sends it to triage.
        with kbc.connect_closing() as other:
            assert kb.specify_triage_task(
                other, tid, title="reviewed original", body="retained source and evidence",
                assignee="worker", author="operator",
            )
            _repeat_block(other, tid, with_run=with_run)
            after_history.update(_snapshot(other))
        return json.dumps({
            "fanout": fanout, "tasks": CHILDREN,
            "title": "obsolete spec", "body": "discard original evidence", "assignee": "other",
        }), ""

    aux = Mock(side_effect=reply_after_recovery)
    monkeypatch.setattr(decomp, "_call_aux", aux)

    outcome = decomp.decompose_task(tid)

    assert not outcome.ok, "late aux reply must neither fan out nor overwrite the original spec"
    assert aux.call_count == 1
    assert _snapshot(board) == after_history
    assert not board.in_transaction


def test_explicit_specify_still_edits_recovery_on_original_card(board):
    tid = _work(board)
    _repeat_block(board, tid)
    runs = kb.list_runs(board, tid)

    assert kb.specify_triage_task(
        board, tid, title="explicit revision", body="revised acceptance", author="operator",
    )
    task = kb.get_task(board, tid)
    assert task is not None
    assert (task.title, task.body, task.assignee) == ("explicit revision", "revised acceptance", "worker")
    assert kb.list_runs(board, tid) == runs
    assert len(kb.list_tasks(board)) == 1


@pytest.mark.parametrize("fanout", [True, False])
def test_fresh_intake_keeps_ordinary_dependencies(board, monkeypatch, fanout):
    prerequisite = kb.create_task(board, title="true prerequisite", tenant="project")
    tid = kb.create_task(
        board, title="fresh idea", triage=True, parents=[prerequisite], tenant="project",
        assignee="worker",
    )
    downstream = kb.create_task(board, title="downstream", parents=[tid], tenant="project")
    aux = Mock(return_value=(json.dumps({
        "fanout": fanout, "tasks": CHILDREN, "title": "concrete spec", "body": "acceptance",
    }), ""))
    monkeypatch.setattr(decomp, "_call_aux", aux)
    assert tid in decomp.list_triage_ids(tenant="project")
    assert tid not in decomp.list_triage_ids(tenant="other-project")

    outcome = decomp.decompose_task(tid)

    assert outcome.ok, outcome.reason
    assert aux.call_count == 1
    assert kb.get_task(board, tid).status == "todo"
    assert board.execute(
        "SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?", (prerequisite, tid),
    ).fetchone()
    assert board.execute(
        "SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?", (tid, downstream),
    ).fetchone()
    if fanout:
        assert outcome.child_ids and len(outcome.child_ids) == 4
        assert all(kb.get_task(board, cid).tenant == "project" for cid in outcome.child_ids)
    else:
        assert kb.get_task(board, tid).assignee == "worker"
        assert len(kb.list_tasks(board)) == 3


def test_native_concurrent_calls_and_retriage_after_unlink_keep_one_graph(board, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    tid = _work(board, triage=True)
    barrier = Barrier(2)

    def fanout():
        with kbc.connect_closing() as conn:
            barrier.wait(timeout=10)
            return decompose_triage_task(conn, tid, root_assignee="worker", children=CHILDREN)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: fanout(), range(2)))
    winners = [ids for ids in results if ids is not None]
    assert len(winners) == 1
    assert len(winners[0]) == 4
    before = _snapshot(board)
    assert decompose_triage_task(board, tid, root_assignee="other", children=CHILDREN) is None
    assert _snapshot(board) == before
    for cid in winners[0]:
        assert kb.unlink_tasks(board, cid, tid)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
    before = _snapshot(board)
    aux = Mock()
    monkeypatch.setattr(decomp, "_call_aux", aux)

    assert tid not in decomp.list_triage_ids()
    assert not decomp.decompose_task(tid).ok
    assert decompose_triage_task(board, tid, root_assignee="other", children=CHILDREN) is None
    aux.assert_not_called()
    assert _snapshot(board) == before
    assert len(kb.list_tasks(board)) == 5
    assert sum(e.kind == "decomposed" for e in kb.list_events(board, tid)) == 1
    assert board.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("fanouts", [(True, True), (False, False), (True, False)])
def test_concurrent_aux_replies_commit_only_one_result(board, monkeypatch, fanouts):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, local

    tid = _work(board, triage=True)
    barrier = Barrier(2)
    reply = local()

    def aux_reply(*args, **kwargs):
        barrier.wait(timeout=10)
        return json.dumps({
            "fanout": reply.fanout, "tasks": CHILDREN, "title": "single spec", "body": "acceptance",
        }), ""

    aux = Mock(side_effect=aux_reply)
    monkeypatch.setattr(decomp, "_call_aux", aux)

    def decompose(fanout):
        reply.fanout = fanout
        return decomp.decompose_task(tid)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(decompose, fanouts))
    winners = [result for result in results if result.ok]
    assert len(winners) == 1
    assert aux.call_count == 2
    assert len(kb.list_tasks(board)) == (5 if winners[0].fanout else 1)
    assert sum(e.kind in {"decomposed", "specified"} for e in kb.list_events(board, tid)) == 1
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
    before = _snapshot(board)
    assert not decomp.decompose_task(tid).ok
    assert aux.call_count == 2
    assert _snapshot(board) == before


@pytest.mark.parametrize("history", ["run", "block_loop_event"])
def test_durable_history_still_blocks_after_display_fields_are_reset(board, monkeypatch, history):
    tid = _work(board)
    _repeat_block(board, tid, with_run=history == "run")
    with kb.write_txn(board):
        board.execute(
            "UPDATE tasks SET started_at=NULL, current_run_id=NULL, block_kind=NULL, "
            "block_recurrences=0 WHERE id=?", (tid,),
        )
        if history == "run":
            board.execute("DELETE FROM task_events WHERE task_id=?", (tid,))
    before = _snapshot(board)
    aux = Mock()
    monkeypatch.setattr(decomp, "_call_aux", aux)

    assert tid not in decomp.list_triage_ids()
    assert not decomp.decompose_task(tid).ok
    assert decompose_triage_task(board, tid, root_assignee="other", children=CHILDREN) is None
    assert not kb.specify_triage_task(board, tid, title="late spec", require_fresh_triage=True)
    aux.assert_not_called()
    assert _snapshot(board) == before
