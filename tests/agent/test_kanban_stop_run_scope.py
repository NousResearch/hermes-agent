from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent.kanban_stop import build_kanban_stop_nudge
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

_NARRATED_STOP = [{"role": "assistant", "content": "Verdict posted; wrapping up."}]


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for key in tuple(os.environ):
        if key.startswith("HERMES_KANBAN_") or key == "HERMES_DELEGATED_CHILD_CONTEXT":
            monkeypatch.delenv(key)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn, monkeypatch


def _become_worker(env: pytest.MonkeyPatch, task_id: str, run_id: int) -> None:
    env.setenv("HERMES_KANBAN_TASK", task_id)
    env.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def _card_in_review(conn) -> tuple[str, int]:
    tid = kb.create_task(conn, title="rework loop", assignee="implementer")
    impl = kb.claim_task(conn, tid)
    assert kb.request_review(
        conn, tid, summary="ready", reviewer="reviewer", expected_run_id=impl.current_run_id,
    )
    review = kb.claim_review_task(conn, tid)
    assert review is not None
    return tid, review.current_run_id


def _snapshot(conn) -> list[str]:
    return list(conn.iterdump())


def test_reviewer_run_is_silent_after_changes_requested_and_immediate_reclaim(board):
    conn, env = board
    tid, run_a = _card_in_review(conn)
    _become_worker(env, tid, run_a)
    assert build_kanban_stop_nudge(messages=_NARRATED_STOP) is not None

    ok, _ = kb.request_changes(conn, tid, reason="fix it", expected_run_id=run_a)
    assert ok
    run_b = kb.claim_task(conn, tid).current_run_id
    assert run_b != run_a
    assert kb.get_task(conn, tid).status == "running"

    before = _snapshot(conn)
    assert build_kanban_stop_nudge(messages=_NARRATED_STOP) is None
    assert _snapshot(conn) == before


def test_implementer_run_is_silent_after_review_requested_and_immediate_reviewer_claim(board):
    conn, env = board
    tid = kb.create_task(conn, title="impl", assignee="implementer")
    run_a = kb.claim_task(conn, tid).current_run_id
    _become_worker(env, tid, run_a)
    assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer", expected_run_id=run_a)
    run_b = kb.claim_review_task(conn, tid).current_run_id
    assert run_b != run_a and kb.get_task(conn, tid).status == "running"

    assert build_kanban_stop_nudge(messages=_NARRATED_STOP) is None


def test_run_reclaimed_by_dispatcher_is_not_driven_to_close_the_successor(board):
    conn, env = board
    tid = kb.create_task(conn, title="impl", assignee="implementer")
    run_a = kb.claim_task(conn, tid).current_run_id
    _become_worker(env, tid, run_a)
    conn.execute("UPDATE tasks SET claim_expires = 0 WHERE id = ?", (tid,))
    kb.release_stale_claims(conn)
    run_b = kb.claim_task(conn, tid).current_run_id
    assert run_b != run_a

    assert build_kanban_stop_nudge(messages=_NARRATED_STOP) is None


def test_active_run_that_never_transitioned_is_still_nudged(board):
    conn, env = board
    tid = kb.create_task(conn, title="impl", assignee="implementer")
    run_a = kb.claim_task(conn, tid).current_run_id
    _become_worker(env, tid, run_a)

    nudge = build_kanban_stop_nudge(messages=_NARRATED_STOP)
    assert nudge is not None and tid in nudge


def test_active_run_whose_terminal_call_was_refused_is_still_nudged(board):
    conn, env = board
    tid = kb.create_task(conn, title="impl", assignee="implementer")
    run_a = kb.claim_task(conn, tid).current_run_id
    _become_worker(env, tid, run_a)
    refused = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "kanban_complete", "arguments": "{}"}},
        ]},
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1",
         "content": json.dumps({"error": "kanban_complete could not preserve the declared artifacts"})},
    ]

    assert build_kanban_stop_nudge(messages=refused) is not None


def test_unreadable_board_falls_back_to_the_transcript(board, tmp_path):
    conn, env = board
    tid = kb.create_task(conn, title="impl", assignee="implementer")
    run_a = kb.claim_task(conn, tid).current_run_id
    _become_worker(env, tid, run_a)
    env.setenv("HERMES_KANBAN_DB", str(tmp_path / "absent.db"))

    assert build_kanban_stop_nudge(messages=_NARRATED_STOP) is not None
    assert not (tmp_path / "absent.db").exists()


def test_stale_reviewer_lifecycle_calls_cannot_touch_the_successor_run(board):
    from tools import kanban_tools as kt

    conn, env = board
    tid, run_a = _card_in_review(conn)
    assert kb.request_changes(conn, tid, reason="fix it", expected_run_id=run_a)[0]
    run_b = kb.claim_task(conn, tid).current_run_id
    _become_worker(env, tid, run_a)
    before = _snapshot(conn)

    for handler, args in (
        (kt._handle_complete, {"summary": "stale approval"}),
        (kt._handle_block, {"reason": "stale block"}),
        (kt._handle_request_review, {"summary": "stale review"}),
        (kt._handle_heartbeat, {"note": "stale beat"}),
    ):
        out = json.loads(handler(args))
        assert out.get("ok") is not True, (handler.__name__, out)

    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.current_run_id == run_b
    run_b_row = conn.execute("SELECT ended_at, outcome FROM task_runs WHERE id = ?", (run_b,)).fetchone()
    assert run_b_row["ended_at"] is None and run_b_row["outcome"] is None
    lifecycle = lambda dump: [l for l in dump if "task_comments" not in l]
    assert lifecycle(_snapshot(conn)) == lifecycle(before)


def test_stale_worker_exit_books_nothing_against_the_successor(board):
    conn, env = board
    tid, run_a = _card_in_review(conn)
    stale = subprocess.Popen([sys.executable, "-c", "pass"])
    stale.wait()
    conn.execute(
        "UPDATE task_runs SET worker_pid = ?, worker_started_at = 'gone' WHERE id = ?", (stale.pid, run_a),
    )
    assert kb.request_changes(conn, tid, reason="fix it", expected_run_id=run_a)[0]
    successor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        time.sleep(0.2)
        run_b = kb.claim_task(conn, tid, claimer=kb._claimer_id()).current_run_id
        kbd._set_worker_pid(conn, tid, successor.pid)
        conn.execute("UPDATE tasks SET started_at = started_at - 3600 WHERE id = ?", (tid,))
        conn.execute("UPDATE task_runs SET ended_at = ended_at - 3600 WHERE id = ?", (run_a,))

        assert kbd.detect_crashed_workers(conn) == []
        kbd.reap_terminal_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.status == "running" and task.current_run_id == run_b
        assert task.consecutive_failures == 0
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? AND id > (SELECT MAX(id) FROM task_events "
            "WHERE task_id = ? AND kind = 'spawned')", (tid, tid))]
        assert not {"crashed", "protocol_violation", "blocked", "reclaimed"} & set(kinds)
        assert successor.poll() is None
    finally:
        successor.kill()
        successor.wait()
