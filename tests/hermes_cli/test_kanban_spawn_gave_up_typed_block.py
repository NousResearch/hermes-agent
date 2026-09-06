"""Circuit-breaker spawn give-up is a typed capability block.

A workspace/project/capability spawn that burns ``effective_limit`` retries
used to land in ``blocked`` with a ``gave_up`` event and empty comments â€”
indistinguishable from a ``needs_input`` open-questions block. The trip now
writes ``block_kind=capability`` and a ``blocked`` event whose reason is the
error. Timeout/crash trips and worker ``needs_input`` blocks are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn, task_id: str, kind: str):
    return [e for e in kb.list_events(conn, task_id) if e.kind == kind]


def test_spawn_failed_trip_writes_typed_capability_block(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no repo to check out", assignee="worker")
        kb.claim_task(conn, tid)
        error = (
            "workspace: task %s has workspace_kind=worktree but no workspace_path, "
            "and board 'vx' has no default_workdir set." % tid
        )
        tripped = kbd._record_task_failure(
            conn, tid, error,
            outcome="spawn_failed", failure_limit=1,
            release_claim=True, end_run=True,
        )
        assert tripped is True
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "capability"
        assert task.last_failure_error == error[:500]
        assert _events(conn, tid, "gave_up") == []
        blocked = _events(conn, tid, "blocked")
        assert len(blocked) == 1
        payload = blocked[0].payload or {}
        assert payload["kind"] == "capability"
        assert payload["reason"] == error[:500]
        assert payload["trigger_outcome"] == "spawn_failed"
        assert payload["error"] == error[:500]
        comments = conn.execute(
            "SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (tid,),
        ).fetchone()[0]
        assert comments == 0


def test_spawn_failed_below_limit_does_not_type_a_block(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="first spawn miss", assignee="worker")
        kb.claim_task(conn, tid)
        tripped = kbd._record_task_failure(
            conn, tid, "workspace: missing workdir",
            outcome="spawn_failed", failure_limit=3,
            release_claim=True, end_run=True,
        )
        assert tripped is False
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.block_kind is None
        assert _events(conn, tid, "blocked") == []
        assert _events(conn, tid, "gave_up") == []
        failed = _events(conn, tid, "spawn_failed")
        assert len(failed) == 1
        assert (failed[0].payload or {}).get("error") == "workspace: missing workdir"


def test_timeout_trip_still_emits_gave_up(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ran out of time", assignee="worker")
        kb.claim_task(conn, tid)
        conn.execute(
            "UPDATE tasks SET status = 'ready', consecutive_failures = 0 "
            "WHERE id = ?", (tid,),
        )
        conn.commit()
        tripped = kbd._record_task_failure(
            conn, tid, "worker exceeded max_runtime_seconds",
            outcome="timed_out", failure_limit=1,
            release_claim=False, end_run=False,
        )
        assert tripped is True
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind is None
        assert _events(conn, tid, "blocked") == []
        gave_up = _events(conn, tid, "gave_up")
        assert len(gave_up) == 1
        assert (gave_up[0].payload or {}).get("trigger_outcome") == "timed_out"


def test_needs_input_block_is_unchanged(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="open questions", assignee="worker")
        kb.claim_task(conn, tid)
        ok = kb.block_task(
            conn, tid,
            reason="open-questions: awaiting Lucas",
            kind="needs_input",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert ok is True
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        blocked = _events(conn, tid, "blocked")
        assert len(blocked) == 1
        payload = blocked[0].payload or {}
        assert payload["kind"] == "needs_input"
        assert payload["reason"] == "open-questions: awaiting Lucas"
        assert "trigger_outcome" not in payload
        assert _events(conn, tid, "gave_up") == []
