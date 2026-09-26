"""P0-C research failure classification and synthesis-only recovery contracts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent.tool_guardrails import (
    RESEARCH_SYNTHESIS_ONLY,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_failure import (
    FAILURE_CLASS_RESEARCH_BUDGET_EXHAUSTED,
    FAILURE_CLASS_TERMINAL_PROTOCOL_VIOLATION,
    FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS,
    FINALIZATION_PROTOCOL_FAILURE,
    checkpoint_for_task,
    classify_failure,
    read_checkpoint,
    write_research_checkpoint,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in (
        "HERMES_KANBAN_DB", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    kb.init_db()
    return home


_POLICY = {
    "web_search_max": 2,
    "browser_extract_max": 1,
    "collection_tools": ["web_search", "web_extract"],
}


def test_failure_classification_prefers_authoritative_terminal_and_budget_states():
    assert classify_failure(
        outcome="timed_out", research_budget={"enabled": True}, timed_out=True,
    ) == FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS
    assert classify_failure(
        outcome="timed_out", research_budget={"enabled": True, "exhausted": True}, timed_out=True,
    ) == FAILURE_CLASS_RESEARCH_BUDGET_EXHAUSTED
    assert classify_failure(protocol_violation=True, research_budget={"exhausted": True}) == (
        FAILURE_CLASS_TERMINAL_PROTOCOL_VIOLATION
    )
    assert classify_failure(outcome="crashed") is None


def test_synthesis_only_blocks_collection_but_keeps_finalize_tools_available():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig.from_mapping({"research_mode": "synthesis_only"}, platform="cron")
    )

    collection = controller.before_call("web_search", {"query": "new sources"})
    assert collection.action == "block"
    assert collection.code == RESEARCH_SYNTHESIS_ONLY
    assert not collection.allows_execution

    for tool_name in ("terminal", "read_file", "write_file", "kanban_complete"):
        assert controller.before_call(tool_name, {}).allows_execution


def test_timeout_without_preserved_evidence_fails_closed(kanban_home, monkeypatch):
    now = int(time.time())
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="bounded timeout", assignee="worker", research_budget=_POLICY,
        )
        task = kb.claim_task(conn, task_id)
        assert task is not None
        pid = 71234
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = NULL WHERE id = ?",
            (pid, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = ?",
            (now - 100, task.current_run_id),
        )
        conn.execute(
            "UPDATE tasks SET max_runtime_seconds = ? WHERE id = ?",
            (10, task_id),
        )
        conn.commit()

        timed_out = kbd.enforce_max_runtime(conn, signal_fn=lambda *_args: None)
        assert timed_out == [task_id]
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert (row["status"], row["block_kind"]) == ("blocked", "needs_input")
        runs = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        metadata = json.loads(runs["metadata"])
        assert metadata["failure_class"] == FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS
        assert metadata["evidence_present"] is False
        assert metadata["research_recovery"] is True


def test_preserved_checkpoint_routes_next_attempt_to_synthesis_only(kanban_home, monkeypatch):
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    captured = {}

    def spawn(task, workspace, *, board=None, recovery_mode=None):
        captured["task_id"] = task.id
        captured["recovery_mode"] = recovery_mode
        return None

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="recover bounded research", assignee="worker", research_budget=_POLICY,
        )
        task = kb.claim_task(conn, task_id)
        assert task is not None
        checkpoint = checkpoint_for_task(task_id)
        write_research_checkpoint(
            task_id=task_id,
            research_budget={"evidence_count": 1},
            failure_class=FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS,
            path=checkpoint,
        )
        assert read_checkpoint(checkpoint)["evidence_present"] is True
        kbd._record_task_failure(
            conn,
            task_id,
            "collection timed out",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            event_payload_extra={
                "failure_class": FAILURE_CLASS_TIMEOUT_BEFORE_SYNTHESIS,
                "failure_code": "TIMEOUT_BEFORE_SYNTHESIS",
                "evidence_present": True,
                "checkpoint_path": str(checkpoint),
                "research_recovery": True,
            },
        )

        result = kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
        assert result.spawned

    assert captured == {"task_id": task_id, "recovery_mode": "synthesis_only"}


def test_protocol_failure_preserves_evidence_and_never_auto_completes(kanban_home, monkeypatch):
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    captured = {}

    def spawn(task, workspace, *, board=None, recovery_mode=None):
        captured["recovery_mode"] = recovery_mode
        return None

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="protocol failure", assignee="worker")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        checkpoint = checkpoint_for_task(task_id)
        write_research_checkpoint(
            task_id=task_id,
            research_budget={"evidence_count": 1},
            path=checkpoint,
        )
        pid = 71235
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = NULL WHERE id = ?",
            (pid, task_id),
        )
        conn.commit()
        kbd._record_worker_exit(pid, 0)

        assert kbd.detect_crashed_workers(conn) == [task_id]
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert row["status"] == "ready"
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        metadata = json.loads(run["metadata"])
        assert metadata["failure_class"] == FAILURE_CLASS_TERMINAL_PROTOCOL_VIOLATION
        assert metadata["failure_code"] == FINALIZATION_PROTOCOL_FAILURE
        assert metadata["evidence_present"] is True

        result = kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
        assert result.spawned
        recovered_task = kb.get_task(conn, task_id)
        assert recovered_task is not None
        assert recovered_task.status == "running"

    assert captured["recovery_mode"] == "synthesis_only"


def test_protocol_failure_without_evidence_stays_fail_closed_after_unblock(kanban_home, monkeypatch):
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    spawned = []

    def spawn(*_args, **_kwargs):
        spawned.append(True)
        return None

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="protocol failure without evidence", assignee="worker",
            research_budget=_POLICY,
        )
        task = kb.claim_task(conn, task_id)
        assert task is not None
        pid = 71236
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = NULL WHERE id = ?",
            (pid, task_id),
        )
        conn.commit()
        kbd._record_worker_exit(pid, 0)

        assert kbd.detect_crashed_workers(conn) == [task_id]
        row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert row["status"] == "blocked"

        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        metadata = json.loads(run["metadata"])
        assert metadata["failure_class"] == FAILURE_CLASS_TERMINAL_PROTOCOL_VIOLATION
        assert metadata["research_recovery"] is True
        assert metadata["evidence_present"] is False

        assert kb.unblock_task(conn, task_id) is True
        result = kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
        assert not result.spawned
        assert result.auto_blocked == [task_id]
        assert spawned == []
