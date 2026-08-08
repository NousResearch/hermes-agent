"""Machine-enforced not-before safety invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


FUTURE = "2030-01-01T00:00:00Z"
PAST = "2020-01-01T00:00:00Z"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn, task_id: str, kind: str):
    return [e for e in kb.list_events(conn, task_id) if e.kind == kind]


def test_future_task_is_scheduled_and_dependency_resolver_releases_only_when_due(
    kanban_home, monkeypatch
):
    clock = [1_700_000_000.0]
    monkeypatch.setattr(kb.time, "time", lambda: clock[0])

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="wait for release",
            assignee="default",
            not_before=FUTURE,
        )
        assert kb.get_task(conn, task_id).status == "scheduled"

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, task_id).status == "scheduled"
        blocked = _events(conn, task_id, "not_before_blocked")
        assert len(blocked) == 1
        assert blocked[0].payload["operation"] == "dependency_resolver"

        # Repeated dispatcher ticks do not flood the audit stream.
        assert kb.recompute_ready(conn) == 0
        assert len(_events(conn, task_id, "not_before_blocked")) == 1

        clock[0] = 1_900_000_000.0
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, task_id).status == "ready"


def test_direct_claim_and_completion_bypasses_are_blocked_without_state_mutation(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(kb.time, "time", lambda: 1_700_000_000.0)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="protected", assignee="default")
        conn.execute(
            "UPDATE tasks SET not_before = ?, status = 'ready' WHERE id = ?",
            (FUTURE, task_id),
        )

        assert kb.claim_task(conn, task_id, claimer="bypass") is None
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.claim_lock is None
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0] == 0

        # Simulate an already-running drain/root-drain worker that gets gated
        # after claim. Completion must leave task and run evidence untouched.
        conn.execute(
            "UPDATE tasks SET not_before = NULL WHERE id = ?", (task_id,)
        )
        claimed = kb.claim_task(conn, task_id, claimer="worker")
        assert claimed is not None
        run_id = kb.get_task(conn, task_id).current_run_id
        conn.execute(
            "UPDATE tasks SET not_before = ? WHERE id = ?", (FUTURE, task_id)
        )

        assert kb.complete_task(
            conn,
            task_id,
            summary="unsafe early completion",
            expected_run_id=run_id,
        ) is False
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.current_run_id == run_id
        run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert tuple(run) == ("running", None, None)
        complete_block = _events(conn, task_id, "not_before_blocked")[-1]
        assert complete_block.payload["operation"] == "complete"


def test_dispatcher_and_default_assignment_make_no_mutation_before_release(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(kb.time, "time", lambda: 1_700_000_000.0)
    spawned = []

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="future unassigned", not_before=FUTURE)
        # Simulate a stale/external writer bypassing the scheduled status. The
        # dispatcher backstop must run before default-assignee mutation.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args: spawned.append(args),
            default_assignee="default",
            reconcile_orphans=False,
        )
        task = kb.get_task(conn, task_id)
        assert result.spawned == []
        assert result.skipped_not_before == [task_id]
        assert spawned == []
        assert task.status == "ready"
        assert task.assignee is None
        assert _events(conn, task_id, "not_before_blocked")[-1].payload[
            "operation"
        ] == "dispatcher"


def test_manual_promotion_requires_sealed_human_override_and_audits_it(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(kb.time, "time", lambda: 1_700_000_000.0)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="operator release",
            assignee="default",
            not_before=FUTURE,
        )

        ok, reason = kb.promote_task(
            conn,
            task_id,
            actor="worker",
            force=True,
        )
        assert ok is False
        assert "not-before" in reason
        assert kb.get_task(conn, task_id).status == "scheduled"

        override = kb.authenticated_human_not_before_override(
            actor="alex",
            reason="approved emergency release",
            authenticated_by="local_tty_os_user",
        )
        ok, reason = kb.promote_task(
            conn,
            task_id,
            actor="alex",
            reason="approved emergency release",
            force=True,
            not_before_override=override,
        )
        assert (ok, reason) == (True, None)
        promoted = kb.get_task(conn, task_id)
        assert promoted.status == "ready"
        assert promoted.not_before is None
        event = _events(conn, task_id, "not_before_overridden")[-1]
        assert event.payload == {
            "operation": "manual_promotion",
            "not_before": FUTURE,
            "actor": "alex",
            "reason": "approved emergency release",
            "authenticated_by": "local_tty_os_user",
        }


def test_synthetic_completed_evidence_is_blocked_before_release(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(kb.time, "time", lambda: 1_700_000_000.0)

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="no synthetic proof",
            assignee="default",
            not_before=FUTURE,
        )
        with kb.write_txn(conn):
            synthetic_id = kb._synthesize_ended_run(
                conn,
                task_id,
                outcome="completed",
                summary="fabricated early proof",
            )
        assert synthetic_id == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0] == 0
        event = _events(conn, task_id, "not_before_blocked")[-1]
        assert event.payload["operation"] == "synthetic_evidence_creation"


def test_tool_execution_backstop_blocks_before_relay_or_provider_mutation(
    kanban_home, monkeypatch
):
    from agent import relay_tools, tool_executor

    monkeypatch.setattr(kb.time, "time", lambda: 1_700_000_000.0)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="no provider mutation",
            assignee="default",
            not_before=FUTURE,
        )
        db_path = kb.kanban_db_path()

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    relay_called = []
    execute_called = []
    monkeypatch.setattr(
        relay_tools,
        "execute",
        lambda *args, **kwargs: relay_called.append((args, kwargs)),
    )

    outcome = tool_executor._run_agent_tool_execution_middleware(
        object(),
        function_name="terminal",
        function_args={"command": "vercel remove production"},
        effective_task_id="session-task",
        tool_call_id="call-1",
        execute=lambda args: execute_called.append(args),
    )

    assert outcome.blocked is True
    assert outcome.dispatched is False
    assert relay_called == []
    assert execute_called == []
    assert "not-before deadline" in json.loads(outcome.result)["error"]

    with kb.connect(db_path=db_path) as conn:
        event = _events(conn, task_id, "not_before_blocked")[-1]
        assert event.payload["operation"] == "side_effect_execution:terminal"
