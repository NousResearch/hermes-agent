"""Tests for Kanban per-task/run budget enforcement."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _usage_report(**overrides) -> dict:
    report = {
        "estimated_cost_usd": 0.25,
        "cost_status": "estimated",
        "cost_source": "official_docs_snapshot",
        "input_tokens": 1000,
        "output_tokens": 250,
        "cache_read_tokens": 50,
        "cache_write_tokens": 5,
        "reasoning_tokens": 10,
        "total_tokens": 1315,
        "api_calls": 2,
        "model": "openai/gpt-5.5",
        "provider": "openrouter",
        "session_id": "sess-budget",
        "completed": True,
        "failed": False,
    }
    report.update(overrides)
    return report


def _event_kinds(conn: sqlite3.Connection, task_id: str) -> list[str]:
    return [event.kind for event in kb.list_events(conn, task_id)]


def _claim_and_release(conn: sqlite3.Connection, task_id: str) -> int:
    claimed = kb.claim_task(conn, task_id, claimer="budget-test")
    assert claimed is not None
    run_id = int(claimed.current_run_id)
    kb.block_task(conn, task_id, reason="retry later", kind="transient")
    kb.unblock_task(conn, task_id)
    return run_id


def _claim_with_usage_path_and_release(
    conn: sqlite3.Connection, task_id: str
) -> tuple[int, Path]:
    claimed = kb.claim_task(conn, task_id, claimer="budget-test")
    assert claimed is not None
    run_id = int(claimed.current_run_id)
    path = Path(kb._set_run_usage_report_path(conn, task_id))
    kb.block_task(conn, task_id, reason="retry later", kind="transient")
    kb.unblock_task(conn, task_id)
    return run_id, path


def _write_usage_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_fresh_schema_has_budget_and_usage_columns(kanban_home):
    with kb.connect() as conn:
        task_cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        run_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_runs)")
        }

    assert {
        "budget_usd",
        "budget_spent_usd",
        "budget_unknown_cost_runs",
    } <= task_cols
    assert {
        "usage_report_path",
        "usage_report_ingested_at",
        "estimated_cost_usd",
        "cost_status",
        "cost_source",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "total_tokens",
        "api_calls",
        "model",
        "provider",
        "usage_session_id",
    } <= run_cols


def test_legacy_db_migrates_budget_and_usage_columns(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path()
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    legacy.execute(
        """
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    legacy.execute(
        """
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            profile TEXT,
            step_key TEXT,
            status TEXT NOT NULL,
            claim_lock TEXT,
            claim_expires INTEGER,
            worker_pid INTEGER,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            outcome TEXT,
            summary TEXT,
            metadata TEXT,
            error TEXT
        )
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, priority, created_at, workspace_kind) "
        "VALUES ('legacy1', 'old', 'ready', 0, 1, 'scratch')"
    )
    legacy.execute(
        "INSERT INTO task_runs (task_id, status, started_at) "
        "VALUES ('legacy1', 'running', 1)"
    )
    legacy.commit()
    legacy.close()

    kb.init_db()
    with kb.connect() as conn:
        task_cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        run_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(task_runs)")
        }
        task = kb.get_task(conn, "legacy1")
        run = kb.list_runs(conn, "legacy1")[0]

    assert {"budget_usd", "budget_spent_usd", "budget_unknown_cost_runs"} <= task_cols
    assert {
        "usage_report_path",
        "usage_report_ingested_at",
        "estimated_cost_usd",
        "cost_status",
        "cost_source",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "total_tokens",
        "api_calls",
        "model",
        "provider",
        "usage_session_id",
    } <= run_cols
    assert task.budget_usd is None
    assert task.budget_spent_usd == 0.0
    assert task.budget_unknown_cost_runs == 0
    assert run.usage_report_path is None
    assert run.estimated_cost_usd is None


def test_create_task_persists_explicit_default_and_no_budget(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(
        kb,
        "_resolve_task_budget_default_usd",
        lambda: 0.75,
    )

    with kb.connect() as conn:
        explicit = kb.create_task(conn, title="explicit", budget_usd=0.5)
        inherited = kb.create_task(conn, title="default")
        opted_out = kb.create_task(conn, title="none", use_default_budget=False)

        assert kb.get_task(conn, explicit).budget_usd == 0.5
        assert kb.get_task(conn, inherited).budget_usd == 0.75
        assert kb.get_task(conn, opted_out).budget_usd is None


def test_invalid_budget_is_rejected(kanban_home):
    with kb.connect() as conn, pytest.raises(ValueError, match="budget_usd"):
        kb.create_task(conn, title="bad budget", budget_usd=0)


def test_ingests_worker_usage_report_once_and_updates_task_spend(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="account me", budget_usd=1.0)
        run_id, path = _claim_with_usage_path_and_release(conn, task_id)
        _write_usage_report(path, _usage_report())

        assert kb.ingest_worker_usage_reports(conn) == 1
        assert kb.ingest_worker_usage_reports(conn) == 0

        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, run_id)
        events = kb.list_events(conn, task_id)

    assert task.budget_spent_usd == pytest.approx(0.25)
    assert task.budget_unknown_cost_runs == 0
    assert run.usage_report_ingested_at is not None
    assert run.estimated_cost_usd == pytest.approx(0.25)
    assert run.cost_status == "estimated"
    assert run.input_tokens == 1000
    assert run.usage_session_id == "sess-budget"
    assert [event.kind for event in events].count("usage_report_ingested") == 1


def test_negative_cost_report_is_clamped_without_reducing_spend(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="bad report", budget_usd=1.0)
        run_id, path = _claim_with_usage_path_and_release(conn, task_id)
        conn.execute(
            "UPDATE tasks SET budget_spent_usd = 0.5 WHERE id = ?",
            (task_id,),
        )
        _write_usage_report(path, _usage_report(estimated_cost_usd=-0.25))

        assert kb.ingest_worker_usage_reports(conn) == 1

        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, run_id)

    assert task.budget_spent_usd == pytest.approx(0.5)
    assert run.estimated_cost_usd == pytest.approx(0.0)
    assert run.cost_status == "estimated"


def test_budget_exhaustion_blocks_next_ready_claim(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="cap", assignee="worker", budget_usd=0.2)
        _claim_and_release(conn, task_id)
        conn.execute(
            "UPDATE tasks SET budget_spent_usd = 0.2 WHERE id = ?",
            (task_id,),
        )

        claimed = kb.claim_task(conn, task_id, claimer="budget-test")
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
        runs = kb.list_runs(conn, task_id)

    assert claimed is None
    assert task.status == "blocked"
    assert task.block_kind == "capability"
    assert "budget_exhausted" in [event.kind for event in events]
    assert runs[-1].outcome == "blocked"


def test_unknown_cost_policy_allow_counts_unknown_but_allows_claim(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(kb, "_resolve_task_budget_unknown_cost_policy", lambda: "allow")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="unknown allowed", budget_usd=1.0)
        run_id, path = _claim_with_usage_path_and_release(conn, task_id)
        _write_usage_report(
            path,
            _usage_report(
                estimated_cost_usd=0.0,
                cost_status="unknown",
                cost_source="none",
            ),
        )

        assert kb.ingest_worker_usage_reports(conn) == 1
        claimed = kb.claim_task(conn, task_id, claimer="budget-test")
        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, run_id)

    assert claimed is not None
    assert task.status == "running"
    assert task.budget_spent_usd == pytest.approx(0.0)
    assert task.budget_unknown_cost_runs == 1
    assert run.cost_status == "unknown"


def test_unknown_cost_policy_block_blocks_budgeted_task_before_claim(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(kb, "_resolve_task_budget_unknown_cost_policy", lambda: "block")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="unknown blocked", budget_usd=1.0)
        _claim_and_release(conn, task_id)
        conn.execute(
            "UPDATE tasks SET budget_unknown_cost_runs = 1 WHERE id = ?",
            (task_id,),
        )

        claimed = kb.claim_task(conn, task_id, claimer="budget-test")
        task = kb.get_task(conn, task_id)
        budget_events = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "budget_exhausted"
        ]

    assert claimed is None
    assert task.status == "blocked"
    assert budget_events[-1].payload["reason"] == "unknown_cost"
    assert budget_events[-1].payload["unknown_cost_policy"] == "block"


def test_unknown_cost_policy_resolves_from_config_module(monkeypatch):
    fake_config = SimpleNamespace(
        load_config=lambda: {
            "kanban": {
                "task_budget": {
                    "unknown_cost_policy": "block",
                }
            }
        }
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.config", fake_config)

    assert kb._resolve_task_budget_unknown_cost_policy() == "block"


def test_dispatch_sets_usage_report_path_and_env_for_current_run(
    kanban_home, all_assignees_spawnable
):
    observed: dict[str, object] = {}

    def _spawn(task, _workspace, *, board=None):
        observed["task_id"] = task.id
        observed["run_id"] = task.current_run_id
        observed["board"] = board
        return 4321

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="spawn", assignee="worker")

        result = kb.dispatch_once(conn, spawn_fn=_spawn, board="default")
        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, int(task.current_run_id))

    assert result.spawned == [(task_id, "worker", task.workspace_path)]
    assert observed == {"task_id": task_id, "run_id": task.current_run_id, "board": "default"}
    assert run.usage_report_path
    assert run.usage_report_path.endswith(f"{task_id}.run-{run.id}.usage.json")


def test_default_spawn_exports_usage_file_for_current_run(
    kanban_home, monkeypatch, tmp_path
):
    captured: dict[str, object] = {}

    class DummyProc:
        pid = 2468

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        return DummyProc()

    monkeypatch.setattr(kb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env",
        lambda _profile: str(tmp_path),
    )

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="env", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="budget-test")
        assert claimed is not None
        kb._set_run_usage_report_path(conn, task_id)
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        pid = kb._default_spawn(claimed, str(workspace), board="default")

    env = captured["env"]
    assert pid == 2468
    assert env["HERMES_KANBAN_RUN_ID"] == str(claimed.current_run_id)
    assert env["HERMES_KANBAN_USAGE_FILE"].endswith(
        f"{task_id}.run-{claimed.current_run_id}.usage.json"
    )
    assert "chat" in captured["cmd"]
    assert "-q" in captured["cmd"]


def test_dispatch_ingests_existing_report_then_blocks_spawn(
    kanban_home, all_assignees_spawnable
):
    spawned: list[str] = []

    def _spawn(task, _workspace):
        spawned.append(task.id)
        return 1234

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="no respawn", assignee="worker", budget_usd=0.1)
        _run_id, path = _claim_with_usage_path_and_release(conn, task_id)
        _write_usage_report(path, _usage_report(estimated_cost_usd=0.25))

        result = kb.dispatch_once(conn, spawn_fn=_spawn)
        task = kb.get_task(conn, task_id)

    assert result.usage_reports_ingested == 1
    assert spawned == []
    assert result.spawned == []
    assert task.status == "blocked"


def test_review_claim_obeys_task_budget_chain(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_resolve_task_budget_unknown_cost_policy", lambda: "allow")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review cap", budget_usd=0.1)
        conn.execute(
            "UPDATE tasks SET status = 'review', budget_spent_usd = 0.1 "
            "WHERE id = ?",
            (task_id,),
        )

        claimed = kb.claim_review_task(conn, task_id, claimer="reviewer")
        task = kb.get_task(conn, task_id)

    assert claimed is None
    assert task.status == "blocked"
    assert task.block_kind == "capability"


def test_kanban_quiet_usage_report_uses_cumulative_agent_totals():
    pytest.importorskip("yaml")

    from cli import _kanban_usage_result_from_agent

    cli = SimpleNamespace(
        session_id="cli-session",
        agent=SimpleNamespace(
            session_estimated_cost_usd=0.42,
            session_cost_status="estimated",
            session_cost_source="official_docs_snapshot",
            session_input_tokens=100,
            session_output_tokens=50,
            session_cache_read_tokens=10,
            session_cache_write_tokens=2,
            session_reasoning_tokens=5,
            session_total_tokens=167,
            session_api_calls=4,
            model="openai/gpt-5.5",
            provider="openrouter",
            session_id="agent-session",
        ),
    )

    result = _kanban_usage_result_from_agent(cli, {"completed": True, "failed": False})

    assert result["estimated_cost_usd"] == 0.42
    assert result["api_calls"] == 4
    assert result["session_id"] == "agent-session"
    assert result["completed"] is True
    assert result["failed"] is False
