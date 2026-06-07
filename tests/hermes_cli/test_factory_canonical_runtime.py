"""Canonical Factory runtime contracts.

These tests protect Jean's Factory architecture: Agent Core Postgres is the only
runtime source of truth, project docs are namespaced per project, and dashboard
control actions are exposed as deterministic Factory operations rather than chat
handoffs or SQLite fallbacks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import factory_backend


def test_factory_worker_semantic_done_overrides_process_abort_exit_code():
    from hermes_cli import factory_pg

    assert factory_pg._effective_exit_code(-6, "RESULT ok\nSTATE: DONE\n") == 0
    assert factory_pg._effective_exit_code(1, "STATE:DONE") == 0
    assert factory_pg._effective_exit_code(1, "STATE: BLOCKED") == 1
    # Reviewers can exit 0 after writing a final blocked/rework report. The
    # final semantic marker must win over process status, but historical markers
    # embedded in prompt/result_summary must not override a later DONE.
    assert factory_pg._effective_exit_code(0, "STATE: BLOCKED") == 1
    assert factory_pg._effective_exit_code(0, "STATE:BLOCKED") == 1
    assert factory_pg._effective_exit_code(0, "Prompt history STATE: BLOCKED\nFinal answer\nSTATE: DONE\n") == 0
    assert factory_pg._effective_exit_code(0, "Earlier STATE: DONE\nReview found a blocker\nSTATE: BLOCKED\n") == 1


def test_factory_monitor_summary_preserves_semantic_state_outside_tail(tmp_path: Path):
    """A long final report must not hide STATE: DONE from monitor_runs()."""

    from hermes_cli import factory_pg

    log = tmp_path / "worker.log"
    log.write_text("STATE: DONE\n" + ("x" * 6000), encoding="utf-8")

    summary = factory_pg._read_worker_output_summary(log)

    assert "STATE: DONE" in summary
    assert summary.endswith("x" * 4000)

def test_factory_rework_is_active_but_runnable_not_in_flight():
    from hermes_cli import factory_pg

    assert factory_pg._has_active_increment([{"status": "rework"}]) is True
    assert factory_pg._has_in_flight_increment([{"status": "rework"}]) is False
    assert factory_pg._has_in_flight_increment([{"status": "running"}]) is True


def test_factory_monitor_repairs_orphan_review_running_task(monkeypatch):
    from hermes_cli import factory_pg

    queries: list[str] = []
    writes: list[str] = []
    monkeypatch.setattr(factory_pg, "ensure_runtime_schema", lambda: None)
    monkeypatch.setattr(factory_pg, "_user", lambda: "factory_runtime")

    def fake_json_query(query, *, user):
        queries.append(query)
        assert user == "factory_runtime"
        if "t.status='review_running'" in query:
            return [{"project_id": "demo", "lane_id": "demo-hybrid", "task_id": "demo-f8", "status": "review_ready"}]
        return []

    monkeypatch.setattr(factory_pg.sql, "json_query", fake_json_query)
    monkeypatch.setattr(factory_pg.sql, "psql", lambda query, *, user: writes.append(query))

    repaired = factory_pg._repair_orphan_in_flight_tasks("demo")

    assert repaired[0]["project_id"] == "demo"
    assert repaired[0]["lane_id"] == "demo-hybrid"
    assert repaired[0]["task_id"] == "demo-f8"
    assert repaired[0]["status"] == "review_ready"
    combined = "\n".join(queries)
    assert "NOT EXISTS" in combined
    assert "factory.task_runs" in combined
    assert "review_running_without_active_run" in combined
    assert "orphan_inflight_repaired" in writes[0]


def test_factory_reconciliation_detects_incomplete_methodology_contract(tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    project_dir = repo / "factory" / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "INDEX.md").write_text("# partial legacy index\n", encoding="utf-8")
    project = {
        "project_id": "demo",
        "name": "Demo",
        "repo_path": str(repo),
        "status": "active",
        "risk_level": "high",
        "metadata": {"artifact_dir": "factory/projects/demo"},
    }

    findings = factory_pg.reconciliation_findings(project, [], [])
    codes = {finding["code"] for finding in findings}

    assert "missing_notion_project" in codes
    assert "missing_required_docs" in codes
    assert "missing_task_graph" in codes
    assert "deliverable_unverified" in codes


def test_factory_reconciliation_honors_required_docs_waiver_only_when_authorized(tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    project_dir = repo / "factory" / "projects" / "demo"
    project_dir.mkdir(parents=True)
    project = {
        "project_id": "demo",
        "name": "Demo",
        "repo_path": str(repo),
        "status": "active",
        "risk_level": "high",
        "metadata": {
            "artifact_dir": "factory/projects/demo",
            "notion_waived": True,
            "notion_waived_authorized_by": "jean-garcia",
            "notion_waived_reason": "Jean explicitly accepted no Notion for this project.",
            "required_docs_waived": True,
            "required_docs_waived_authorized_by": "jean-garcia",
            "required_docs_waived_reason": "Jean explicitly accepted reduced docs for this project.",
        },
    }
    tasks = [{"task_id": "demo-f0", "status": "todo", "title": "F0", "phase": "planning"}]

    codes = {finding["code"] for finding in factory_pg.reconciliation_findings(project, tasks, [])}

    assert "missing_notion_project" not in codes
    assert "missing_required_docs" not in codes


def _write_complete_factory_doc_pack(project_dir: Path, *, omit_from_index: set[str] | None = None) -> None:
    from hermes_cli import factory_pg

    omit_from_index = omit_from_index or set()
    project_dir.mkdir(parents=True, exist_ok=True)
    for name in factory_pg.FACTORY_REQUIRED_DOCS:
        (project_dir / name).write_text(f"# {name}\n\nRequired Factory artifact.\n", encoding="utf-8")
    indexed = [name for name in factory_pg.FACTORY_REQUIRED_DOCS if name not in omit_from_index]
    (project_dir / "DOCUMENTATION_INDEX.md").write_text(
        "# Documentation Index\n\n" + "\n".join(f"- `{name}`" for name in indexed) + "\n",
        encoding="utf-8",
    )


def _repo_first_ready_project(repo: Path) -> dict:
    return {
        "project_id": "demo",
        "name": "Demo",
        "repo_path": str(repo),
        "status": "active",
        "risk_level": "high",
        "metadata": {
            "artifact_dir": "factory/projects/demo",
            "notion_tracker_url": "https://notion.example/demo",
        },
    }


def test_factory_reconciliation_detects_docs_missing_from_documentation_index(tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    project_dir = repo / "factory" / "projects" / "demo"
    _write_complete_factory_doc_pack(project_dir, omit_from_index={"SECURITY_REVIEW.md"})
    project = _repo_first_ready_project(repo)
    tasks = [{"task_id": "demo-f0", "status": "todo", "title": "F0", "phase": "planning"}]
    gates = [{"gate_type": "delivery", "status": "passed"}]

    findings = factory_pg.reconciliation_findings(project, tasks, [], gates)
    by_code = {finding["code"]: finding for finding in findings}

    assert "missing_required_docs" not in by_code
    assert by_code["docs_not_indexed"]["metadata"]["missing_from_index"] == ["SECURITY_REVIEW.md"]


def test_factory_reconciliation_detects_uncommitted_project_artifacts(monkeypatch, tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    project_dir = repo / "factory" / "projects" / "demo"
    _write_complete_factory_doc_pack(project_dir)
    project = _repo_first_ready_project(repo)
    tasks = [{"task_id": "demo-f0", "status": "todo", "title": "F0", "phase": "planning"}]
    gates = [{"gate_type": "delivery", "status": "passed"}]
    monkeypatch.setattr(
        factory_pg,
        "_uncommitted_project_artifacts",
        lambda repo_path, artifact_dir: [" M factory/projects/demo/PRD.md"],
    )

    findings = factory_pg.reconciliation_findings(project, tasks, [], gates)
    by_code = {finding["code"]: finding for finding in findings}

    assert by_code["uncommitted_project_artifacts"]["metadata"]["uncommitted_paths"] == [" M factory/projects/demo/PRD.md"]


def test_factory_critical_readiness_requires_index_and_commit_checkpoint(monkeypatch, tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    project_dir = repo / "factory" / "projects" / "demo"
    _write_complete_factory_doc_pack(project_dir, omit_from_index={"TASK_GRAPH.md"})
    project = _repo_first_ready_project(repo)
    monkeypatch.setattr(factory_pg, "_project", lambda project_id: project)
    monkeypatch.setattr(factory_pg, "_user", lambda: "factory_runtime")
    monkeypatch.setattr(factory_pg.sql, "rows", lambda query, *, user: [])
    monkeypatch.setattr(
        factory_pg,
        "_uncommitted_project_artifacts",
        lambda repo_path, artifact_dir: ["?? factory/projects/demo/TASK_GRAPH.md"],
    )

    findings = factory_pg.critical_readiness_findings("demo")

    assert "documentation index missing required docs: TASK_GRAPH.md" in findings
    assert "uncommitted project-local factory artifacts: ?? factory/projects/demo/TASK_GRAPH.md" in findings


def test_factory_reconciliation_ignores_unauthorized_completion_waivers(tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    (repo / "factory" / "projects" / "demo").mkdir(parents=True)
    project = {
        "project_id": "demo",
        "name": "Demo",
        "repo_path": str(repo),
        "status": "completed",
        "risk_level": "high",
        "metadata": {
            "artifact_dir": "factory/projects/demo",
            "notion_waived": True,
            "required_docs_waived": True,
        },
    }
    tasks = [{"task_id": "demo-f0", "status": "done", "title": "F0", "phase": "planning"}]

    codes = {finding["code"] for finding in factory_pg.reconciliation_findings(project, tasks, [])}

    assert "missing_notion_project" in codes
    assert "missing_required_docs" in codes


def test_factory_critical_readiness_honors_tracker_and_docs_waivers_when_authorized(monkeypatch, tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    (repo / "factory" / "projects" / "demo").mkdir(parents=True)
    project = {
        "project_id": "demo",
        "repo_path": str(repo),
        "risk_level": "high",
        "metadata": {
            "artifact_dir": "factory/projects/demo",
            "notion_waived": True,
            "notion_waived_authorized_by": "jean-garcia",
            "notion_waived_reason": "Jean explicitly accepted no Notion for this project.",
            "required_docs_waived": True,
            "required_docs_waived_authorized_by": "jean-garcia",
            "required_docs_waived_reason": "Jean explicitly accepted reduced docs for this project.",
        },
    }
    monkeypatch.setattr(factory_pg, "_project", lambda project_id: project)
    monkeypatch.setattr(factory_pg, "_user", lambda: "factory_runtime")
    monkeypatch.setattr(factory_pg.sql, "rows", lambda query, *, user: [])

    assert factory_pg.critical_readiness_findings("demo") == []


def test_factory_reconcile_creates_recovery_tasks_and_moves_empty_project_to_planned(monkeypatch, tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    (repo / "factory" / "projects" / "demo").mkdir(parents=True)
    project = {
        "project_id": "demo",
        "name": "Demo",
        "repo_path": str(repo),
        "status": "active",
        "risk_level": "high",
        "metadata": {"artifact_dir": "factory/projects/demo", "autonomous_enabled": True},
    }
    writes: list[str] = []
    monkeypatch.setattr(factory_pg, "ensure_runtime_schema", lambda: None)
    monkeypatch.setattr(factory_pg, "_project", lambda project_id: project)
    monkeypatch.setattr(factory_pg, "_tasks", lambda project_id: [])
    monkeypatch.setattr(factory_pg, "_active_pending_gates", lambda project_id: [])
    monkeypatch.setattr(factory_pg, "_user", lambda: "factory_runtime")
    monkeypatch.setattr(factory_pg.sql, "rows", lambda query, *, user: [])
    monkeypatch.setattr(factory_pg.sql, "psql", lambda query, *, user: writes.append(query))

    result = factory_pg.reconcile_project("demo")

    assert result["status"] == "planned"
    assert result["anomalies"]
    assert result["reconciliation_tasks_created"] >= 1
    combined = "\n".join(writes)
    assert "reconciliation_anomaly" in combined
    assert "factory_reconciliation_task" in combined
    assert "UPDATE factory.projects" in combined


def test_factory_reconcile_holds_documentation_only_work_open(monkeypatch, tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    project_dir = repo / "factory" / "projects" / "demo"
    project_dir.mkdir(parents=True)
    for name in factory_pg.FACTORY_REQUIRED_DOCS:
        (project_dir / name).write_text(f"# {name}\n", encoding="utf-8")
    (project_dir / "DOCUMENTATION_INDEX.md").write_text(
        "# Documentation Index\n\n" + "\n".join(f"- `{name}`" for name in factory_pg.FACTORY_REQUIRED_DOCS) + "\n",
        encoding="utf-8",
    )
    project = {
        "project_id": "demo",
        "name": "Demo",
        "repo_path": str(repo),
        "status": "completed",
        "risk_level": "high",
        "metadata": {
            "artifact_dir": "factory/projects/demo",
            "notion_waived": True,
        },
    }
    before_tasks = [
        {"task_id": "demo-f0", "status": "done", "title": "F0", "phase": "implementation"},
        {
            "task_id": "demo-reconcile-missing-notion-project",
            "status": "cancelled",
            "title": "R0 — Reconciliation: create Notion",
            "phase": "documentation",
            "metadata": {"factory_reconciliation_task": True, "reconciliation_anomaly": "missing_notion_project"},
        },
    ]
    after_tasks = [
        before_tasks[0],
        {**before_tasks[1], "status": "todo"},
    ]
    task_snapshots = iter([before_tasks, after_tasks])
    writes: list[str] = []

    monkeypatch.setattr(factory_pg, "ensure_runtime_schema", lambda: None)
    monkeypatch.setattr(factory_pg, "_project", lambda project_id: project)
    monkeypatch.setattr(factory_pg, "_tasks", lambda project_id: next(task_snapshots))
    monkeypatch.setattr(factory_pg, "_active_pending_gates", lambda project_id: [])
    monkeypatch.setattr(factory_pg, "_latest_gate_rows", lambda project_id: [{"gate_type": "delivery", "status": "passed"}])
    monkeypatch.setattr(factory_pg, "_user", lambda: "factory_runtime")
    monkeypatch.setattr(factory_pg.sql, "rows", lambda query, *, user: [])
    monkeypatch.setattr(factory_pg.sql, "psql", lambda query, *, user: writes.append(query))

    result = factory_pg.reconcile_project("demo")

    assert result["status"] == "delivery_hold"
    assert result["anomalies"] == ["missing_notion_project"]
    combined = "\n".join(writes)
    assert "status=CASE WHEN factory.tasks.status IN" in combined
    assert "Project reconciled as delivery_hold" in combined


def test_factory_reconcile_treats_documentation_review_as_delivery_hold(monkeypatch, tmp_path: Path):
    from hermes_cli import factory_pg

    repo = tmp_path / "repo"
    project = {
        "project_id": "demo",
        "name": "Demo",
        "repo_path": str(repo),
        "status": "blocked",
        "risk_level": "high",
        "metadata": {"artifact_dir": "factory/projects/demo"},
    }
    tasks = [
        {"task_id": "demo-f0", "status": "done", "title": "F0", "phase": "implementation"},
        {
            "task_id": "demo-docs-review",
            "status": "review_pending_human",
            "title": "PM/Docs tracker ownership",
            "phase": "documentation",
        },
    ]
    writes: list[str] = []

    monkeypatch.setattr(factory_pg, "ensure_runtime_schema", lambda: None)
    monkeypatch.setattr(factory_pg, "_project", lambda project_id: project)
    monkeypatch.setattr(factory_pg, "_tasks", lambda project_id: tasks)
    monkeypatch.setattr(factory_pg, "_active_pending_gates", lambda project_id: [])
    monkeypatch.setattr(factory_pg, "_latest_gate_rows", lambda project_id: [])
    monkeypatch.setattr(factory_pg, "_user", lambda: "factory_runtime")
    monkeypatch.setattr(factory_pg.sql, "rows", lambda query, *, user: [])
    monkeypatch.setattr(factory_pg.sql, "psql", lambda query, *, user: writes.append(query))

    result = factory_pg.reconcile_project("demo")

    assert result["status"] == "delivery_hold"
    assert result["task_counts"]["review_pending_human"] == 1


def test_factory_force_tick_claims_rework_before_new_task(monkeypatch):
    from hermes_cli import factory_pg

    calls: list[str] = []
    monkeypatch.setattr(factory_pg, "ensure_runtime_schema", lambda: None)
    monkeypatch.setattr(factory_pg, "monitor_runs", lambda: {"checked": 0, "finished": 0})
    monkeypatch.setattr(factory_pg, "clear_resolved_blockers", lambda project_id: {"project_id": project_id, "reopened": []})
    monkeypatch.setattr(factory_pg, "reconcile_project", lambda project_id: {"project_id": project_id, "status": "active"})
    monkeypatch.setattr(factory_pg, "claim_next_review", lambda project_id=None, *, worker: None)

    def fake_rework(project_id=None, *, worker):
        calls.append(f"rework:{project_id}:{worker}")
        return {"run_id": "run-rework", "run_type": "rework"}

    def fake_task(project_id=None, *, worker):  # pragma: no cover - must not be reached
        raise AssertionError("todo/ready task claimed before rework was resubmitted")

    monkeypatch.setattr(factory_pg, "claim_next_rework", fake_rework)
    monkeypatch.setattr(factory_pg, "claim_next_task", fake_task)

    tick = factory_pg.force_tick("demo-project")

    assert tick["claimed"] == {"run_id": "run-rework", "run_type": "rework"}
    assert calls == ["rework:demo-project:factory-force-tick"]


def test_factory_claim_next_rework_creates_owner_run(monkeypatch):
    from hermes_cli import factory_pg

    queries: list[str] = []
    writes: list[str] = []
    monkeypatch.setattr(factory_pg, "ensure_runtime_schema", lambda: None)
    monkeypatch.setattr(factory_pg, "_user", lambda: "factory_runtime")
    monkeypatch.setattr(factory_pg.time, "time", lambda: 1234567890)

    class FakeUUID:
        hex = "abcdef1234567890"

    monkeypatch.setattr(factory_pg.uuid, "uuid4", lambda: FakeUUID())

    def fake_statement_one(query, *, user):
        queries.append(query)
        assert user == "factory_runtime"
        return {
            "task_id": "demo-f3",
            "project_id": "demo",
            "lane_id": "demo-hybrid",
            "title": "F3",
            "owner_profile": "implementation-planner",
            "reviewer_profile": "factory-orchestrator",
            "engine": "zeus",
        }

    def fake_psql(query, *, user):
        writes.append(query)
        assert user == "factory_runtime"

    monkeypatch.setattr(factory_pg.sql, "statement_one", fake_statement_one)
    monkeypatch.setattr(factory_pg.sql, "psql", fake_psql)

    claim = factory_pg.claim_next_rework("demo", worker="factory-force-tick")

    assert claim is not None
    assert claim["run_id"] == "run-1234567890-abcdef12"
    assert claim["worker_profile"] == "implementation-planner"
    assert claim["run_type"] == "rework"
    assert "t.status='rework'" in queries[0]
    assert "t_busy.status IN" in queries[0]
    assert "run_type" in writes[0] and "rework" in writes[0]
    assert "rework_claimed" in writes[0]
    assert "p.status IN ('active','planned','intake','blocked')" in queries[0]


def test_factory_blocker_classifier_action_categories():
    from hermes_cli import factory_pg

    payload = {
        "gates": [{"gate_type": "test", "status": "passed"}],
        "task_runs": [],
        "tasks": [
            {
                "project_id": "demo",
                "task_id": "demo-auto",
                "status": "blocked",
                "title": "F1",
                "result_summary": "Necesita aprobación para correr pytest; test gate already passed.",
            },
            {
                "project_id": "demo",
                "task_id": "demo-human",
                "status": "blocked",
                "title": "F2",
                "result_summary": "Requires API key credential / 2FA owner decision.",
            },
            {
                "project_id": "demo",
                "task_id": "demo-tech",
                "status": "blocked",
                "title": "F3",
                "result_summary": "pytest failed with TypeError in activity_tool.",
            },
            {
                "project_id": "demo",
                "task_id": "demo-orphan",
                "status": "review_running",
                "title": "F4",
                "result_summary": "Reviewer session vanished.",
            },
        ],
    }

    by_task = {item["task_id"]: item for item in factory_pg.classify_factory_blockers(payload)}

    assert by_task["demo-auto"]["action_category"] == "auto_resolvable"
    assert by_task["demo-human"]["action_category"] == "human_question_required"
    assert by_task["demo-human"]["requires_human"] is True
    assert by_task["demo-tech"]["action_category"] == "technical_rework"
    assert by_task["demo-orphan"]["action_category"] == "stale_orphan_state"


def test_factory_watchdog_alerts_detects_blocked_without_question_and_orphan():
    from hermes_cli import factory_pg

    payload = {
        "projects": [{"project_id": "demo", "status": "blocked", "autonomous_enabled": True}],
        "tasks": [
            {"project_id": "demo", "task_id": "demo-blocked", "status": "blocked", "updated_at": "2026-01-01T00:00:00Z"},
            {"project_id": "demo", "task_id": "demo-orphan", "status": "running", "updated_at": "2026-01-01T00:00:00Z"},
        ],
        "task_runs": [],
        "human_questions": [],
        "gates": [],
    }

    alert_types = {alert["alert_type"] for alert in factory_pg.factory_watchdog_alerts(payload, blocked_minutes=0, claimed_null_rounds=3)}

    assert "autonomous_project_blocked_too_long" in alert_types
    assert "blocked_without_human_question" in alert_types
    assert "orphan_inflight_without_active_run" in alert_types
    assert "cron_claimed_null_repeated" in alert_types


def test_factory_claim_next_task_considers_blocked_projects(monkeypatch):
    from hermes_cli import factory_pg

    queries: list[str] = []
    writes: list[str] = []
    monkeypatch.setattr(factory_pg, "ensure_runtime_schema", lambda: None)
    monkeypatch.setattr(factory_pg, "_user", lambda: "factory_runtime")
    monkeypatch.setattr(factory_pg, "_tasks", lambda project_id: [])
    monkeypatch.setattr(factory_pg, "_next_runnable_task", lambda project_id: {
        "task_id": "demo-f2",
        "project_id": "demo",
        "lane_id": "demo-hybrid",
        "owner_profile": "claude-builder",
        "reviewer_profile": "quality-reviewer",
        "engine": "claude-code",
    })
    monkeypatch.setattr(factory_pg.time, "time", lambda: 1234567890)

    class FakeUUID:
        hex = "abcdef1234567890"

    monkeypatch.setattr(factory_pg.uuid, "uuid4", lambda: FakeUUID())

    def fake_rows(query, *, user):
        queries.append(query)
        assert user == "factory_runtime"
        assert "p.status IN ('active','planned','intake','blocked')" in query
        return [{"project_id": "demo"}]

    def fake_statement_one(query, *, user):
        queries.append(query)
        assert user == "factory_runtime"
        return {
            "task_id": "demo-f2",
            "project_id": "demo",
            "lane_id": "demo-hybrid",
            "owner_profile": "claude-builder",
            "reviewer_profile": "quality-reviewer",
            "engine": "claude-code",
        }

    monkeypatch.setattr(factory_pg.sql, "rows", fake_rows)
    monkeypatch.setattr(factory_pg.sql, "statement_one", fake_statement_one)
    monkeypatch.setattr(factory_pg.sql, "psql", lambda query, *, user: writes.append(query))

    claim = factory_pg.claim_next_task("demo", worker="factory-force-tick")

    assert claim is not None
    assert claim["run_id"] == "run-1234567890-abcdef12"
    assert claim["worker_profile"] == "claude-builder"
    assert "task_claimed" in writes[0]


def test_factory_backend_refuses_sqlite_even_when_requested(monkeypatch):
    """Factory runtime must not silently route to legacy SQLite anymore."""

    monkeypatch.setenv("HERMES_FACTORY_BACKEND", "sqlite")
    monkeypatch.setattr("hermes_cli.factory_pg.available", lambda: False)

    with pytest.raises(factory_backend.FactoryBackendUnavailable) as excinfo:
        factory_backend.get_backend(explicit_sqlite_path="/tmp/legacy-factory.db")

    message = str(excinfo.value).lower()
    assert "postgres" in message
    assert "sqlite" in message
    assert "disabled" in message


def test_factory_cli_parser_has_no_sqlite_db_option():
    from hermes_cli import factory

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    factory_parser = factory.add_parser(subparsers)

    option_dests = {action.dest for action in factory_parser._actions}
    assert "db" not in option_dests


def test_factory_doc_inventory_is_project_namespaced(tmp_path):
    from hermes_cli import web_server

    repo = tmp_path / "repo"
    # This global legacy doc should not satisfy the project-specific contract.
    legacy_dir = repo / "factory"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "PRD.md").write_text("# Wrong global PRD\n", encoding="utf-8")

    project_dir = repo / "factory" / "projects" / "demo-project"
    project_dir.mkdir(parents=True)
    (project_dir / "PRD.md").write_text("# Demo project PRD\n", encoding="utf-8")

    docs = web_server._factory_doc_inventory(
        {"project_id": "demo-project", "repo_path": str(repo)}
    )

    prd = next(doc for doc in docs if doc["name"] == "PRD.md")
    assert prd["exists"] is True
    assert prd["path"].endswith("factory/projects/demo-project/PRD.md")

    missing = web_server._factory_doc_inventory(
        {"project_id": "another-project", "repo_path": str(repo)}
    )
    missing_prd = next(doc for doc in missing if doc["name"] == "PRD.md")
    assert missing_prd["exists"] is False
    assert missing_prd["path"].endswith("factory/projects/another-project/PRD.md")


def test_factory_project_dashboard_reports_active_run_and_workflow():
    from hermes_cli import web_server

    project = {"project_id": "demo", "repo_path": ""}
    tasks = [
        {
            "project_id": "demo",
            "task_id": "demo-inc-01",
            "title": "Increment 01",
            "status": "running",
            "phase": "implementation",
            "updated_at": "2026-06-05T00:00:00Z",
            "created_at": "2026-06-05T00:00:00Z",
            "evidence_status": "missing",
            "owner_agent_id": "claude-builder",
        }
    ]
    events = []
    gates = []
    runs = [
        {
            "run_id": "run-1",
            "project_id": "demo",
            "task_id": "demo-inc-01",
            "worker_profile": "claude-builder",
            "status": "running",
            "heartbeat_at": "2026-06-05T00:01:00Z",
        }
    ]

    dashboard = web_server._factory_project_dashboard(project, tasks, gates, events, runs)

    assert dashboard["active_run"] == runs[0]
    assert dashboard["workflow"]["operative"] is True
    assert dashboard["workflow"]["stage"] == "running"
    assert dashboard["workflow"]["worker"] == "claude-builder"
