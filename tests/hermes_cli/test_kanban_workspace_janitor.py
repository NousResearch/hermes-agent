from __future__ import annotations

import sqlite3

from hermes_cli.kanban_workspace_janitor import (
    CleanupAction,
    WorkspaceCleanupAction,
    WorkspaceReport,
    apply_artifact_cleanup_actions,
    apply_workspace_cleanup_actions,
    classify_workspace,
    classify_workspaces,
    collect_disk_pressure_report,
    discover_artifacts,
    format_disk_pressure_report,
    path_size,
    plan_artifact_cleanup_actions,
    plan_workspace_cleanup_actions,
)


def test_discover_artifacts_counts_allowlisted_without_descending(tmp_path):
    workspace = tmp_path / "t_done"
    nested = workspace / "app" / "node_modules" / "pkg"
    nested.mkdir(parents=True)
    (nested / "index.js").write_text("x" * 10)
    ignored = workspace / "app" / "important"
    ignored.mkdir()
    (ignored / "note.txt").write_text("keep")

    artifacts = discover_artifacts(workspace)

    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "node_modules"
    assert artifacts[0]["path"].endswith("app/node_modules")
    assert artifacts[0]["size_bytes"] >= 10


def test_classify_non_terminal_workspace_as_blocked_active(tmp_path):
    workspace = tmp_path / "t_running"
    (workspace / "node_modules").mkdir(parents=True)

    report = classify_workspace(
        workspace,
        {"id": "t_running", "status": "running", "completed_at": None},
        now=10_000,
        proc_cwds=[],
        pane_cwds=[],
    )

    assert report.state == "blocked-active"
    assert report.gates["terminal_status"] is False


def test_classify_active_cwd_under_workspace_as_blocked_active(tmp_path):
    workspace = tmp_path / "t_done"
    active_dir = workspace / "subdir"
    active_dir.mkdir(parents=True)

    report = classify_workspace(
        workspace,
        {
            "id": "t_done",
            "status": "done",
            "completed_at": 1,
            "result": "preserved",
        },
        now=10_000,
        proc_cwds=[str(active_dir)],
        pane_cwds=[],
    )

    assert report.state == "blocked-active"
    assert report.gates["active_refs"] == [str(active_dir)]


def test_classify_terminal_old_artifact_as_safe_artifact_candidate(tmp_path):
    workspace = tmp_path / "t_done"
    artifact = workspace / "node_modules"
    artifact.mkdir(parents=True)
    (artifact / "package.js").write_text("x" * 20)

    report = classify_workspace(
        workspace,
        {
            "id": "t_done",
            "status": "done",
            "completed_at": 1,
            "result": "summary preserved",
        },
        now=200_000,
        proc_cwds=[],
        pane_cwds=[],
    )

    assert report.state == "safe-artifact-candidate"
    assert report.gates["artifact_ttl_met"] is True
    assert report.artifacts[0]["kind"] == "node_modules"


def test_classify_terminal_without_evidence_requires_approval(tmp_path):
    workspace = tmp_path / "t_done"
    (workspace / "node_modules").mkdir(parents=True)

    report = classify_workspace(
        workspace,
        {"id": "t_done", "status": "done", "completed_at": 1},
        now=200_000,
        proc_cwds=[],
        pane_cwds=[],
    )

    assert report.state == "approval-required"
    assert "lacks preserved" in report.reason


def test_classify_workspaces_loads_task_metadata_from_db(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    workspaces = tmp_path / "workspaces"
    workspace = workspaces / "t_db"
    artifact = workspace / "node_modules"
    artifact.mkdir(parents=True)
    (artifact / "package.js").write_text("x" * 12)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            public_id TEXT,
            title TEXT,
            status TEXT,
            review_phase TEXT,
            completed_at INTEGER,
            result TEXT,
            closeout_evidence TEXT,
            current_run_id INTEGER,
            worker_pid INTEGER,
            assignee TEXT,
            workspace_kind TEXT,
            workspace_path TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "t_db",
            "BO-999",
            "Fixture task",
            "done",
            "worker_done",
            1,
            "summary preserved",
            None,
            None,
            None,
            None,
            "scratch",
            None,
        ),
    )
    conn.commit()

    monkeypatch.setattr("hermes_cli.kanban_workspace_janitor.process_cwds", lambda: [])
    monkeypatch.setattr("hermes_cli.kanban_workspace_janitor.tmux_cwds", lambda: [])

    reports = classify_workspaces(db_path, workspaces, now=200_000)

    assert len(reports) == 1
    assert reports[0].task["public_id"] == "BO-999"
    assert reports[0].state == "safe-artifact-candidate"
    assert reports[0].size_bytes == path_size(workspace)


def test_plan_artifact_cleanup_actions_only_uses_safe_candidates(tmp_path):
    safe_workspace = tmp_path / "safe"
    safe_artifact = safe_workspace / "node_modules"
    safe_artifact.mkdir(parents=True)
    (safe_artifact / "pkg.js").write_text("x")
    blocked_workspace = tmp_path / "blocked"
    blocked_artifact = blocked_workspace / "node_modules"
    blocked_artifact.mkdir(parents=True)

    safe_report = classify_workspace(
        safe_workspace,
        {"id": "safe", "status": "done", "completed_at": 1, "result": "kept"},
        now=200_000,
        proc_cwds=[],
        pane_cwds=[],
    )
    blocked_report = classify_workspace(
        blocked_workspace,
        {"id": "blocked", "status": "blocked", "completed_at": None},
        now=200_000,
        proc_cwds=[],
        pane_cwds=[],
    )

    actions = plan_artifact_cleanup_actions([blocked_report, safe_report])

    assert len(actions) == 1
    assert actions[0].task_id == "safe"
    assert actions[0].artifact_path == str(safe_artifact)


def test_apply_artifact_cleanup_actions_dry_run_never_deletes(tmp_path):
    workspace = tmp_path / "safe"
    artifact = workspace / "node_modules"
    artifact.mkdir(parents=True)
    (artifact / "pkg.js").write_text("x")
    action = CleanupAction(
        task_id="safe",
        workspace_path=str(workspace),
        artifact_path=str(artifact),
        kind="node_modules",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )

    result = apply_artifact_cleanup_actions([action])

    assert result == [
        {
            "task_id": "safe",
            "workspace_path": str(workspace),
            "artifact_path": str(artifact),
            "kind": "node_modules",
            "size_bytes": 1,
            "candidate_state": "safe-artifact-candidate",
            "reason": "fixture",
            "dry_run": True,
            "deleted": False,
            "guard_errors": [],
        }
    ]
    assert artifact.exists()


def test_apply_artifact_cleanup_actions_apply_requires_exact_guards(tmp_path):
    workspace = tmp_path / "safe"
    artifact = workspace / "node_modules"
    artifact.mkdir(parents=True)
    (artifact / "pkg.js").write_text("x")
    bad = CleanupAction(
        task_id="safe",
        workspace_path=str(workspace),
        artifact_path=str(tmp_path / "outside" / "node_modules"),
        kind="node_modules",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )
    good = CleanupAction(
        task_id="safe",
        workspace_path=str(workspace),
        artifact_path=str(artifact),
        kind="node_modules",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )

    results = apply_artifact_cleanup_actions([bad, good], dry_run=False)

    assert results[0]["deleted"] is False
    assert "artifact_not_under_workspace" in results[0]["guard_errors"]
    assert results[1]["deleted"] is True
    assert not artifact.exists()


def test_plan_workspace_cleanup_actions_requires_clean_git_and_evidence(tmp_path):
    clean_report = WorkspaceReport(
        task_id="t_clean",
        workspace_path=str(tmp_path / "t_clean"),
        state="future-workspace-cleanup-candidate",
        reason="clean terminal workspace",
        size_bytes=10,
        task={"id": "t_clean"},
        artifacts=[],
        gates={
            "git": {"is_git_worktree": True, "dirty": False},
            "active_refs": [],
            "active_worker": False,
            "has_evidence": True,
        },
    )
    dirty_report = WorkspaceReport(
        task_id="t_dirty",
        workspace_path=str(tmp_path / "t_dirty"),
        state="future-workspace-cleanup-candidate",
        reason="dirty terminal workspace",
        size_bytes=20,
        task={"id": "t_dirty"},
        artifacts=[],
        gates={
            "git": {"is_git_worktree": True, "dirty": True},
            "active_refs": [],
            "active_worker": False,
            "has_evidence": True,
        },
    )

    actions = plan_workspace_cleanup_actions([dirty_report, clean_report])

    assert len(actions) == 1
    assert actions[0].task_id == "t_clean"


def test_apply_workspace_cleanup_actions_dry_run_and_guards(tmp_path, monkeypatch):
    workspace = tmp_path / "t_clean"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / "file.txt").write_text("kept during dry run")
    action = WorkspaceCleanupAction(
        task_id="t_clean",
        workspace_path=str(workspace),
        size_bytes=1,
        candidate_state="future-workspace-cleanup-candidate",
        reason="fixture",
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_workspace_janitor.git_state",
        lambda path: {"is_git_worktree": True, "dirty": False, "status_short": ""},
    )

    result = apply_workspace_cleanup_actions([action])

    assert result[0]["dry_run"] is True
    assert result[0]["deleted"] is False
    assert result[0]["guard_errors"] == []
    assert workspace.exists()


def test_apply_workspace_cleanup_actions_apply_rejects_dirty_and_deletes_clean(tmp_path, monkeypatch):
    dirty = tmp_path / "t_dirty"
    clean = tmp_path / "t_clean"
    dirty.mkdir()
    clean.mkdir()
    (dirty / ".git").mkdir()
    (clean / ".git").mkdir()
    (clean / "file.txt").write_text("remove me")

    def fake_git_state(path):
        return {"is_git_worktree": True, "dirty": path.name == "t_dirty", "status_short": " M x" if path.name == "t_dirty" else ""}

    monkeypatch.setattr("hermes_cli.kanban_workspace_janitor.git_state", fake_git_state)
    results = apply_workspace_cleanup_actions(
        [
            WorkspaceCleanupAction("t_dirty", str(dirty), 1, "future-workspace-cleanup-candidate", "fixture"),
            WorkspaceCleanupAction("t_clean", str(clean), 1, "future-workspace-cleanup-candidate", "fixture"),
        ],
        dry_run=False,
    )

    assert results[0]["deleted"] is False
    assert "workspace_git_not_clean" in results[0]["guard_errors"]
    assert dirty.exists()
    assert results[1]["deleted"] is True
    assert not clean.exists()


def test_collect_disk_pressure_report_is_read_only_and_counts_candidates(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    workspaces = tmp_path / "workspaces"
    workspace = workspaces / "t_done"
    artifact = workspace / "node_modules"
    artifact.mkdir(parents=True)
    (artifact / "pkg.js").write_text("x" * 32)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            public_id TEXT,
            title TEXT,
            status TEXT,
            review_phase TEXT,
            completed_at INTEGER,
            result TEXT,
            closeout_evidence TEXT,
            current_run_id INTEGER,
            worker_pid INTEGER,
            assignee TEXT,
            workspace_kind TEXT,
            workspace_path TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("t_done", "BO-999", "Fixture", "done", "worker_done", 1, "kept", None, None, None, None, "scratch", None),
    )
    conn.commit()
    monkeypatch.setattr("hermes_cli.kanban_workspace_janitor.process_cwds", lambda: [])
    monkeypatch.setattr("hermes_cli.kanban_workspace_janitor.tmux_cwds", lambda: [])

    report = collect_disk_pressure_report(
        root_path=tmp_path,
        data_path=tmp_path,
        db_path=db_path,
        workspaces_root=workspaces,
        top_paths=[workspace],
        min_workspace_bytes=1,
        min_artifact_bytes=1,
        now=200_000,
    )

    assert artifact.exists()
    assert report["safe_to_apply_without_approval"] is False
    assert report["workspace_state_counts"] == {"safe-artifact-candidate": 1}
    assert len(report["artifact_cleanup_candidates"]) == 1
    assert report["workspace_cleanup_candidates"] == []


def test_format_disk_pressure_report_includes_boundary_and_candidates():
    rendered = format_disk_pressure_report(
        {
            "root": {"used_percent": 96.0, "free_bytes": 1024},
            "data": {"used_percent": 10.0, "free_bytes": 2048},
            "pressure_level": "critical",
            "artifact_cleanup_candidates": [
                {"kind": "node_modules", "artifact_path": "/tmp/t/node_modules", "size_bytes": 1024}
            ],
            "workspace_cleanup_candidates": [],
            "top_paths": [{"path": "/tmp/t", "exists": True, "size_bytes": 2048}],
            "workspace_state_counts": {"safe-artifact-candidate": 1},
        }
    )

    assert "Daily disk pressure report" in rendered
    assert "root: 96.0% used" in rendered
    assert "cleanup candidates: artifacts=1, full_workspaces=0" in rendered
    assert "boundary: read-only" in rendered


def test_format_disk_pressure_report_shows_resolved_symlink_target():
    rendered = format_disk_pressure_report(
        {
            "root": {"used_percent": 91.0, "free_bytes": 1024},
            "data": None,
            "pressure_level": "warning",
            "artifact_cleanup_candidates": [],
            "workspace_cleanup_candidates": [],
            "top_paths": [
                {
                    "path": "/home/ubuntu/.hermes/kanban/workspaces",
                    "resolved_path": "/mnt/hermes-data/hermes/kanban-workspaces",
                    "exists": True,
                    "size_bytes": 2048,
                }
            ],
            "workspace_state_counts": {},
        }
    )

    assert "/home/ubuntu/.hermes/kanban/workspaces -> /mnt/hermes-data/hermes/kanban-workspaces" in rendered


# --- Slice 3: artifact cleanup executor safety gate tests ---

from hermes_cli.kanban_workspace_janitor import ARTIFACT_NAMES


def test_artifact_names_includes_coverage():
    """RALPLAN Slice 3: 'coverage' must be in the allowlisted artifact names."""
    assert "coverage" in ARTIFACT_NAMES


def test_apply_artifact_cleanup_rejects_active_process_cwd(tmp_path):
    """Active process CWD under target blocks artifact deletion."""
    workspace = tmp_path / "t_active"
    artifact = workspace / "node_modules"
    artifact.mkdir(parents=True)
    (artifact / "pkg.js").write_text("x")

    action = CleanupAction(
        task_id="t_active",
        workspace_path=str(workspace),
        artifact_path=str(artifact),
        kind="node_modules",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )

    results = apply_artifact_cleanup_actions(
        [action], dry_run=False, proc_cwds=[str(artifact)]
    )

    assert results[0]["deleted"] is False
    assert "active_process_cwd" in results[0]["guard_errors"]
    assert artifact.exists()


def test_apply_artifact_cleanup_rejects_active_tmux_cwd(tmp_path):
    """Active tmux pane CWD under target blocks artifact deletion."""
    workspace = tmp_path / "t_tmux"
    artifact = workspace / "node_modules"
    artifact.mkdir(parents=True)
    (artifact / "pkg.js").write_text("x")

    action = CleanupAction(
        task_id="t_tmux",
        workspace_path=str(workspace),
        artifact_path=str(artifact),
        kind="node_modules",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )

    results = apply_artifact_cleanup_actions(
        [action], dry_run=False, pane_cwds=[str(artifact)]
    )

    assert results[0]["deleted"] is False
    assert "active_tmux_cwd" in results[0]["guard_errors"]
    assert artifact.exists()


def test_apply_artifact_cleanup_rejects_active_kanban_worker(
    tmp_path, monkeypatch
):
    """Active Kanban worker/run blocks artifact deletion."""
    workspace = tmp_path / "t_worker"
    artifact = workspace / ".pytest_cache"
    artifact.mkdir(parents=True)
    (artifact / "v").mkdir()

    action = CleanupAction(
        task_id="t_worker",
        workspace_path=str(workspace),
        artifact_path=str(artifact),
        kind=".pytest_cache",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )

    # Simulate an active kanban DB row for this task.
    db_path = tmp_path / "kanban.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            public_id TEXT,
            title TEXT,
            status TEXT,
            review_phase TEXT,
            completed_at INTEGER,
            result TEXT,
            closeout_evidence TEXT,
            current_run_id INTEGER,
            worker_pid INTEGER,
            assignee TEXT,
            workspace_kind TEXT,
            workspace_path TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "t_worker", "BO-999", "Active", "running", None,
            None, None, None, 42, 99999, "arisu", "scratch", None,
        ),
    )
    conn.commit()

    results = apply_artifact_cleanup_actions(
        [action], dry_run=False, kanban_db_path=str(db_path)
    )

    assert results[0]["deleted"] is False
    assert "active_kanban_worker" in results[0]["guard_errors"]
    assert artifact.exists()


def test_apply_artifact_cleanup_allows_worker_task_without_active_run(
    tmp_path
):
    """Task with no current_run_id and no worker_pid is safe to clean."""
    workspace = tmp_path / "t_idle"
    artifact = workspace / ".ruff_cache"
    artifact.mkdir(parents=True)
    (artifact / "0.1").mkdir()

    action = CleanupAction(
        task_id="t_idle",
        workspace_path=str(workspace),
        artifact_path=str(artifact),
        kind=".ruff_cache",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )

    db_path = tmp_path / "kanban.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            public_id TEXT,
            title TEXT,
            status TEXT,
            review_phase TEXT,
            completed_at INTEGER,
            result TEXT,
            closeout_evidence TEXT,
            current_run_id INTEGER,
            worker_pid INTEGER,
            assignee TEXT,
            workspace_kind TEXT,
            workspace_path TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "t_idle", "BO-888", "Done task", "done", "worker_done",
            1, "kept", None, None, None, "arisu", "scratch", None,
        ),
    )
    conn.commit()

    results = apply_artifact_cleanup_actions(
        [action], dry_run=False, kanban_db_path=str(db_path)
    )

    assert results[0]["deleted"] is True
    assert not artifact.exists()


def test_discover_artifacts_finds_coverage_dir(tmp_path):
    """coverage directory is discovered as a cleanable artifact."""
    workspace = tmp_path / "t_cov"
    cov = workspace / "coverage"
    cov.mkdir(parents=True)
    (cov / "lcov.info").write_text("SF:src/app.py\nDA:1,1\n")

    artifacts = discover_artifacts(workspace)

    assert any(a["kind"] == "coverage" for a in artifacts)


def test_apply_artifact_cleanup_dry_run_reports_without_deleting(tmp_path):
    """Dry-run mode reports candidates but never mutates the workspace."""
    workspace = tmp_path / "t_dry"
    artifact = workspace / ".mypy_cache"
    artifact.mkdir(parents=True)
    (artifact / "3.8").mkdir()

    action = CleanupAction(
        task_id="t_dry",
        workspace_path=str(workspace),
        artifact_path=str(artifact),
        kind=".mypy_cache",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )

    results = apply_artifact_cleanup_actions([action], dry_run=True)

    assert results[0]["dry_run"] is True
    assert results[0]["deleted"] is False
    assert artifact.exists()  # never mutated


def test_apply_artifact_cleanup_rejects_symlink_artifact(tmp_path):
    """Symlink artifact path is rejected even if basename is allowlisted."""
    workspace = tmp_path / "t_sym"
    real_artifact = tmp_path / "real_node_modules"
    real_artifact.mkdir(parents=True)
    (real_artifact / "pkg.js").write_text("x")
    workspace.mkdir(parents=True)
    (workspace / "node_modules").symlink_to(real_artifact)

    action = CleanupAction(
        task_id="t_sym",
        workspace_path=str(workspace),
        artifact_path=str(workspace / "node_modules"),
        kind="node_modules",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )

    results = apply_artifact_cleanup_actions([action], dry_run=False)

    assert results[0]["deleted"] is False
    assert "artifact_is_symlink" in results[0]["guard_errors"]
    assert real_artifact.exists()


def test_apply_artifact_cleanup_rejects_outside_workspace(tmp_path):
    """Path outside the declared workspace is rejected."""
    workspace = tmp_path / "t_out"
    workspace.mkdir()
    outside = tmp_path / "outside" / "dist"
    outside.mkdir(parents=True)

    action = CleanupAction(
        task_id="t_out",
        workspace_path=str(workspace),
        artifact_path=str(outside),
        kind="dist",
        size_bytes=1,
        candidate_state="safe-artifact-candidate",
        reason="fixture",
    )

    results = apply_artifact_cleanup_actions([action], dry_run=False)

    assert results[0]["deleted"] is False
    assert "artifact_not_under_workspace" in results[0]["guard_errors"]
    assert outside.exists()


def test_validate_artifact_safety_exported():
    """validate_artifact_safety function is importable for closeout verifier reuse."""
    from hermes_cli.kanban_workspace_janitor import validate_artifact_safety

    result = validate_artifact_safety(
        artifact_path="/tmp/nonexistent_node_modules",
        workspace_path="/tmp",
        kind="node_modules",
    )
    assert isinstance(result, dict)
    assert "safe" in result
    assert "guard_errors" in result
