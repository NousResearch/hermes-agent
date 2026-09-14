"""Fail-closed attestation for dispatcher-owned implementer workers."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


def _attested_env(monkeypatch, workspace, *, task_id="t_worker", run_id="7", claim="claim-7"):
    monkeypatch.setenv("HERMES_PROFILE", "implementer")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", run_id)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claim)
    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_GRANT", "grant-value")
    monkeypatch.setenv("HERMES_KANBAN_BRANCH", "fix/worker")


def _task(workspace, *, run_id=7, claim="claim-7", branch="fix/worker"):
    return SimpleNamespace(
        id="t_worker",
        workspace_path=str(workspace),
        current_run_id=run_id,
        claim_lock=claim,
        branch_name=branch,
    )


def test_implementer_refuses_missing_dispatch_attestation(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_PROFILE", "implementer")
    from agent.implementer_workspace import ImplementerWorkspaceError, require_attested_workspace

    with pytest.raises(ImplementerWorkspaceError, match="HERMES_KANBAN_TASK"):
        require_attested_workspace(task_loader=lambda _: None)


def test_implementer_accepts_exact_dispatcher_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _attested_env(monkeypatch, workspace)
    from agent.implementer_workspace import require_attested_workspace

    attestation = require_attested_workspace(
        task_loader=lambda task_id: _task(workspace),
        cwd_getter=lambda: str(workspace),
        git_runner=lambda args, cwd: "fix/worker" if args[-1] == "--show-current" else str(workspace),
    )

    assert attestation.workspace == workspace.resolve()
    assert attestation.task_id == "t_worker"


def test_implementer_refuses_git_top_level_mismatch(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _attested_env(monkeypatch, workspace)
    from agent.implementer_workspace import ImplementerWorkspaceError, require_attested_workspace

    with pytest.raises(ImplementerWorkspaceError, match="git top-level"):
        require_attested_workspace(
            task_loader=lambda task_id: _task(workspace),
            cwd_getter=lambda: str(workspace),
            git_runner=lambda args, cwd: str(tmp_path / "other"),
        )


def test_implementer_refuses_task_run_or_claim_mismatch(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _attested_env(monkeypatch, workspace)
    from agent.implementer_workspace import ImplementerWorkspaceError, require_attested_workspace

    with pytest.raises(ImplementerWorkspaceError, match="claim"):
        require_attested_workspace(
            task_loader=lambda task_id: _task(workspace, claim="different"),
            cwd_getter=lambda: str(workspace),
            git_runner=lambda args, cwd: "fix/worker" if args[-1] == "--show-current" else str(workspace),
        )


def test_non_implementer_does_not_require_worker_attestation(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    from agent.implementer_workspace import require_attested_workspace

    assert require_attested_workspace(task_loader=lambda _: None) is None


def test_original_baseline_rejects_changed_original_checkout(monkeypatch, tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    _attested_env(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "HERMES_KANBAN_ORIGINAL_BASELINE",
        '{"branch":"main","head":"expected","path":"%s","status":""}' % original,
    )
    from agent.implementer_workspace import ImplementerWorkspaceError, verify_original_baseline

    def git_runner(args, cwd):
        return {"HEAD": "changed", "--show-current": "main", "--untracked-files=all": ""}[args[-1]]

    with pytest.raises(ImplementerWorkspaceError, match="differs"):
        verify_original_baseline(git_runner=git_runner)
