"""Regression tests for ORBIT worker isolation cron guards (#497)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace


def _git(*args: str, cwd):
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )


def test_orbit_worker_rejects_non_allowlisted_git_remote_before_script(tmp_path):
    import cron.scheduler as sched

    workspace = tmp_path / "t_orbit"
    workspace.mkdir()
    _git("init", cwd=workspace)
    _git("remote", "add", "origin", "https://github.com/adavidson510/glucapet.git", cwd=workspace)

    job = {
        "id": "job-workspace-guard",
        "name": "orbit guarded worker",
        "no_agent": True,
        "script": "would-run-if-guard-failed.sh",
        "workdir": str(workspace),
        "orbit_worker_guard": True,
        "task_id": "t_1ecf7ff4",
        "run_id": "4303",
        # No profile here: this test is about workspace confinement, and should
        # fail before any attribution/GitHub actor lookup is needed.
    }

    success, output, response, error = sched.run_job(job)

    assert success is False
    assert response == ""
    assert sched.WORKSPACE_ALLOWLIST_VIOLATION in (error or "")
    assert sched.WORKSPACE_ALLOWLIST_VIOLATION in output
    assert "glucapet" in output
    assert "t_1ecf7ff4" in output
    # The guard must fire before the script path is resolved/executed.
    assert "script failed" not in output.lower()


def test_orbit_worker_rejects_same_repo_name_wrong_owner_before_script(tmp_path):
    import cron.scheduler as sched

    workspace = tmp_path / "t_wrong_owner"
    workspace.mkdir()
    _git("init", cwd=workspace)
    _git(
        "remote",
        "add",
        "origin",
        "https://github.com/not-orbit/orbit-governance.git",
        cwd=workspace,
    )

    job = {
        "id": "job-owner-guard",
        "name": "orbit guarded worker",
        "no_agent": True,
        "script": "would-run-if-owner-guard-failed.sh",
        "workdir": str(workspace),
        "orbit_worker_guard": True,
        "task_id": "t_owner_guard",
        "run_id": "4304",
    }

    success, output, response, error = sched.run_job(job)

    assert success is False
    assert response == ""
    assert sched.WORKSPACE_ALLOWLIST_VIOLATION in (error or "")
    assert sched.WORKSPACE_ALLOWLIST_VIOLATION in output
    assert "not-orbit/orbit-governance" in output
    assert "script failed" not in output.lower()


def test_observed_github_actor_ignores_spoofable_env(monkeypatch):
    import cron.scheduler as sched

    monkeypatch.setenv("HERMES_GITHUB_ACTOR", "expected-spoof")
    monkeypatch.setenv("GITHUB_ACTOR", "expected-spoof")
    monkeypatch.setenv("GH_ACTOR", "expected-spoof")
    monkeypatch.setattr(sched.shutil, "which", lambda name: "/usr/bin/gh")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="real-token-user\n", stderr="")

    monkeypatch.setattr(sched.subprocess, "run", fake_run)

    assert sched._observed_github_actor() == "real-token-user"


def test_orbit_worker_rejects_profile_actor_mismatch_before_mutation(tmp_path, monkeypatch):
    import cron.scheduler as sched

    workspace = tmp_path / "t_attr"
    workspace.mkdir()
    monkeypatch.setattr(sched, "_observed_github_actor", lambda: "orbitrivet")

    job = {
        "id": "job-attr-guard",
        "name": "orbit attribution worker",
        "no_agent": True,
        "script": "would-post-if-guard-failed.sh",
        "workdir": str(workspace),
        "orbit_worker_guard": True,
        "task_id": "t_1ecf7ff4",
        "run_id": "4303",
        "profile": "orbitcommand",
    }

    success, output, response, error = sched.run_job(job)

    assert success is False
    assert response == ""
    assert sched.WORKER_ATTRIBUTION_MISMATCH in (error or "")
    assert sched.WORKER_ATTRIBUTION_MISMATCH in output
    assert "orbitcommand" in output
    assert "orbitrivet" in output
    assert "4303" in output
    # The guard must fire before the script/comment/receipt mutation path.
    assert "script failed" not in output.lower()
