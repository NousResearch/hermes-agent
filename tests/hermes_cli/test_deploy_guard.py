"""Tests for hermes_cli.deploy_guard — gate that fails loud when the running
tree is not on the canonical deploy branch (t_7161d9a9).

These tests do not depend on the live git tree; they synthesize a tempdir
with a custom ``HERMES_AGENT_HOME`` (and a fake ``.git``) so the assertions
exercise each branch / reason in isolation. The live-tree end-to-end check
is part of the deployed-tree acceptance drive (see kanban-block-loop-quarantine
acceptance in t_7161d9a9).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli.deploy_guard import (
    DEFAULT_DEPLOY_BRANCH,
    DeployBranchDriftError,
    check_deploy_branch,
    check_deploy_state,
)


def _make_git_repo(tmp_path: Path, branch: str) -> Path:
    """Initialize a fresh git repo at tmp_path and leave it on ``branch``."""
    env = {"GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@y", "GIT_COMMITTER_NAME": "x",
           "GIT_COMMITTER_EMAIL": "x@y"}
    subprocess.run(["git", "init", "-q", "-b", branch, str(tmp_path)], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "x@y"], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "x"], check=True, env=env)
    (tmp_path / "README").write_text("hi")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README"], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True, env=env)
    return tmp_path


def test_match_when_on_canonical_branch(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, "canonical-deploy-t_7161d9a9")
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_DEPLOY_BRANCH", "canonical-deploy-t_7161d9a9")
    monkeypatch.setenv("HERMES_DEPLOY_GUARD", "warn")
    r = check_deploy_state()
    assert r.ok is True
    assert r.reason == "match"
    assert r.expected_branch == "canonical-deploy-t_7161d9a9"
    assert r.observed_branch == "canonical-deploy-t_7161d9a9"
    assert r.observed_sha and len(r.observed_sha) >= 7


def test_wrong_branch_returns_reason_wrong_branch(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, "fix/some-feature")
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_DEPLOY_BRANCH", "canonical-deploy-t_7161d9a9")
    monkeypatch.setenv("HERMES_DEPLOY_GUARD", "warn")
    r = check_deploy_state()
    assert r.ok is False
    assert r.reason == "wrong_branch"
    assert r.expected_branch == "canonical-deploy-t_7161d9a9"
    assert r.observed_branch == "fix/some-feature"


def test_detached_head_returns_reason_detached(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, "main")
    # Detach HEAD at the current commit.
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", sha], check=True)
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_DEPLOY_BRANCH", "main")
    r = check_deploy_state()
    assert r.ok is False
    assert r.reason == "detached"
    assert r.observed_branch is None
    assert r.observed_sha == sha


def test_missing_repo_returns_reason_missing_repo(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("HERMES_AGENT_HOME", str(empty))
    r = check_deploy_state()
    assert r.ok is False
    assert r.reason == "missing_repo"


def test_env_override_changes_expected_branch(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, "main")
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_DEPLOY_BRANCH", "main")
    r = check_deploy_state()
    assert r.ok is True
    assert r.expected_branch == "main"


def test_default_branch_is_canonical_deploy_t_7161d9a9(monkeypatch):
    """Without HERMES_DEPLOY_BRANCH the default targets the deploy branch
    built in this card. Future promotions to ``main`` should bump the
    default AND update this test in the same commit.
    """
    monkeypatch.delenv("HERMES_DEPLOY_BRANCH", raising=False)
    assert DEFAULT_DEPLOY_BRANCH == "canonical-deploy-t_7161d9a9"


def test_strict_mode_raises_on_drift(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, "fix/feature")
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_DEPLOY_BRANCH", "canonical-deploy-t_7161d9a9")
    monkeypatch.setenv("HERMES_DEPLOY_GUARD", "warn")
    with pytest.raises(DeployBranchDriftError) as ei:
        check_deploy_branch(strict=True)
    assert "fix/feature" in str(ei.value)
    assert "canonical-deploy-t_7161d9a9" in str(ei.value)


def test_strict_mode_does_not_raise_on_match(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, "canonical-deploy-t_7161d9a9")
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_DEPLOY_BRANCH", "canonical-deploy-t_7161d9a9")
    r = check_deploy_branch(strict=True)
    assert r.ok is True


def test_off_mode_returns_synthetic_match(tmp_path, monkeypatch):
    repo = _make_git_repo(tmp_path, "fix/anything")
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_DEPLOY_BRANCH", "canonical-deploy-t_7161d9a9")
    monkeypatch.setenv("HERMES_DEPLOY_GUARD", "off")
    r = check_deploy_branch()
    assert r.ok is True
    assert r.reason == "match"  # synthetic, not a real inspection


def test_warn_mode_returns_real_result_on_drift(tmp_path, monkeypatch, caplog):
    repo = _make_git_repo(tmp_path, "fix/leak")
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_DEPLOY_BRANCH", "canonical-deploy-t_7161d9a9")
    monkeypatch.setenv("HERMES_DEPLOY_GUARD", "warn")
    with caplog.at_level("WARNING", logger="hermes_cli.deploy_guard"):
        r = check_deploy_branch()
    assert r.ok is False
    assert r.reason == "wrong_branch"
    assert any("DRIFT" in rec.message for rec in caplog.records)


def test_strict_override_wins_over_env(tmp_path, monkeypatch):
    """``strict=False`` passed as kwarg downgrades a strict env var to warn."""
    repo = _make_git_repo(tmp_path, "fix/leak")
    monkeypatch.setenv("HERMES_AGENT_HOME", str(repo))
    monkeypatch.setenv("HERMES_DEPLOY_BRANCH", "canonical-deploy-t_7161d9a9")
    monkeypatch.setenv("HERMES_DEPLOY_GUARD", "strict")
    r = check_deploy_branch(strict=False)
    assert r.ok is False  # still detected
    # No raise — kwarg downgraded the env var.