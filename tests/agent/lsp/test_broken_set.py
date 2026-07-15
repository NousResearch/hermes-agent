"""Tests for transient LSP failure cooldowns.

Request-level and cold-index timeouts must stop repeated edits from paying the
same timeout immediately, without permanently disabling the workspace. A
short monotonic cooldown replaces the old process-lifetime broken mark.
Deterministic configuration failures (for example, no spawn command) still
use ``_broken``.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.workspace import clear_cache


@pytest.fixture(autouse=True)
def _clear_workspace_cache():
    clear_cache()
    yield
    clear_cache()


def _make_git_workspace(tmp_path: Path) -> Path:
    """Build a minimal git repo with a pyproject so pyright's root resolver fires."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='t'\n")
    return repo


def test_mark_failed_file_adds_retryable_cooldown_key(tmp_path, monkeypatch):
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    src = repo / "x.py"
    src.write_text("")
    now = [100.0]

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        lifecycle_enabled=True,
        clock=lambda: now[0],
    )
    try:
        svc._mark_broken_for_file(str(src), RuntimeError("simulated"))
        key = ("pyright", str(repo.resolve()))
        assert key in svc._cooldowns
        assert key not in svc._broken
        assert svc.enabled_for(str(src)) is False
        now[0] = 106.0
        assert svc.enabled_for(str(src)) is True
        assert key not in svc._cooldowns
    finally:
        svc.shutdown()


def test_process_lifetime_mode_preserves_permanent_broken_pair(tmp_path, monkeypatch):
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    src = repo / "x.py"
    src.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    try:
        svc._mark_broken_for_file(str(src), RuntimeError("simulated"))
        key = ("pyright", str(repo.resolve()))
        assert key in svc._broken
        assert key not in svc._cooldowns
        assert svc.enabled_for(str(src)) is False
    finally:
        svc.shutdown()


def test_enabled_for_returns_false_during_cooldown(tmp_path, monkeypatch):
    """Once a (server_id, root) pair is cooling down,
    ``enabled_for`` returns False so the file_operations layer skips
    the LSP path entirely."""
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    src = repo / "x.py"
    src.write_text("")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        lifecycle_enabled=True,
    )
    try:
        # Initially enabled.
        assert svc.enabled_for(str(src)) is True
        # Start the retry cooldown.
        svc._mark_broken_for_file(str(src), RuntimeError("simulated"))
        # Now disabled — the cooldown short-circuits.
        assert svc.enabled_for(str(src)) is False
    finally:
        svc.shutdown()


def test_enabled_for_other_file_in_same_project_also_skipped(tmp_path, monkeypatch):
    """The cooldown key is (server_id, root), so ALL files routed through
    the same server in the same project are skipped — not just the one
    that triggered the failure."""
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    a = repo / "a.py"
    a.write_text("")
    b = repo / "b.py"
    b.write_text("")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        lifecycle_enabled=True,
    )
    try:
        svc._mark_broken_for_file(str(a), RuntimeError("simulated"))
        # Both files in the same project skip pyright now.
        assert svc.enabled_for(str(a)) is False
        assert svc.enabled_for(str(b)) is False
    finally:
        svc.shutdown()


def test_unrelated_project_not_affected_by_cooldown(tmp_path, monkeypatch):
    """Cooling down pyright for project A must NOT affect project B."""
    repo_a = _make_git_workspace(tmp_path)
    repo_b = tmp_path / "repo-b"
    repo_b.mkdir()
    (repo_b / ".git").mkdir()
    (repo_b / "pyproject.toml").write_text("[project]\nname='b'\n")
    a_src = repo_a / "x.py"
    a_src.write_text("")
    b_src = repo_b / "x.py"
    b_src.write_text("")

    monkeypatch.chdir(str(repo_a))
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        lifecycle_enabled=True,
    )
    try:
        svc._mark_broken_for_file(str(a_src), RuntimeError("simulated"))
        # Project A skipped.
        assert svc.enabled_for(str(a_src)) is False
        # Project B still enabled — the cooldown key is per-project.
        monkeypatch.chdir(str(repo_b))
        assert svc.enabled_for(str(b_src)) is True
    finally:
        svc.shutdown()


def test_mark_broken_handles_missing_server_silently(tmp_path):
    """If the file extension doesn't match any registered server,
    ``_mark_broken_for_file`` no-ops — nothing to mark."""
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        lifecycle_enabled=True,
    )
    try:
        # No registered server for .xyz; must not raise.
        svc._mark_broken_for_file(str(tmp_path / "weird.xyz"), RuntimeError("x"))
        assert len(svc._broken) == 0
        assert len(svc._cooldowns) == 0
    finally:
        svc.shutdown()


def test_mark_broken_handles_no_workspace_silently(tmp_path):
    """File outside any git worktree → no workspace → no key to add."""
    src = tmp_path / "orphan.py"
    src.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        lifecycle_enabled=True,
    )
    try:
        svc._mark_broken_for_file(str(src), RuntimeError("x"))
        assert len(svc._broken) == 0
        assert len(svc._cooldowns) == 0
    finally:
        svc.shutdown()


def test_snapshot_failure_enters_retryable_cooldown(tmp_path, monkeypatch):
    """A snapshot failure skips immediate retries without permanent disablement."""
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    src = repo / "x.py"
    src.write_text("")
    now = [100.0]

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        lifecycle_enabled=True,
        clock=lambda: now[0],
    )
    try:
        async def boom(_path):
            raise RuntimeError("outer-timeout simulated")

        with patch.object(svc, "_snapshot_async", boom):
            assert svc.enabled_for(str(src)) is True
            svc.snapshot_baseline(str(src))

        key = ("pyright", str(repo.resolve()))
        assert key in svc._cooldowns
        assert key not in svc._broken
        assert svc.enabled_for(str(src)) is False
        now[0] = 106.0
        assert svc.enabled_for(str(src)) is True
    finally:
        svc.shutdown()
