"""Tests for the broken-set short-circuit added to handle outer-timeout failures.

When ``snapshot_baseline`` or ``get_diagnostics_sync`` time out from the
service layer (because a language server hangs during initialize, or
the binary is wedged), the inner spawn task is cancelled — but the
inner exception handler that adds to ``_broken`` never runs.  Without
the service-layer fallback added in this module, every subsequent
edit re-pays the full timeout cost until the process exits.

This module verifies:
- ``_mark_broken_for_file`` adds the right key
- ``enabled_for`` short-circuits on broken keys
- a missing binary is broken-set'd after one snapshot attempt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec
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


def test_mark_broken_for_file_adds_correct_key(tmp_path, monkeypatch):
    """``_mark_broken_for_file`` keys the broken-set on
    (server_id, per_server_root) so subsequent ``enabled_for`` calls
    for files in the same project skip immediately."""
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
        # The pyright server resolves to the repo root via pyproject.toml.
        assert ("pyright", str(repo)) in svc._broken
    finally:
        svc.shutdown()


def test_enabled_for_returns_false_after_broken(tmp_path, monkeypatch):
    """Once a (server_id, root) pair is in the broken-set,
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
    )
    try:
        # Initially enabled.
        assert svc.enabled_for(str(src)) is True
        # Mark broken.
        svc._mark_broken_for_file(str(src), RuntimeError("simulated"))
        # Now disabled — the broken-set short-circuits.
        assert svc.enabled_for(str(src)) is False
    finally:
        svc.shutdown()


def test_enabled_for_other_file_in_same_project_also_skipped(tmp_path, monkeypatch):
    """The broken key is (server_id, root), so ALL files routed through
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
    )
    try:
        svc._mark_broken_for_file(str(a), RuntimeError("simulated"))
        # Both files in the same project skip pyright now.
        assert svc.enabled_for(str(a)) is False
        assert svc.enabled_for(str(b)) is False
    finally:
        svc.shutdown()


def test_unrelated_project_not_affected_by_broken(tmp_path, monkeypatch):
    """Marking pyright broken for project A must NOT affect project B."""
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
    )
    try:
        svc._mark_broken_for_file(str(a_src), RuntimeError("simulated"))
        # Project A skipped.
        assert svc.enabled_for(str(a_src)) is False
        # Project B still enabled — the broken key is per-project.
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
    )
    try:
        # No registered server for .xyz; must not raise.
        svc._mark_broken_for_file(str(tmp_path / "weird.xyz"), RuntimeError("x"))
        assert len(svc._broken) == 0
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
    )
    try:
        svc._mark_broken_for_file(str(src), RuntimeError("x"))
        assert len(svc._broken) == 0
    finally:
        svc.shutdown()


def test_snapshot_failure_marks_broken_via_outer_timeout(tmp_path, monkeypatch):
    """End-to-end: ``snapshot_baseline``'s outer ``_loop.run`` timeout
    triggers ``_mark_broken_for_file``, so a second call to
    ``enabled_for`` returns False."""
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
        # Force the inner snapshot coroutine to raise.
        async def boom(_path):
            raise RuntimeError("outer-timeout simulated")

        with patch.object(svc, "_snapshot_async", boom):
            assert svc.enabled_for(str(src)) is True
            svc.snapshot_baseline(str(src))

        # After the failure, the file's pair is in the broken-set and
        # ``enabled_for`` skips it.
        assert ("pyright", str(repo)) in svc._broken
        assert svc.enabled_for(str(src)) is False
    finally:
        svc.shutdown()
