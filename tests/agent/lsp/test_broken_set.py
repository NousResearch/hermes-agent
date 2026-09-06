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

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import SpawnSpec
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
    (repo / "pyproject.toml").write_text(
        "[project]\nname='t'\n", encoding="utf-8"
    )
    return repo








def test_unrelated_project_not_affected_by_broken(tmp_path, monkeypatch):
    """Marking pyright broken for project A must NOT affect project B."""
    repo_a = _make_git_workspace(tmp_path)
    repo_b = tmp_path / "repo-b"
    repo_b.mkdir()
    (repo_b / ".git").mkdir()
    (repo_b / "pyproject.toml").write_text(
        "[project]\nname='b'\n", encoding="utf-8"
    )
    a_src = repo_a / "x.py"
    a_src.write_text("", encoding="utf-8")
    b_src = repo_b / "x.py"
    b_src.write_text("", encoding="utf-8")

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
        assert svc.shutdown() is True




def test_mark_broken_handles_no_workspace_silently(tmp_path, monkeypatch):
    """File outside any git worktree → no workspace → no key to add."""
    src = tmp_path / "orphan.py"
    src.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "agent.lsp.manager.resolve_workspace_for_file",
        lambda _path: (None, False),
    )
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
        assert svc.shutdown() is True


def test_snapshot_failure_marks_broken_via_outer_timeout(tmp_path, monkeypatch):
    """End-to-end: ``snapshot_baseline``'s outer ``_loop.run`` timeout
    triggers ``_mark_broken_for_file``, so a second call to
    ``enabled_for`` returns False."""
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    src = repo / "x.py"
    src.write_text("", encoding="utf-8")

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
        assert svc.shutdown() is True


@pytest.mark.asyncio
async def test_spawn_failure_stays_permanently_broken(monkeypatch, tmp_path):
    """Lifecycle retirement must not reintroduce cooldown retries."""
    from agent.lsp import manager as manager_module

    repo = _make_git_workspace(tmp_path)
    source = repo / "x.py"
    source.write_text("", encoding="utf-8")
    starts = {"count": 0}

    class FakeServer:
        server_id = "pyright"
        seed_first_push = False

        @staticmethod
        def resolve_root(file_path, workspace_root):
            return workspace_root

        @staticmethod
        def build_spawn(root, ctx):
            return SpawnSpec(
                command=["fake-lsp"],
                workspace_root=root,
                cwd=root,
                env={},
                initialization_options={},
            )

    class FailingClient:
        def __init__(self, **kwargs):
            self.state = "stopped"

        @property
        def is_running(self):
            return False

        async def start(self):
            starts["count"] += 1
            self.state = "error"
            raise RuntimeError("wedged forever")

        async def shutdown(self):
            self.state = "stopped"

    monkeypatch.setattr(manager_module, "find_server_for_file", lambda path: FakeServer())
    monkeypatch.setattr(
        manager_module,
        "resolve_workspace_for_file",
        lambda path: (str(repo), True),
    )
    monkeypatch.setattr(manager_module, "LSPClient", FailingClient)

    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        idle_timeout=0,
    )
    svc._enabled = True
    svc._admitting = True
    svc._shutdown_state = "running"
    try:
        assert await svc._acquire_client(str(source)) is None
        assert ("pyright", str(repo)) in svc._broken

        # No elapsed-time or reap transition can make this key retry.
        await svc._reap_idle_once()
        assert await svc._acquire_client(str(source)) is None
        assert starts["count"] == 1
    finally:
        assert await svc._shutdown_async() is True
