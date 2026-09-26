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

import time
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
    (repo / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    return repo








def test_unrelated_project_not_affected_by_broken(tmp_path, monkeypatch):
    """Marking pyright broken for project A must NOT affect project B."""
    repo_a = _make_git_workspace(tmp_path)
    repo_b = tmp_path / "repo-b"
    repo_b.mkdir()
    (repo_b / ".git").mkdir()
    (repo_b / "pyproject.toml").write_text("[project]\nname='b'\n", encoding="utf-8")
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
        svc.shutdown()




def test_mark_broken_handles_no_workspace_silently(tmp_path):
    """File outside any git worktree → no workspace → no key to add."""
    src = tmp_path / "orphan.py"
    src.write_text("", encoding="utf-8")
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
        svc.shutdown()


def test_skipped_request_on_broken_root_is_logged_at_info(tmp_path, monkeypatch, caplog):
    """A request skipped because its root is broken must leave a visible trace (#116446): at default
    levels ``log_clean`` is DEBUG, so a silently skipped file was indistinguishable from a clean one."""
    from agent.lsp import eventlog

    repo = _make_git_workspace(tmp_path)
    src = repo / "x.py"
    src.write_text("", encoding="utf-8")
    monkeypatch.chdir(str(repo))
    eventlog.reset_announce_caches()
    svc = LSPService(enabled=True, wait_mode="document", wait_timeout=2.0, install_strategy="manual")
    try:
        svc._mark_broken_for_file(str(src), RuntimeError("simulated"))
        with caplog.at_level("INFO", logger=eventlog.event_log.name):
            assert svc.get_diagnostics_sync(str(src)) == []
            assert svc.get_diagnostics_sync(str(src)) == []
    finally:
        svc.shutdown()
    skipped = [r for r in caplog.records if "marked broken" in r.getMessage()]
    assert [r.levelname for r in skipped] == ["INFO"]  # once per root; the repeat is DEBUG
    assert str(repo) in skipped[0].getMessage() and "x.py" in skipped[0].getMessage()


def test_broken_root_is_retried_after_broken_retry_seconds(tmp_path, monkeypatch):
    """With ``lsp.broken_retry_seconds`` set, a root poisoned by one outer-timeout failure on the pre-write
    path is re-tried once the window passes instead of staying dark for the process lifetime (#116446).
    The default (0) keeps the lifetime behaviour."""
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    src = repo / "x.py"
    src.write_text("", encoding="utf-8")
    cfg = {"lsp": {"wait_timeout": 1.0, "install_strategy": "manual", "broken_retry_seconds": 0.2}}
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        svc = LSPService.create_from_config()
    assert svc is not None
    try:
        async def boom(*_a, **_k):
            raise RuntimeError("outer-timeout simulated")

        with patch.object(svc, "_snapshot_async", boom):
            svc.snapshot_baseline(str(src))
        assert svc.enabled_for(str(src)) is False
        assert ("pyright", str(repo)) in svc.get_status()["broken"]
        time.sleep(0.3)
        assert svc.enabled_for(str(src)) is True
        assert svc.get_status()["broken"] == []
    finally:
        svc.shutdown()
