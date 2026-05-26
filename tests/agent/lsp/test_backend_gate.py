"""Integration test: LSP layer is skipped on non-local backends.

The host-side LSP server can't see files inside a Docker/Modal/SSH
sandbox.  When the agent's terminal env isn't ``LocalEnvironment``,
the file_operations layer must skip both ``snapshot_baseline`` and
``get_diagnostics_sync`` calls — falling back to the in-process
syntax check exactly as if LSP were disabled.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

from agent.lsp import eventlog


@pytest.fixture(autouse=True)
def _reset():
    eventlog.reset_announce_caches()


def test_local_only_helper_returns_true_for_local_env():
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    fops = ShellFileOperations(LocalEnvironment(cwd="/tmp"))
    assert fops._lsp_local_only() is True


def test_local_only_helper_returns_false_for_non_local_env():
    """A mocked non-local env (Docker/Modal/SSH stand-in) returns False."""
    from tools.file_operations import ShellFileOperations

    # Build something that's NOT a LocalEnvironment.  We use a bare
    # MagicMock — isinstance() against LocalEnvironment is False.
    fake_env = MagicMock()
    fake_env.execute = MagicMock(return_value=MagicMock(exit_code=0, stdout=""))
    fake_env.cwd = "/sandbox"
    fops = ShellFileOperations(fake_env)
    assert fops._lsp_local_only() is False


def test_snapshot_baseline_skipped_for_non_local(monkeypatch):
    """Verify the LSP service's snapshot_baseline is NOT called when
    the backend isn't local."""
    from tools.file_operations import ShellFileOperations

    fake_env = MagicMock()
    fake_env.execute = MagicMock(return_value=MagicMock(exit_code=0, stdout=""))
    fake_env.cwd = "/sandbox"
    fops = ShellFileOperations(fake_env)

    snapshot_called = []

    class FakeService:
        def snapshot_baseline(self, path):
            snapshot_called.append(path)

    monkeypatch.setattr("agent.lsp.get_service", lambda: FakeService())

    fops._snapshot_lsp_baseline("/sandbox/x.py")
    assert snapshot_called == [], "snapshot must be skipped for non-local backends"


def test_maybe_lsp_diagnostics_returns_empty_for_non_local(monkeypatch):
    from tools.file_operations import ShellFileOperations

    fake_env = MagicMock()
    fake_env.execute = MagicMock(return_value=MagicMock(exit_code=0, stdout=""))
    fake_env.cwd = "/sandbox"
    fops = ShellFileOperations(fake_env)

    called = []

    class FakeService:
        def enabled_for(self, path):
            called.append(("enabled_for", path))
            return True
        def get_diagnostics_sync(self, path, **kw):
            called.append(("get_diagnostics_sync", path))
            return [{"severity": 1, "message": "should not see this"}]

    monkeypatch.setattr("agent.lsp.get_service", lambda: FakeService())

    result = fops._maybe_lsp_diagnostics("/sandbox/x.py")
    assert result == ""
    assert called == [], "service must not be queried for non-local backends"


def test_snapshot_baseline_called_for_local_env(tmp_path, monkeypatch):
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    fops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))

    snapshot_called = []

    class FakeService:
        def snapshot_baseline(self, path):
            snapshot_called.append(path)

    monkeypatch.setattr("agent.lsp.get_service", lambda: FakeService())

    fops._snapshot_lsp_baseline(str(tmp_path / "x.py"))
    assert snapshot_called == [str(tmp_path / "x.py")]
