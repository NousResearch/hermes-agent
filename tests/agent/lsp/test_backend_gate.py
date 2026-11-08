"""Integration test: LSP layer is skipped on non-local backends.

The host-side LSP server can't see files inside a Docker/Modal/SSH
sandbox.  When the agent's terminal env isn't ``LocalEnvironment``,
the file_operations layer must not query the service at all — falling back to the in-process
syntax check exactly as if LSP were disabled.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.lsp import eventlog


@pytest.fixture(autouse=True)
def _reset():
    eventlog.reset_announce_caches()








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


def test_edit_check_started_for_local_env(tmp_path, monkeypatch):
    """The local backend hands the written file and its pre-/post-edit text to the service's edit check."""
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    fops = ShellFileOperations(LocalEnvironment(cwd=str(tmp_path)))
    started = []

    class FakeService:
        def enabled_for(self, path):
            return True

        def start_edit_check(self, path, *, pre_text, post_text=None, owner=None, carry=()):
            started.append((path, pre_text, post_text))

    monkeypatch.setattr("agent.lsp.get_service", lambda: FakeService())

    assert fops._maybe_lsp_diagnostics(str(tmp_path / "x.py"), pre_content="x = 1\n", post_content="x = 2\n") == ""
    assert started == [(str(tmp_path / "x.py"), "x = 1\n", "x = 2\n")]
