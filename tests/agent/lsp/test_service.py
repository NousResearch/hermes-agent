"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent.lsp.manager import DEFAULT_IDLE_TIMEOUT, LSPService
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _service(**kwargs):
    defaults = {
        "enabled": True,
        "wait_mode": "document",
        "wait_timeout": 2.0,
        "install_strategy": "manual",
    }
    defaults.update(kwargs)
    return LSPService(**defaults)


class FakeClient:
    def __init__(self, server_id: str = "pyright", workspace_root: str = "/tmp/workspace"):
        self.shutdown_called = False
        self.server_id = server_id
        self.workspace_root = workspace_root
        self.is_running = True

    async def shutdown(self):
        self.shutdown_called = True


def _install_mock_server(monkeypatch, script: str = "errors", server_id: str = "pyright"):
    """Replace one registered server with a wrapper that spawns the mock.

    We reuse ``pyright`` so .py files route to it.  This keeps the
    test free of any LSP toolchain dependency.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        env = {"MOCK_LSP_SCRIPT": script}
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env=env,
            initialization_options={},
        )

    replacement = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,  # always use workspace root
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )
    # Patch the SERVERS list element directly + restore on teardown.
    SERVERS[target_index] = replacement

    yield

    SERVERS[target_index] = original


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and create a fake git workspace."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")  # so pyright's root resolver finds it
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(monkeypatch, "errors", "pyright")
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass


def test_service_returns_empty_when_disabled(tmp_path):
    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="auto",
    )
    assert not svc.is_active()
    f = tmp_path / "x.py"
    f.write_text("")
    assert svc.get_diagnostics_sync(str(f)) == []
    svc.shutdown()


def test_service_skips_files_outside_workspace(tmp_path):
    """Files outside any git worktree must not trigger LSP."""
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    f = tmp_path / "x.py"
    f.write_text("")
    # No .git anywhere — service should report not enabled for this file.
    assert not svc.enabled_for(str(f))
    svc.shutdown()


def test_service_e2e_delta_filter(mock_pyright):
    """End-to-end: snapshot baseline → wait → delta returned."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        assert svc.enabled_for(str(f))
        # Baseline first — server pushes 1 error.
        svc.snapshot_baseline(str(f))
        # Re-poll: same error is in baseline, so delta is empty.
        new_diags = svc.get_diagnostics_sync(str(f))
        assert new_diags == []
    finally:
        svc.shutdown()


def test_service_e2e_delta_filter_with_line_shift(mock_pyright):
    """End-to-end: an edit that shifts the diagnostic's line still
    filters correctly when ``line_shift`` is supplied.

    The mock LSP server emits a fixed error at line 0; for this test
    we don't need to actually shift the server's output — we just
    need to prove that supplying a line_shift through the API works
    and doesn't break the existing delta path.  The unit tests in
    test_delta_key.py cover the shift semantics in detail.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.snapshot_baseline(str(f))
        # Identity shift — should behave exactly like no shift.
        new_diags = svc.get_diagnostics_sync(str(f), line_shift=lambda L: L)
        assert new_diags == []
    finally:
        svc.shutdown()


def test_service_status_includes_clients(mock_pyright):
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.get_diagnostics_sync(str(f))
        info = svc.get_status()
        assert info["enabled"] is True
        assert any(c["server_id"] == "pyright" for c in info["clients"])
    finally:
        svc.shutdown()


def test_reap_idle_clients_shuts_down_stale_client(monkeypatch):
    svc = _service(idle_timeout=10)
    client = FakeClient()
    key = (client.server_id, client.workspace_root)
    svc._clients[key] = client
    svc._last_used[key] = 100.0
    monkeypatch.setattr("agent.lsp.manager.time.time", lambda: 111.0)

    try:
        svc._reap_idle_clients()

        assert client.shutdown_called is True
        assert key not in svc._clients
        assert key not in svc._last_used
    finally:
        svc.shutdown()


def test_reap_idle_clients_keeps_recent_client(monkeypatch):
    svc = _service(idle_timeout=10)
    client = FakeClient()
    key = (client.server_id, client.workspace_root)
    svc._clients[key] = client
    svc._last_used[key] = 105.0
    monkeypatch.setattr("agent.lsp.manager.time.time", lambda: 111.0)

    try:
        svc._reap_idle_clients()

        assert client.shutdown_called is False
        assert svc._clients[key] is client
        assert svc._last_used[key] == 105.0
    finally:
        svc.shutdown()


def test_reap_idle_clients_disabled_when_timeout_nonpositive(monkeypatch):
    svc = _service(idle_timeout=0)
    client = FakeClient()
    key = (client.server_id, client.workspace_root)
    svc._clients[key] = client
    svc._last_used[key] = 1.0
    monkeypatch.setattr("agent.lsp.manager.time.time", lambda: 999.0)

    try:
        svc._reap_idle_clients()

        assert client.shutdown_called is False
        assert svc._clients[key] is client
    finally:
        svc.shutdown()


def test_get_or_spawn_sweeps_before_reuse(monkeypatch, mock_pyright):
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = _service()
    called = False

    async def fake_reap():
        nonlocal called
        called = True

    monkeypatch.setattr(svc, "_reap_idle_clients_async", fake_reap)

    try:
        svc.get_diagnostics_sync(str(f))
        assert called is True
    finally:
        svc.shutdown()


def test_create_from_config_reads_idle_timeout(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"lsp": {"enabled": False, "idle_timeout": 42}},
    )

    svc = LSPService.create_from_config()

    assert svc is not None
    assert svc._idle_timeout == 42.0
    assert svc._idle_timeout != DEFAULT_IDLE_TIMEOUT
