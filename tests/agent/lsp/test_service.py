"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


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


class _FakeIdleClient:
    def __init__(self, server_id: str = "pyright", workspace_root: str = "/tmp/repo"):
        self.server_id = server_id
        self.workspace_root = workspace_root
        self.state = "running"
        self.is_running = True
        self.shutdown_calls = 0

    async def shutdown(self):
        self.shutdown_calls += 1
        self.is_running = False
        self.state = "stopped"


def test_service_reap_idle_clients_shuts_down_stale_lsp_clients():
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0.1,
    )
    client = _FakeIdleClient()
    key = (client.server_id, client.workspace_root)
    try:
        with svc._state_lock:
            svc._clients[key] = client  # pyright: ignore[reportArgumentType]
            svc._last_used[key] = time.time() - 5.0

        assert svc.reap_idle_clients() == 1
        assert client.shutdown_calls == 1
        assert key not in svc._clients
        assert key not in svc._last_used
    finally:
        svc.shutdown()


def test_service_idle_reaper_runs_without_new_lsp_requests():
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0.05,
    )
    client = _FakeIdleClient()
    key = (client.server_id, client.workspace_root)
    try:
        with svc._state_lock:
            svc._clients[key] = client  # pyright: ignore[reportArgumentType]
            svc._last_used[key] = time.time() - 5.0

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if client.shutdown_calls:
                break
            time.sleep(0.02)

        assert client.shutdown_calls == 1
        assert key not in svc._clients
        assert key not in svc._last_used
    finally:
        svc.shutdown()
