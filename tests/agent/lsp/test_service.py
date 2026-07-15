"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
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


def test_idle_reaper_shuts_down_idle_clients(mock_pyright):
    """Clients idle past ``idle_timeout`` must be reaped automatically."""
    import time as _time

    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        lifecycle_enabled=True,
        idle_timeout=0.5,  # 0.5 second — triggers quickly for test
        sweep_interval=0.05,
    )
    try:
        assert svc.enabled_for(str(f))
        svc.get_diagnostics_sync(str(f))
        # Client should be alive now.
        info = svc.get_status()
        assert len(info["clients"]) == 1
        assert info["clients"][0]["running"] is True

        # Wait for idle timeout to expire.
        _time.sleep(1.0)

        # Directly invoke reap — should remove the idle client.
        svc._loop.run(svc._reap_idle_clients(), timeout=5.0)

        # Client should be gone.
        info = svc.get_status()
        assert len(info["clients"]) == 0

        # Intentional retirement is not a permanent broken mark. The next
        # edit cold-starts a fresh generation.
        svc.get_diagnostics_sync(str(f))
        info = svc.get_status()
        assert svc.enabled_for(str(f))
        assert len(info["clients"]) == 1
        assert info["clients"][0]["generation"] == 2
        assert info["clients"][0]["running"] is True
    finally:
        svc.shutdown()


def test_idle_reaper_keeps_active_clients(mock_pyright):
    """Clients that were recently used must NOT be reaped."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        lifecycle_enabled=True,
        idle_timeout=300.0,  # 5 minutes — won't expire
        sweep_interval=60.0,
    )
    try:
        svc.get_diagnostics_sync(str(f))
        info = svc.get_status()
        assert len(info["clients"]) == 1

        # Directly invoke reap — should NOT remove the active client.
        svc._loop.run(svc._reap_idle_clients(), timeout=5.0)

        info = svc.get_status()
        assert len(info["clients"]) == 1
        assert info["clients"][0]["running"] is True
    finally:
        svc.shutdown()


def test_concurrent_same_workspace_requests_share_one_spawn(mock_pyright):
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        lifecycle_enabled=True,
        idle_timeout=300.0,
        sweep_interval=60.0,
    )
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda _: svc.get_diagnostics_sync(str(f)), range(8))
            )
        assert all(isinstance(result, list) for result in results)
        status = svc.get_status()
        assert len(status["clients"]) == 1
        assert status["clients"][0]["generation"] == 1
    finally:
        svc.shutdown()


def test_capacity_retires_lru_before_spawning_replacement(
    tmp_path, mock_pyright
):
    files = []
    for index in range(3):
        repo = tmp_path / f"repo-{index}"
        repo.mkdir()
        (repo / ".git").mkdir()
        file_path = repo / "x.py"
        file_path.write_text("print('hi')\n")
        files.append(file_path)

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        lifecycle_enabled=True,
        idle_timeout=0.0,
        sweep_interval=60.0,
        max_clients=2,
    )
    try:
        for file_path in files:
            assert len(svc.get_diagnostics_sync(str(file_path))) == 1
        status = svc.get_status()
        assert len(status["clients"]) == 2
        assert status["lifecycle"]["capacity_evictions"] == 1
        roots = {client["workspace_root"] for client in status["clients"]}
        assert str(files[0].parent.resolve()) not in roots
        assert str(files[1].parent.resolve()) in roots
        assert str(files[2].parent.resolve()) in roots
    finally:
        svc.shutdown()
