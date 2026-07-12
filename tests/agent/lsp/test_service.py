"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

from agent.lsp.manager import LSPService


class _IdleClient:
    def __init__(self):
        self.shutdown_calls = 0
        self.shutdown_event = threading.Event()

    async def shutdown(self):
        self.shutdown_calls += 1
        self.shutdown_event.set()


def test_reap_idle_clients_closes_and_forgets_stale_workspaces(monkeypatch):
    """Long-lived gateways must not retain one server per old workspace."""
    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1,
        install_strategy="manual",
        idle_timeout=10,
    )
    stale = _IdleClient()
    fresh = _IdleClient()
    stale_key = ("typescript", "/old/project")
    fresh_key = ("typescript", "/active/project")
    svc._clients = {stale_key: stale, fresh_key: fresh}
    svc._last_used = {stale_key: 80, fresh_key: 95}
    monkeypatch.setattr("agent.lsp.manager.time.time", lambda: 100)

    reaped = __import__("asyncio").run(svc._reap_idle_async())

    assert reaped == 1
    assert stale.shutdown_calls == 1
    assert fresh.shutdown_calls == 0
    assert stale_key not in svc._clients
    assert stale_key not in svc._last_used
    assert fresh_key in svc._clients


def test_background_reaper_runs_without_a_later_tool_call():
    """An abandoned workspace is reaped while the gateway is otherwise idle."""
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1,
        install_strategy="manual",
        idle_timeout=0.02,
    )
    client = _IdleClient()
    key = ("typescript", "/abandoned/project")
    with svc._state_lock:
        svc._clients[key] = client
        svc._last_used[key] = 0
    try:
        assert client.shutdown_event.wait(1.0), "background reaper did not run"
        assert client.shutdown_calls == 1
        assert key not in svc._clients
    finally:
        svc.shutdown()


def test_reap_idle_clients_does_not_hold_state_lock_during_shutdown(monkeypatch):
    """A slow language-server exit must not block unrelated client lookups."""
    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1,
        install_strategy="manual",
        idle_timeout=10,
    )

    class _LockCheckingClient(_IdleClient):
        async def shutdown(self):
            assert svc._state_lock.acquire(blocking=False)
            svc._state_lock.release()
            await super().shutdown()

    client = _LockCheckingClient()
    key = ("typescript", "/old/project")
    svc._clients[key] = client
    svc._last_used[key] = 0
    monkeypatch.setattr("agent.lsp.manager.time.time", lambda: 100)

    asyncio.run(svc._reap_idle_async())
    assert client.shutdown_calls == 1


def test_request_crossing_idle_timeout_is_not_reaped(monkeypatch):
    """The timeout measures completed idle time, not request duration."""
    svc = LSPService(enabled=False, wait_mode="document", wait_timeout=1,
                     install_strategy="manual", idle_timeout=10)

    class _BusyClient(_IdleClient):
        server_id = "typescript"
        workspace_root = "/project"
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.finish = asyncio.Event()
        async def open_file(self, *_args, **_kwargs):
            self.started.set()
            await self.finish.wait()
            return 1
        async def wait_for_diagnostics(self, *_args, **_kwargs):
            pass
        def diagnostics_for(self, _file_path):
            return []

    client = _BusyClient()
    key = (client.server_id, client.workspace_root)
    svc._clients[key] = client
    svc._last_used[key] = 80

    async def fake_get_or_spawn(_file_path):
        return client

    monkeypatch.setattr(svc, "_get_or_spawn", fake_get_or_spawn)
    monkeypatch.setattr("agent.lsp.manager.time.time", lambda: 100)

    async def exercise():
        request = asyncio.create_task(svc._snapshot_async("/project/a.ts"))
        await client.started.wait()
        assert await svc._reap_idle_async() == 0
        assert client.shutdown_calls == 0
        client.finish.set()
        await request
        assert svc._last_used[key] == 100

    asyncio.run(exercise())


def test_shutdown_waits_for_reaper_critical_section():
    svc = LSPService(enabled=True, wait_mode="document", wait_timeout=1,
                     install_strategy="manual", idle_timeout=60)
    client = _IdleClient()
    key = ("typescript", "/project")
    with svc._state_lock:
        svc._clients[key] = client
        svc._last_used[key] = 0
    svc._reaper_run_lock.acquire()
    thread = threading.Thread(target=svc.shutdown)
    thread.start()
    time.sleep(0.05)
    assert thread.is_alive()
    assert client.shutdown_calls == 0
    svc._reaper_run_lock.release()
    thread.join(2)
    assert not thread.is_alive()
    assert client.shutdown_calls == 1


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
