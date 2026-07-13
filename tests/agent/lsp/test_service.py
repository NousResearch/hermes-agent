"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from agent.lsp import client as client_module
from agent.lsp import manager as manager_module
from agent.lsp.client import LSPClient
from agent.lsp.manager import LSPService
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")

_BLOCKED_START_LEADER = """
import os
import subprocess
import sys
import time

child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
with open(os.environ["LSP_TEST_LEADER_PID"], "w", encoding="ascii") as fh:
    fh.write(str(os.getpid()))
with open(os.environ["LSP_TEST_CHILD_PID"], "w", encoding="ascii") as fh:
    fh.write(str(child.pid))
time.sleep(60)
"""


def _wait_for_pid_file(path: Path) -> int:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            return int(path.read_text(encoding="ascii"))
        except (FileNotFoundError, ValueError):
            time.sleep(0.01)
    raise AssertionError(f"{path.name} was not created")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text(encoding="ascii").split()[2] != "Z"
        except (OSError, IndexError):
            pass
    return True


def _wait_for_pid_exit(pid: int) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.01)
    raise AssertionError(f"process {pid} survived service shutdown")


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


def _concurrent_spawn_service(tmp_path: Path, monkeypatch, client_type):
    root = str(tmp_path)
    source = str(tmp_path / "x.py")
    server = next(s for s in SERVERS if s.server_id == "pyright")
    spawn = SpawnSpec(
        command=["test-lsp"],
        workspace_root=root,
        cwd=root,
        env={},
        initialization_options={},
    )
    monkeypatch.setattr(manager_module, "find_server_for_file", lambda _path: server)
    monkeypatch.setattr(
        manager_module, "resolve_workspace_for_file", lambda _path: (root, True)
    )
    monkeypatch.setattr(server, "resolve_root", lambda _path, _root: root)
    monkeypatch.setattr(server, "build_spawn", lambda _root, _ctx: spawn)
    monkeypatch.setattr(manager_module, "LSPClient", client_type)
    service = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    return service, source, (server.server_id, root)


@pytest.mark.asyncio
async def test_concurrent_spawn_waiter_is_cancelled_when_owner_is_cancelled(
    tmp_path: Path, monkeypatch
):
    started = asyncio.Event()

    class CancelledClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            started.set()
            await asyncio.Event().wait()

    service, source, key = _concurrent_spawn_service(
        tmp_path, monkeypatch, CancelledClient
    )
    owner = asyncio.create_task(service._get_or_spawn(source))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    waiter = asyncio.create_task(service._get_or_spawn(source))
    await asyncio.sleep(0)

    owner.cancel()
    results = await asyncio.wait_for(
        asyncio.gather(owner, waiter, return_exceptions=True), timeout=1.0
    )

    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert key not in service._spawning


@pytest.mark.asyncio
async def test_concurrent_spawn_failure_resolves_all_waiters(tmp_path: Path, monkeypatch):
    started = asyncio.Event()
    fail = asyncio.Event()

    class FailingClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            started.set()
            await fail.wait()
            raise RuntimeError("spawn failed")

    service, source, key = _concurrent_spawn_service(
        tmp_path, monkeypatch, FailingClient
    )
    owner = asyncio.create_task(service._get_or_spawn(source))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    waiter = asyncio.create_task(service._get_or_spawn(source))
    await asyncio.sleep(0)
    fail.set()

    assert await asyncio.wait_for(asyncio.gather(owner, waiter), timeout=1.0) == [
        None,
        None,
    ]
    assert key not in service._spawning
    assert key in service._broken


@pytest.mark.asyncio
async def test_shutdown_cancels_in_flight_spawn_and_settles_waiters(
    tmp_path: Path, monkeypatch
):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    shut_down = asyncio.Event()

    class BlockedClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def shutdown(self):
            shut_down.set()

    service, source, key = _concurrent_spawn_service(
        tmp_path, monkeypatch, BlockedClient
    )
    owner = asyncio.create_task(service._get_or_spawn(source))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    waiter = asyncio.create_task(service._get_or_spawn(source))
    await asyncio.sleep(0)

    try:
        await asyncio.wait_for(service._shutdown_async(), timeout=1.0)

        results = await asyncio.wait_for(
            asyncio.gather(owner, waiter, return_exceptions=True), timeout=1.0
        )
        assert cancelled.is_set()
        assert shut_down.is_set()
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert key not in service._spawning
    finally:
        for task in (owner, waiter):
            if not task.done():
                task.cancel()
        await asyncio.gather(owner, waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_spawn_finishing_after_shutdown_begins_cleans_unpublished_client(
    tmp_path: Path, monkeypatch
):
    started = asyncio.Event()
    finish_start = asyncio.Event()
    shut_down = asyncio.Event()

    class FinishingClient:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            started.set()
            await finish_start.wait()

        async def shutdown(self):
            shut_down.set()

    service, source, key = _concurrent_spawn_service(
        tmp_path, monkeypatch, FinishingClient
    )
    owner = asyncio.create_task(service._get_or_spawn(source))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    waiter = asyncio.create_task(service._get_or_spawn(source))
    await asyncio.sleep(0)

    with service._state_lock:
        service._shutting_down = True
    finish_start.set()

    results = await asyncio.wait_for(
        asyncio.gather(owner, waiter, return_exceptions=True), timeout=1.0
    )
    assert results[0] is None
    assert isinstance(results[1], asyncio.CancelledError)
    assert shut_down.is_set()
    assert key not in service._clients
    assert key not in service._spawning


@pytest.mark.asyncio
async def test_mark_broken_leaves_registered_client_owned_by_shutdown(
    tmp_path: Path, monkeypatch
):
    shut_down = asyncio.Event()

    class RegisteredClient:
        async def shutdown(self):
            shut_down.set()

    service, source, key = _concurrent_spawn_service(
        tmp_path, monkeypatch, RegisteredClient
    )
    client = RegisteredClient()
    with service._state_lock:
        service._clients[key] = client
        service._shutting_down = True

    service._mark_broken_for_file(source, asyncio.TimeoutError())

    assert service._clients[key] is client
    await asyncio.wait_for(service._shutdown_async(), timeout=1.0)
    assert shut_down.is_set()
    assert key not in service._clients


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or client_module._PIDFD_OPEN is None,
    reason="process-tree service shutdown requires Linux pidfds",
)
def test_service_shutdown_contains_real_in_flight_process_tree(
    tmp_path: Path, monkeypatch
):
    root = str(tmp_path)
    source = str(tmp_path / "x.py")
    leader_pid_file = tmp_path / "leader.pid"
    child_pid_file = tmp_path / "child.pid"
    server = next(s for s in SERVERS if s.server_id == "pyright")
    spawn = SpawnSpec(
        command=[sys.executable, "-c", _BLOCKED_START_LEADER],
        workspace_root=root,
        cwd=root,
        env={
            "LSP_TEST_LEADER_PID": str(leader_pid_file),
            "LSP_TEST_CHILD_PID": str(child_pid_file),
        },
        initialization_options={},
    )

    class BlockedStartClient(LSPClient):
        async def _initialize(self):
            await asyncio.Event().wait()

    monkeypatch.setattr(client_module, "SHUTDOWN_GRACE", 0.2)
    monkeypatch.setattr(manager_module, "find_server_for_file", lambda _path: server)
    monkeypatch.setattr(
        manager_module, "resolve_workspace_for_file", lambda _path: (root, True)
    )
    monkeypatch.setattr(server, "resolve_root", lambda _path, _root: root)
    monkeypatch.setattr(server, "build_spawn", lambda _root, _ctx: spawn)
    monkeypatch.setattr(manager_module, "LSPClient", BlockedStartClient)
    service = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    owner = None
    waiter = None
    leader_pid = None
    child_pid = None
    try:
        loop = service._loop._loop
        assert loop is not None
        owner = asyncio.run_coroutine_threadsafe(service._get_or_spawn(source), loop)
        leader_pid = _wait_for_pid_file(leader_pid_file)
        child_pid = _wait_for_pid_file(child_pid_file)
        waiter = asyncio.run_coroutine_threadsafe(service._get_or_spawn(source), loop)
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=1.0)

        service.shutdown()

        assert owner.done()
        assert waiter.done()
        assert owner.cancelled() or owner.result() is None
        assert waiter.cancelled() or waiter.result() is None
        assert service._spawning == {}
        assert service._loop._loop is None
        assert service._loop._thread is None
        _wait_for_pid_exit(leader_pid)
        _wait_for_pid_exit(child_pid)
        service.shutdown()
    finally:
        service.shutdown()
        for pid in (child_pid, leader_pid):
            if pid is not None and _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
