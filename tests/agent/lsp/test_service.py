"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import sys
import threading
import time
from pathlib import Path

import pytest

from agent.lsp import eventlog
from agent.lsp.client import LSPClient
from agent.lsp.manager import LSPService, _ClientEntry
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _make_repo(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    source = repo / "x.py"
    source.write_text("print('hi')\n", encoding="utf-8")
    monkeypatch.chdir(str(repo))
    return repo, source


def _threadless_service() -> LSPService:
    """Build async manager state without the sync background-loop bridge.

    Lifecycle race tests run directly on pytest's event loop.  The production
    bridge is covered by the existing synchronous service tests; keeping race
    orchestration on one loop makes event ordering deterministic.
    """
    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=5.0,
        install_strategy="manual",
        idle_timeout=0,
    )
    svc._enabled = True
    svc._admitting = True
    svc._shutdown_state = "running"
    return svc


def _install_mock_server(
    monkeypatch, script: str | list[str] = "errors", server_id: str = "pyright"
):
    """Replace one registered server with a wrapper that spawns the mock.

    We reuse ``pyright`` so .py files route to it.  This keeps the
    test free of any LSP toolchain dependency.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]
    scripts = [script] if isinstance(script, str) else script
    spawn_count = {"value": 0}

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        index = min(spawn_count["value"], len(scripts) - 1)
        spawn_count["value"] += 1
        env = {"MOCK_LSP_SCRIPT": scripts[index]}
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

    yield spawn_count

    SERVERS[target_index] = original


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and create a fake git workspace."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    # Marker so pyright's root resolver finds the workspace.
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(monkeypatch, "errors", "pyright")
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass






def test_service_e2e_delta_filter(mock_pyright):
    """End-to-end: snapshot baseline → wait → delta returned."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n", encoding="utf-8")

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
        assert svc.shutdown() is True


@pytest.mark.parametrize("failed_script", ["clean_eof", "malformed_frame"])
def test_service_replaces_client_after_reader_failure(
    tmp_path, monkeypatch, failed_script
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    source = repo / "x.py"
    source.write_text("print('hi')\n", encoding="utf-8")
    monkeypatch.chdir(str(repo))
    server = _install_mock_server(
        monkeypatch, [failed_script, "clean"], "pyright"
    )
    spawn_count = next(server)

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.5,
        install_strategy="manual",
    )
    try:
        async def _break_first_client():
            lease = await svc._acquire_client(str(source))
            assert lease is not None
            async with lease as client:
                reader_task = client._reader_task
                assert reader_task is not None
                await client.open_file(str(source), language_id="python")
                await asyncio.wait_for(asyncio.shield(reader_task), timeout=3.0)
                return client

        async def _get_replacement():
            lease = await svc._acquire_client(str(source))
            assert lease is not None
            async with lease as client:
                return client

        first = svc._loop.run(_break_first_client(), timeout=5.0)
        replacement = svc._loop.run(_get_replacement(), timeout=5.0)

        assert not first.is_running
        assert replacement is not None
        assert replacement is not first
        assert replacement.is_running
        assert spawn_count["value"] == 2
    finally:
        assert svc.shutdown() is True
        try:
            next(server)
        except StopIteration:
            pass


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
    f.write_text("print('hi')\n", encoding="utf-8")

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
        assert svc.shutdown() is True






def test_reused_client_refreshes_last_used_and_survives_reap(mock_pyright):
    """A client re-acquired from the cache must have its ``_last_used``
    timestamp refreshed so a subsequent sweep does NOT evict it.

    Covers the timestamp refresh on the existing-client fast path in
    ``_acquire_client`` — without it, a client in constant use would be
    reaped ``idle_timeout`` seconds after its FIRST use.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("", encoding="utf-8")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=60.0,  # sweeps manually below; loop never fires
    )
    try:
        svc.get_diagnostics_sync(str(f))
        key = next(iter(svc._clients))
        first_used = svc._last_used[key]

        # Age the timestamp past the cutoff, then re-acquire the client.
        svc._last_used[key] = first_used - 120.0
        svc.get_diagnostics_sync(str(f))
        assert svc._last_used[key] > first_used - 120.0, (
            "re-acquiring a cached client must refresh _last_used"
        )

        # A sweep right after reuse must keep the client.
        svc._loop.run(svc._reap_idle_once(), timeout=5.0)
        assert key in svc._clients
        assert svc.get_status()["clients"]
    finally:
        assert svc.shutdown() is True


def test_reaper_survives_sweep_error(mock_pyright):
    """One failing sweep must not kill the reaper loop — the loop's
    ``except Exception`` guard must swallow the error and keep sweeping."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("", encoding="utf-8")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0.1,
    )
    try:
        # Sabotage the sweep itself so the reaper-loop except branch
        # actually runs (a failing client.shutdown() would be swallowed
        # by gather(return_exceptions=True) and never reach the loop).
        calls = {"n": 0}
        real_reap = svc._reap_idle_once

        async def _flaky_reap():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("sweep sabotage")
            await real_reap()

        svc._reap_idle_once = _flaky_reap  # type: ignore[method-assign]

        svc.get_diagnostics_sync(str(f))
        assert svc.get_status()["clients"]

        # First sweep raises; later sweeps must still reap the client.
        deadline = time.monotonic() + 3.0
        while svc.get_status()["clients"] and time.monotonic() < deadline:
            time.sleep(0.02)

        assert calls["n"] >= 2, "reaper loop died after the failing sweep"
        assert svc.get_status()["clients"] == []
        assert svc._idle_reaper_task is not None
        assert not svc._idle_reaper_task.done()
    finally:
        assert svc.shutdown() is True


@pytest.mark.asyncio
async def test_active_diagnostics_lease_survives_idle_reap(tmp_path, monkeypatch):
    """An idle sweep may retire, but cannot kill, an active wait."""
    _repo, source = _make_repo(tmp_path, monkeypatch)
    clients = []

    class ControlledClient:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.state = "stopped"
            self.wait_started = asyncio.Event()
            self.allow_wait = asyncio.Event()
            self.shutdown_started = asyncio.Event()
            self.allow_shutdown = asyncio.Event()
            clients.append(self)

        @property
        def is_running(self):
            return self.state == "running"

        async def start(self):
            self.state = "running"

        async def open_file(self, path, *, language_id):
            return 1

        async def save_file(self, path):
            return None

        async def wait_for_diagnostics(self, path, version, **kwargs):
            self.wait_started.set()
            await self.allow_wait.wait()
            return True

        def diagnostics_for(self, path, *, fresh_only=False):
            return []

        async def shutdown(self):
            self.shutdown_started.set()
            await self.allow_shutdown.wait()
            self.state = "stopped"

    server = _install_mock_server(monkeypatch, "clean", "pyright")
    next(server)
    monkeypatch.setattr("agent.lsp.manager.LSPClient", ControlledClient)
    svc = _threadless_service()
    operation = None
    reap = None
    try:
        operation = asyncio.create_task(svc._open_and_wait_async(str(source)))
        while not clients:
            await asyncio.sleep(0)
        client = clients[0]
        await client.wait_started.wait()
        key, entry = next(iter(svc._clients.items()))
        svc._last_used[key] = 0

        reap = asyncio.create_task(svc._reap_idle_once())
        while not entry.retiring:
            await asyncio.sleep(0)

        assert entry.leases == 1
        assert not client.shutdown_started.is_set()
        assert not operation.done()

        client.allow_wait.set()
        assert await operation == []
        await client.shutdown_started.wait()
        client.allow_shutdown.set()
        await reap

        assert entry.leases == 0
        assert svc._clients == {}
    finally:
        for client in clients:
            client.allow_wait.set()
            client.allow_shutdown.set()
        retained = [task for task in (operation, reap) if task is not None]
        if retained:
            await asyncio.gather(*retained, return_exceptions=True)
        assert await svc._shutdown_async() is True
        try:
            next(server)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_crashed_generation_retires_before_replacement_spawn(
    tmp_path, monkeypatch
):
    _repo, source = _make_repo(tmp_path, monkeypatch)
    clients = []

    class ControlledClient:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.state = "stopped"
            self.shutdown_started = asyncio.Event()
            self.allow_shutdown = asyncio.Event()
            if clients:
                self.allow_shutdown.set()
            clients.append(self)

        @property
        def is_running(self):
            return self.state == "running"

        async def start(self):
            self.state = "running"

        async def shutdown(self):
            self.shutdown_started.set()
            await self.allow_shutdown.wait()
            self.state = "stopped"

    server = _install_mock_server(monkeypatch, ["clean", "clean"], "pyright")
    spawn_count = next(server)
    monkeypatch.setattr("agent.lsp.manager.LSPClient", ControlledClient)
    svc = _threadless_service()
    replacement_task = None
    try:
        first_lease = await svc._acquire_client(str(source))
        assert first_lease is not None
        first = first_lease.client
        assert first_lease.generation == 1
        first_lease.release()
        first.state = "error"
        entry = next(iter(svc._clients.values()))

        replacement_task = asyncio.create_task(svc._acquire_client(str(source)))
        await first.shutdown_started.wait()

        assert len(clients) == 1
        assert spawn_count["value"] == 1
        assert not replacement_task.done()
        assert entry.generation == 1
        assert entry.retiring is True

        first.allow_shutdown.set()
        replacement_lease = await replacement_task
        assert replacement_lease is not None
        assert replacement_lease.generation == 2
        assert len(clients) == 2
        assert spawn_count["value"] == 2
        replacement_lease.release()
    finally:
        for client in clients:
            client.allow_shutdown.set()
        if replacement_task is not None:
            await asyncio.gather(replacement_task, return_exceptions=True)
        assert await svc._shutdown_async() is True
        try:
            next(server)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_shutdown_waits_for_active_lease(tmp_path, monkeypatch):
    _repo, source = _make_repo(tmp_path, monkeypatch)
    clients = []

    class ControlledClient:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.state = "stopped"
            self.shutdown_started = asyncio.Event()
            self.allow_shutdown = asyncio.Event()
            clients.append(self)

        @property
        def is_running(self):
            return self.state == "running"

        async def start(self):
            self.state = "running"

        async def shutdown(self):
            self.shutdown_started.set()
            await self.allow_shutdown.wait()
            self.state = "stopped"

    server = _install_mock_server(monkeypatch, "clean", "pyright")
    next(server)
    monkeypatch.setattr("agent.lsp.manager.LSPClient", ControlledClient)
    svc = _threadless_service()
    shutdown = None
    try:
        lease = await svc._acquire_client(str(source))
        assert lease is not None
        client = lease.client
        entry = next(iter(svc._clients.values()))

        shutdown = asyncio.create_task(svc._shutdown_async())
        while not entry.retiring:
            await asyncio.sleep(0)

        assert entry.leases == 1
        assert not client.shutdown_started.is_set()
        assert not shutdown.done()

        lease.release()
        await client.shutdown_started.wait()
        assert not shutdown.done()
        client.allow_shutdown.set()
        assert await shutdown is True
        assert svc._shutdown_state == "closed"
    finally:
        for client in clients:
            client.allow_shutdown.set()
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)
        assert await svc._shutdown_async() is True
        try:
            next(server)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_shutdown_cancels_inflight_spawn_and_waits_for_cleanup(
    tmp_path, monkeypatch
):
    _repo, source = _make_repo(tmp_path, monkeypatch)
    clients = []

    class ControlledClient:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.state = "stopped"
            self.start_entered = asyncio.Event()
            self.never_finish_start = asyncio.Event()
            self.shutdown_started = asyncio.Event()
            self.allow_shutdown = asyncio.Event()
            clients.append(self)

        @property
        def is_running(self):
            return self.state == "running"

        async def start(self):
            self.state = "starting"
            self.start_entered.set()
            await self.never_finish_start.wait()
            self.state = "running"

        async def shutdown(self):
            self.shutdown_started.set()
            await self.allow_shutdown.wait()
            self.state = "stopped"

    server = _install_mock_server(monkeypatch, "clean", "pyright")
    next(server)
    monkeypatch.setattr("agent.lsp.manager.LSPClient", ControlledClient)
    svc = _threadless_service()
    acquire = None
    shutdown = None
    try:
        acquire = asyncio.create_task(svc._acquire_client(str(source)))
        while not clients:
            await asyncio.sleep(0)
        client = clients[0]
        await client.start_entered.wait()

        shutdown = asyncio.create_task(svc._shutdown_async())
        await client.shutdown_started.wait()
        assert not shutdown.done()
        assert svc._admitting is False

        client.allow_shutdown.set()
        assert await shutdown is True
        acquire_result = await asyncio.gather(acquire, return_exceptions=True)
        assert isinstance(acquire_result[0], asyncio.CancelledError)
        assert svc._spawning == {}
    finally:
        for client in clients:
            client.never_finish_start.set()
            client.allow_shutdown.set()
        retained = [task for task in (acquire, shutdown) if task is not None]
        if retained:
            await asyncio.gather(*retained, return_exceptions=True)
        assert await svc._shutdown_async() is True
        try:
            next(server)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_failed_service_and_entry_tasks_retry_once_for_concurrent_callers(
    tmp_path, monkeypatch
):
    _repo, source = _make_repo(tmp_path, monkeypatch)
    clients = []

    class TransientShutdownClient:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.state = "stopped"
            self.shutdown_calls = 0
            self.retry_started = asyncio.Event()
            self.allow_retry = asyncio.Event()
            clients.append(self)

        @property
        def is_running(self):
            return self.state == "running"

        async def start(self):
            self.state = "running"

        async def shutdown(self):
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                self.state = "error"
                raise RuntimeError("transient cleanup failure")
            if self.shutdown_calls > 2:
                raise AssertionError("concurrent retirement retry")
            self.retry_started.set()
            await self.allow_retry.wait()
            self.state = "stopped"

    server = _install_mock_server(monkeypatch, "clean", "pyright")
    next(server)
    monkeypatch.setattr("agent.lsp.manager.LSPClient", TransientShutdownClient)
    svc = _threadless_service()
    retry_one = None
    retry_two = None
    try:
        lease = await svc._acquire_client(str(source))
        assert lease is not None
        lease.release()
        entry = next(iter(svc._clients.values()))
        client = clients[0]

        assert await svc._shutdown_async() is False
        failed_service_owner = svc._shutdown_task
        failed_entry_owner = entry.retirement_task
        assert svc._shutdown_state == "failed"
        assert svc._admitting is False
        assert entry.retiring is True
        assert entry.retirement_error == (
            "RuntimeError: transient cleanup failure"
        )
        assert await svc._acquire_client(str(source)) is None

        retry_one = asyncio.create_task(svc._shutdown_async())
        retry_two = asyncio.create_task(svc._shutdown_async())
        await client.retry_started.wait()

        assert svc._shutdown_task is not failed_service_owner
        assert entry.retirement_task is not failed_entry_owner
        assert client.shutdown_calls == 2
        assert not retry_one.done()
        assert not retry_two.done()

        client.allow_retry.set()
        assert await asyncio.gather(retry_one, retry_two) == [True, True]
        assert client.shutdown_calls == 2
        assert svc._clients == {}
        assert svc._shutdown_state == "closed"
    finally:
        for client in clients:
            client.allow_retry.set()
        retained = [task for task in (retry_one, retry_two) if task is not None]
        if retained:
            await asyncio.gather(*retained, return_exceptions=True)
        assert await svc._shutdown_async() is True
        try:
            next(server)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_inflight_spawn_cleanup_failure_keeps_generation_tombstone(
    tmp_path, monkeypatch
):
    _repo, source = _make_repo(tmp_path, monkeypatch)
    clients = []

    class FailingSpawnCleanupClient:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.state = "stopped"
            self.start_entered = asyncio.Event()
            self.never_finish_start = asyncio.Event()
            clients.append(self)

        @property
        def is_running(self):
            return False

        async def start(self):
            self.state = "starting"
            self.start_entered.set()
            await self.never_finish_start.wait()

        async def shutdown(self):
            self.state = "error"
            raise RuntimeError("spawn cleanup failed")

    server = _install_mock_server(monkeypatch, "clean", "pyright")
    next(server)
    monkeypatch.setattr(
        "agent.lsp.manager.LSPClient", FailingSpawnCleanupClient
    )
    svc = _threadless_service()
    acquire = asyncio.create_task(svc._acquire_client(str(source)))
    try:
        while not clients:
            await asyncio.sleep(0)
        await clients[0].start_entered.wait()

        assert await svc._shutdown_async() is False
        await asyncio.gather(acquire, return_exceptions=True)

        entry = next(iter(svc._clients.values()))
        assert svc._shutdown_state == "failed"
        assert svc._admitting is False
        assert svc._spawning == {}
        assert entry.generation == 1
        assert entry.retiring is True
        assert entry.retirement_error == (
            "RuntimeError: spawn cleanup failed"
        )
    finally:
        if clients:
            clients[0].never_finish_start.set()
        await asyncio.gather(acquire, return_exceptions=True)
        try:
            next(server)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_reader_cleanup_failure_retains_handle_and_blocks_replacement(
    tmp_path, monkeypatch
):
    _repo, source = _make_repo(tmp_path, monkeypatch)
    allow_eof = asyncio.Event()
    clients = []

    class FakeStdin:
        def is_closing(self):
            return False

    class FailingTerminateProcess:
        def __init__(self):
            self.returncode = None
            self.stdin = FakeStdin()
            self.stdout = object()
            self.stderr = None
            self.fail_terminate = True
            self.terminate_calls = 0

        def terminate(self):
            self.terminate_calls += 1
            if self.fail_terminate:
                raise PermissionError("terminate denied")

        def kill(self):
            raise AssertionError("kill should not run after a successful terminate")

        async def wait(self):
            self.returncode = 0
            return 0

    class ReaderCleanupClient(LSPClient):
        async def start(self):
            self._state = "running"
            self.process = FailingTerminateProcess()
            self._proc = self.process  # type: ignore[assignment]
            self._start_reader_task()
            clients.append(self)

    async def controlled_read(_stream):
        await allow_eof.wait()
        return None

    server = _install_mock_server(monkeypatch, "clean", "pyright")
    spawn_count = next(server)
    monkeypatch.setattr("agent.lsp.manager.LSPClient", ReaderCleanupClient)
    monkeypatch.setattr("agent.lsp.client.read_message", controlled_read)
    svc = _threadless_service()
    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()
    loop_errors = []
    loop.set_exception_handler(
        lambda _loop, context: loop_errors.append(context)
    )
    try:
        lease = await svc._acquire_client(str(source))
        assert lease is not None
        client = lease.client
        entry = next(iter(svc._clients.values()))
        assert client._reader_task is not None
        lease.release()

        allow_eof.set()
        deadline = loop.time() + 1.0
        while client._cleanup_error is None and loop.time() < deadline:
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)

        assert client._proc is client.process
        assert client._cleanup_error == "PermissionError: terminate denied"
        assert loop_errors == []

        assert await svc._acquire_client(str(source)) is None
        assert spawn_count["value"] == 1
        assert len(clients) == 1
        assert next(iter(svc._clients.values())) is entry
        assert entry.retiring is True
        assert entry.retirement_error == "PermissionError: terminate denied"

        client.process.fail_terminate = False
        assert await svc._shutdown_async() is True
        assert client._proc is None
        assert client.process.returncode == 0
        assert svc._clients == {}
    finally:
        try:
            allow_eof.set()
            for client in clients:
                client.process.fail_terminate = False
            assert await svc._shutdown_async() is True
            gc.collect()
            await asyncio.sleep(0)
            try:
                next(server)
            except StopIteration:
                pass
        finally:
            loop.set_exception_handler(previous_exception_handler)
    assert loop_errors == []


def test_manager_reap_info_respawn_active_and_no_first_reuse_debug(
    mock_pyright, caplog
):
    eventlog.reset_announce_caches()
    caplog.set_level(logging.DEBUG, logger="hermes.lint.lsp")
    source = mock_pyright / "x.py"
    source.write_text("print('hi')\n", encoding="utf-8")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=60.0,
    )
    try:
        svc.get_diagnostics_sync(str(source))
        key = next(iter(svc._clients))
        svc._last_used[key] = 0
        svc._loop.run(svc._reap_idle_once(), timeout=5.0)
        svc.get_diagnostics_sync(str(source))

        messages = [record.getMessage() for record in caplog.records]
        assert len([message for message in messages if "active for" in message]) == 2
        assert len([message for message in messages if "reaped 1 idle client" in message]) == 1
        assert not any("reused client" in message for message in messages)
        assert all(
            record.levelno == logging.INFO
            for record in caplog.records
            if "active for" in record.getMessage()
            or "reaped 1 idle client" in record.getMessage()
        )
    finally:
        assert svc.shutdown() is True
        eventlog.reset_announce_caches()


@pytest.mark.asyncio
async def test_multi_key_reap_announces_each_key_before_replacement(
    tmp_path,
    monkeypatch,
    caplog,
):
    repos = []
    sources = []
    for name in ("repo-a", "repo-b"):
        repo = tmp_path / name
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "pyproject.toml").write_text("", encoding="utf-8")
        source = repo / "x.py"
        source.write_text("print('hi')\n", encoding="utf-8")
        repos.append(repo)
        sources.append(source)
    monkeypatch.chdir(repos[0])

    clients = []

    class ControlledClient:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.state = "stopped"
            self.shutdown_started = asyncio.Event()
            self.allow_shutdown = asyncio.Event()
            clients.append(self)

        @property
        def is_running(self):
            return self.state == "running"

        async def start(self):
            self.state = "running"

        async def shutdown(self):
            self.shutdown_started.set()
            await self.allow_shutdown.wait()
            self.state = "stopped"

    eventlog.reset_announce_caches()
    caplog.set_level(logging.DEBUG, logger="hermes.lint.lsp")
    server = _install_mock_server(monkeypatch, "clean", "pyright")
    spawn_count = next(server)
    monkeypatch.setattr("agent.lsp.manager.LSPClient", ControlledClient)
    svc = _threadless_service()
    svc._idle_timeout = 60.0
    reap = None
    try:
        for source in sources:
            lease = await svc._acquire_client(str(source))
            assert lease is not None
            lease.release()

        key_a = ("pyright", str(repos[0]))
        key_b = ("pyright", str(repos[1]))
        assert set(svc._clients) == {key_a, key_b}
        initial = {client.workspace_root: client for client in clients}
        svc._last_used[key_a] = 0
        svc._last_used[key_b] = 0

        reap = asyncio.create_task(svc._reap_idle_once())
        await initial[str(repos[0])].shutdown_started.wait()
        await initial[str(repos[1])].shutdown_started.wait()

        initial[str(repos[0])].allow_shutdown.set()
        deadline = asyncio.get_running_loop().time() + 1.0
        while key_a in svc._clients and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0)
        assert key_a not in svc._clients
        assert key_b in svc._clients
        assert not reap.done()

        replacement = await svc._acquire_client(str(sources[0]))
        assert replacement is not None
        replacement.release()
        assert spawn_count["value"] == 3

        records = list(caplog.records)
        active_a = [
            (index, record)
            for index, record in enumerate(records)
            if f"active for {repos[0]}" in record.getMessage()
        ]
        reaped_a = [
            (index, record)
            for index, record in enumerate(records)
            if "reaped 1 idle client" in record.getMessage()
            and str(repos[0]) in record.getMessage()
        ]
        assert len(active_a) == 2
        assert all(record.levelno == logging.INFO for _, record in active_a)
        assert len(reaped_a) == 1
        assert reaped_a[0][1].levelno == logging.INFO
        assert reaped_a[0][0] < active_a[1][0]
        assert not any(
            f"reused client for {repos[0]}" in record.getMessage()
            for record in records
        )

        initial[str(repos[1])].allow_shutdown.set()
        await reap
        assert any(
            "reaped 1 idle client" in record.getMessage()
            and str(repos[1]) in record.getMessage()
            and record.levelno == logging.INFO
            for record in caplog.records
        )
    finally:
        for client in clients:
            client.allow_shutdown.set()
        if reap is not None:
            await asyncio.gather(reap, return_exceptions=True)
        assert await svc._shutdown_async() is True
        eventlog.reset_announce_caches()
        try:
            next(server)
        except StopIteration:
            pass


@pytest.mark.asyncio
async def test_service_shutdown_drains_reader_request_dispatch(monkeypatch, tmp_path):
    svc = _threadless_service()
    client = LSPClient(
        server_id="pyright",
        workspace_root=str(tmp_path),
        command=["unused"],
        cwd=str(tmp_path),
    )

    class FakeStdin:
        def is_closing(self):
            return False

    class ExitedProcess:
        returncode = 0
        stdin = FakeStdin()
        stdout = object()
        stderr = None

    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()
    keep_reader_open = asyncio.Event()
    reads = 0

    async def controlled_handler(_params):
        handler_started.set()
        try:
            await asyncio.Future()
        finally:
            handler_cancelled.set()

    async def controlled_read(_stream):
        nonlocal reads
        reads += 1
        if reads == 1:
            return {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "workspace/configuration",
                "params": {"items": []},
            }
        await keep_reader_open.wait()
        return None

    client._proc = ExitedProcess()  # type: ignore[assignment]
    client._state = "running"
    client._request_handlers["workspace/configuration"] = controlled_handler
    monkeypatch.setattr("agent.lsp.client.read_message", controlled_read)
    client._reader_task = asyncio.create_task(client._reader_loop())
    key = (client.server_id, client.workspace_root)
    svc._clients[key] = _ClientEntry(client=client, generation=1)

    await handler_started.wait()
    dispatch_tasks = set(client._dispatch_tasks)
    assert len(dispatch_tasks) == 1

    assert await svc._shutdown_async() is True

    assert handler_cancelled.is_set()
    assert all(task.done() for task in dispatch_tasks)
    assert client._dispatch_tasks == set()
    assert client._proc is None
    assert svc._clients == {}


def test_blocking_build_spawn_retains_shutdown_owner_and_honors_fence(
    tmp_path, monkeypatch
):
    repos = []
    sources = []
    for name in ("repo-a", "repo-b"):
        repo = tmp_path / name
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "pyproject.toml").write_text("", encoding="utf-8")
        source = repo / "x.py"
        source.write_text("print('hi')\n", encoding="utf-8")
        repos.append(repo)
        sources.append(source)
    monkeypatch.chdir(repos[0])

    build_blocked = threading.Event()
    allow_build = threading.Event()
    clients = []
    start_records = []

    def controlled_build(root: str, _ctx: ServerContext) -> SpawnSpec:
        if root == str(repos[1]):
            build_blocked.set()
            if not allow_build.wait(timeout=5.0):
                raise RuntimeError("test did not release blocking build_spawn")
        return SpawnSpec(
            command=["controlled-lsp"],
            workspace_root=root,
            cwd=root,
            env={},
            initialization_options={},
        )

    class BridgeClient:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.state = "stopped"
            self.shutdown_calls = 0
            clients.append(self)

        @property
        def is_running(self):
            return self.state == "running"

        async def start(self):
            start_records.append((self.workspace_root, svc.is_active()))
            self.state = "running"

        async def shutdown(self):
            self.shutdown_calls += 1
            self.state = "stopped"

    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == "pyright")
    original_server = SERVERS[target_index]
    SERVERS[target_index] = ServerDef(
        server_id="pyright",
        extensions=original_server.extensions,
        resolve_root=lambda _fp, ws: ws,
        build_spawn=controlled_build,
        seed_first_push=False,
        description="blocking build_spawn test",
    )
    monkeypatch.setattr("agent.lsp.manager.LSPClient", BridgeClient)
    monkeypatch.setattr("agent.lsp.manager.SHUTDOWN_WAIT_TIMEOUT", 0.05)
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=0,
    )

    async def acquire_and_release(path: Path):
        lease = await svc._acquire_client(str(path))
        if lease is None:
            return None
        client = lease.client
        lease.release()
        return client

    blocked_results = []
    blocked_errors = []

    def run_blocked_acquire():
        try:
            blocked_results.append(
                svc._loop.run(acquire_and_release(sources[1]), timeout=5.0)
            )
        except BaseException as exc:  # noqa: BLE001
            blocked_errors.append(exc)

    blocked_thread = threading.Thread(target=run_blocked_acquire)
    try:
        first_client = svc._loop.run(acquire_and_release(sources[0]), timeout=3.0)
        assert first_client is clients[0]

        blocked_thread.start()
        assert build_blocked.wait(timeout=2.0)

        assert svc.shutdown() is False
        retained = svc._shutdown_future
        assert retained is not None
        assert not retained.cancelled()
        assert not retained.done()
        assert svc._admitting is False

        allow_build.set()
        blocked_thread.join(timeout=3.0)
        assert not blocked_thread.is_alive()
        assert retained.result(timeout=3.0) is True

        assert blocked_errors == []
        assert blocked_results == [None]
        assert len(clients) == 1
        assert start_records == [(str(repos[0]), True)]
        assert first_client.shutdown_calls == 1
        assert svc.get_status()["clients"] == []
        assert svc.shutdown() is True
    finally:
        allow_build.set()
        if blocked_thread.ident is not None:
            blocked_thread.join(timeout=3.0)
        retained = svc._shutdown_future
        if retained is not None and not retained.done():
            retained.result(timeout=3.0)
        if not svc._loop_stopped:
            assert svc.shutdown() is True
        SERVERS[target_index] = original_server
