"""End-to-end client tests against the in-process mock LSP server.

Spins up :file:`_mock_lsp_server.py` as an actual subprocess, drives
it through real LSP traffic, and asserts diagnostic flow.  This is
the closest thing we have to integration coverage without requiring
pyright/gopls/etc. to be installed in CI.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient
from agent.lsp.protocol import LSPProtocolError


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _client(workspace: Path, script: str = "clean") -> LSPClient:
    env = {"MOCK_LSP_SCRIPT": script, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    return LSPClient(
        server_id=f"mock-{script}",
        workspace_root=str(workspace),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(workspace),
    )


@pytest.mark.asyncio
async def test_client_lifecycle_clean(tmp_path: Path):
    """Full lifecycle: spawn, initialize, open, get clean diagnostics, shutdown."""
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "clean")
    await client.start()
    try:
        assert client.is_running
        version = await client.open_file(str(f), language_id="python")
        assert version == 0
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert diags == []
    finally:
        await client.shutdown()
    assert not client.is_running


@pytest.mark.asyncio
async def test_client_receives_published_errors(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        version = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert len(diags) == 1
        d = diags[0]
        assert d["severity"] == 1
        assert d["code"] == "MOCK001"
        assert d["source"] == "mock-lsp"
        assert "synthetic error" in d["message"]
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_client_didchange_bumps_version(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        v0 = await client.open_file(str(f), language_id="python")
        f.write_text("print('hi 2')\n")
        v1 = await client.open_file(str(f), language_id="python")  # re-open path = didChange
        assert v1 == v0 + 1
        await client.wait_for_diagnostics(str(f), v1, mode="document")
        # Mock pushed a diagnostic for both events; merged view has one
        # entry (push store keyed by file path).
        diags = client.diagnostics_for(str(f))
        assert len(diags) == 1
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_client_handles_crashing_server(tmp_path: Path):
    """When the server exits right after initialize, subsequent requests
    fail gracefully (not hang)."""
    f = tmp_path / "x.py"
    f.write_text("")

    client = _client(tmp_path, "crash")
    await client.start()  # should succeed (mock answers initialize before crashing)
    # Give the OS a moment to deliver the EOF.
    await asyncio.sleep(0.2)
    # The reader loop should detect EOF and mark pending requests as failed.
    try:
        await asyncio.wait_for(
            client.open_file(str(f), language_id="python"), timeout=2.0
        )
    except Exception:
        pass  # any exception is acceptable; the contract is "doesn't hang"
    await client.shutdown()


@pytest.mark.asyncio
async def test_client_shutdown_idempotent(tmp_path: Path):
    """Calling shutdown twice must be safe."""
    f = tmp_path / "x.py"
    f.write_text("")
    client = _client(tmp_path, "clean")
    await client.start()
    await client.shutdown()
    await client.shutdown()  # must not raise


@pytest.mark.asyncio
async def test_concurrent_shutdown_callers_wait_for_same_cleanup(tmp_path: Path, monkeypatch):
    client = _client(tmp_path, "clean")
    await client.start()
    cleanup_entered = asyncio.Event()
    allow_cleanup = asyncio.Event()
    original_cleanup = client._cleanup_process

    async def blocked_cleanup(descendants=None):
        cleanup_entered.set()
        await allow_cleanup.wait()
        await original_cleanup(descendants)

    monkeypatch.setattr(client, "_cleanup_process", blocked_cleanup)
    first = asyncio.create_task(client.shutdown())
    second = None
    try:
        await cleanup_entered.wait()
        second = asyncio.create_task(client.shutdown())
        await asyncio.sleep(0)
        assert not second.done()
        allow_cleanup.set()
        await asyncio.gather(first, second)
    finally:
        allow_cleanup.set()
        pending = [first]
        if second is not None:
            pending.append(second)
        await asyncio.gather(*pending, return_exceptions=True)
        await client.shutdown()


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_cancelling_shutdown_caller_does_not_cancel_shared_cleanup(
    tmp_path: Path, monkeypatch
):
    import psutil

    pid_file = tmp_path / "cancel-child.pid"
    client = LSPClient(
        server_id="mock-cancel-child",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
        env={
            "MOCK_LSP_SCRIPT": "child",
            "MOCK_LSP_CHILD_PID_FILE": str(pid_file),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
        cwd=str(tmp_path),
    )
    child_pid = None
    allow_cleanup = asyncio.Event()
    first = None
    try:
        await client.start()
        for _ in range(100):
            if pid_file.exists():
                break
            await asyncio.sleep(0.01)
        child_pid = int(pid_file.read_text())
        cleanup_entered = asyncio.Event()
        original_cleanup = client._cleanup_process

        async def blocked_cleanup(descendants=None):
            cleanup_entered.set()
            await allow_cleanup.wait()
            await original_cleanup(descendants)

        monkeypatch.setattr(client, "_cleanup_process", blocked_cleanup)
        first = asyncio.create_task(client.shutdown())
        await cleanup_entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(client.shutdown())
        await asyncio.sleep(0)
        assert not second.done()
        allow_cleanup.set()
        await second
        assert client.process_alive is False
        assert (
            not psutil.pid_exists(child_pid)
            or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        allow_cleanup.set()
        if first is not None:
            await asyncio.gather(first, return_exceptions=True)
        await asyncio.shield(client.shutdown())
        if child_pid is not None and psutil.pid_exists(child_pid):
            try:
                child = psutil.Process(child_pid)
                child.kill()
                child.wait(timeout=2.0)
            except psutil.Error:
                pass


@pytest.mark.asyncio
async def test_client_rejects_restart_after_shutdown(tmp_path: Path):
    client = _client(tmp_path, "clean")
    await client.start()
    await client.shutdown()
    with pytest.raises(LSPProtocolError, match="single-use"):
        await client.start()


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_shutdown_removes_wrapper_descendant_tree(tmp_path: Path):
    import psutil

    pid_file = tmp_path / "child.pid"
    env = {
        "MOCK_LSP_SCRIPT": "child",
        "MOCK_LSP_CHILD_PID_FILE": str(pid_file),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    client = LSPClient(
        server_id="mock-child",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(tmp_path),
    )
    child_pid = None
    try:
        await client.start()
        for _ in range(100):
            if pid_file.exists():
                break
            await asyncio.sleep(0.01)
        child_pid = int(pid_file.read_text())
        assert psutil.pid_exists(child_pid)

        await client.shutdown()

        for _ in range(100):
            if not psutil.pid_exists(child_pid):
                break
            try:
                if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            await asyncio.sleep(0.01)
        assert (
            not psutil.pid_exists(child_pid)
            or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        )
    finally:
        await asyncio.shield(client.shutdown())
        if child_pid is not None and psutil.pid_exists(child_pid):
            try:
                child = psutil.Process(child_pid)
                child.kill()
                child.wait(timeout=2.0)
            except psutil.Error:
                pass


@pytest.mark.asyncio
async def test_client_diagnostics_are_deduped(tmp_path: Path):
    """Repeated identical pushes must not produce duplicate diagnostics."""
    f = tmp_path / "x.py"
    f.write_text("")
    client = _client(tmp_path, "errors")
    await client.start()
    try:
        for _ in range(3):
            v = await client.open_file(str(f), language_id="python")
            await client.wait_for_diagnostics(str(f), v, mode="document")
        diags = client.diagnostics_for(str(f))
        # Push store overwrites on every notification — should have 1.
        assert len(diags) == 1
    finally:
        await client.shutdown()
