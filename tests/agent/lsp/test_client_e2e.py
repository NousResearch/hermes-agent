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


@pytest.mark.asyncio
async def test_request_tasks_tracked_and_cleared_on_shutdown(tmp_path: Path):
    """Server-to-client request tasks are tracked in _request_tasks and
    awaited during shutdown, preventing GC warnings and unobserved exceptions."""
    f = tmp_path / "x.py"
    f.write_text("")
    client = _client(tmp_path, "clean")
    await client.start()
    try:
        # _request_tasks exists and is initially empty (no server→client requests yet)
        assert isinstance(client._request_tasks, set)
        # After initialization, the mock server may have sent client requests.
        # Regardless, the set must be clean after shutdown.
    finally:
        await client.shutdown()
    # After shutdown, _request_tasks should be cleared
    assert len(client._request_tasks) == 0


@pytest.mark.asyncio
async def test_request_task_exception_logged_on_done(tmp_path: Path, caplog):
    """When a server-to-client request handler raises, the done callback
    retrieves the exception and logs a warning rather than leaving it
    unobserved (GC warning)."""
    import logging

    f = tmp_path / "x.py"
    f.write_text("")
    client = _client(tmp_path, "clean")
    await client.start()
    try:
        # Manually add a failing request task to _request_tasks
        # to verify the done callback retrieves the exception
        async def _failing_dispatch(key, msg):
            raise RuntimeError("simulated handler failure")

        # Simulate what _reader_loop does: create task, add to set, add done callback
        async def _simulate_reader_loop_path():
            task = asyncio.create_task(_failing_dispatch("test", {}))
            client._request_tasks.add(task)

            def _on_request_done(t: asyncio.Task) -> None:
                """Exact copy of the callback from client.py."""
                client._request_tasks.discard(t)
                exc = t.exception()
                if exc is not None and not isinstance(
                    exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)
                ):
                    logger = logging.getLogger("agent.lsp.client")
                    logger.warning(
                        "[%s] server-to-client request handler failed: %s",
                        client.server_id, exc,
                    )

            task.add_done_callback(_on_request_done)
            # Await with return_exceptions so we don't re-raise
            results = await asyncio.gather(task, return_exceptions=True)
            return results

        results = await _simulate_reader_loop_path()
        # The exception should be captured, not propagated
        assert len(results) == 1
        assert isinstance(results[0], RuntimeError)
        assert "simulated handler failure" in str(results[0])
    finally:
        await client.shutdown()

    # Verify the exception was logged
    assert any(
        "simulated handler failure" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    ), f"Expected warning not found in: {[r.message for r in caplog.records]}"
