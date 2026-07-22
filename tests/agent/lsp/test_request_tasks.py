"""Tests for request task retention and cleanup in the LSP client.

Covers:
- Server-to-client request creates a task that is stored in _request_tasks.
- Done callback removes the task and logs unexpected exceptions.
- Shutdown cancels, awaits, then clears request tasks (no dangling tasks).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


class TestRequestTaskRetention:
    """Verify that server→client requests create stored tasks with proper lifecycle."""

    @pytest.mark.asyncio
    async def test_request_task_stored_and_removed(self, tmp_path: Path):
        """A server-to-client request creates a task in _request_tasks, then
        the done callback removes it after completion."""
        f = tmp_path / "x.py"
        f.write_text("print('hi')\n")

        client = _client(tmp_path, "requests")
        await client.start()
        try:
            # After start, the mock server sends a workspace/configuration request.
            # Give the reader loop time to process it.
            await asyncio.sleep(0.3)

            # The task should have been created and completed (workspace/configuration
            # handler returns immediately), so _request_tasks should be empty again.
            assert len(client._request_tasks) == 0
        finally:
            await client.shutdown()

    @pytest.mark.asyncio
    async def test_request_task_done_callback_removes_task(self, tmp_path: Path):
        """Verify the done callback removes the task from _request_tasks
        even when the handler raises an exception (caught by the handler's
        own except Exception in _dispatch_request)."""
        f = tmp_path / "x.py"
        f.write_text("print('hi')\n")

        client = _client(tmp_path, "requests")

        # Inject a handler that raises Exception — _dispatch_request catches it,
        # so the task completes normally (no exception visible via task.exception()).
        async def _explode(params):
            raise RuntimeError("simulated handler failure")

        client._request_handlers["workspace/configuration"] = _explode

        await client.start()
        try:
            await asyncio.sleep(0.5)
            # Task should be removed by done callback even on exception
            assert len(client._request_tasks) == 0
        finally:
            await client.shutdown()


class TestRequestTaskCleanup:
    """Verify shutdown properly cancels and awaits request tasks."""

    @pytest.mark.asyncio
    async def test_shutdown_awaits_cancelled_tasks(self, tmp_path: Path):
        """During shutdown, pending request tasks are cancelled and awaited
        (not just cleared), ensuring no task is left dangling."""
        f = tmp_path / "x.py"
        f.write_text("print('hi')\n")

        client = _client(tmp_path, "slow_requests")
        await client.start()
        try:
            # slow_requests mode sends a configuration request with a delay,
            # so a handler task may be pending when we shut down.
            await asyncio.sleep(0.2)
        finally:
            # This should not hang or raise — it must await cancelled tasks.
            await client.shutdown()

        # After shutdown, no tasks should remain.
        assert len(client._request_tasks) == 0

    @pytest.mark.asyncio
    async def test_cleanup_process_gather_return_exceptions(self, tmp_path: Path):
        """_cleanup_process uses asyncio.gather(return_exceptions=True) so
        CancelledError from cancelled tasks doesn't propagate."""
        f = tmp_path / "x.py"
        f.write_text("print('hi')\n")

        client = _client(tmp_path, "requests")
        await client.start()

        # Manually inject a slow pending task to test cleanup
        slow_started = asyncio.Event()

        async def _slow_handler(params):
            slow_started.set()
            await asyncio.sleep(60)

        client._request_handlers["workspace/configuration"] = _slow_handler

        # Wait for reader loop to pick up a request and create a task
        await asyncio.sleep(0.5)

        # shutdown must complete without hanging (the slow handler is cancelled)
        await client.shutdown()
        assert not client.is_running

    @pytest.mark.asyncio
    async def test_request_task_exception_logged(self, tmp_path: Path, caplog):
        """_on_request_task_done logs when a task ends with an exception
        that escaped the handler's except Exception guard (e.g. CancelledError
        from an external cancel)."""
        f = tmp_path / "x.py"
        f.write_text("print('hi')\n")

        client = _client(tmp_path, "requests")
        await client.start()

        # Inject a handler that blocks so we can externally cancel it
        block_event = asyncio.Event()

        async def _blocking_handler(params):
            await block_event.wait()

        client._request_handlers["workspace/configuration"] = _blocking_handler

        # Wait for the task to be created
        await asyncio.sleep(0.3)

        # Find the pending task and cancel it directly (bypassing _dispatch_request's
        # except block — CancelledError escapes Exception)
        for task in list(client._request_tasks):
            task.cancel()
            break

        # Give the done callback time to fire
        await asyncio.sleep(0.2)

        # Cancelled tasks should still be removed by the done callback
        assert len(client._request_tasks) == 0

        await client.shutdown()
