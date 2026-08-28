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


class _FakeStdin:
    def is_closing(self) -> bool:
        return False


class _ScriptedProcess:
    """OS-neutral subprocess fake with explicit terminate/wait/kill outcomes."""

    def __init__(
        self,
        *,
        terminate_error: Exception | None = None,
        wait_steps: list[object] | None = None,
        kill_error: Exception | None = None,
    ) -> None:
        self.returncode = None
        self.stdin = _FakeStdin()
        self.stdout = object()
        self.stderr = None
        self.terminate_error = terminate_error
        self.wait_steps = list(wait_steps or [])
        self.kill_error = kill_error
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_error is not None:
            raise self.terminate_error

    def kill(self) -> None:
        self.kill_calls += 1
        if self.kill_error is not None:
            raise self.kill_error

    async def wait(self) -> int:
        self.wait_calls += 1
        step = self.wait_steps.pop(0) if self.wait_steps else "exit"
        if step == "block":
            await asyncio.Future()
        if isinstance(step, Exception):
            raise step
        self.returncode = 0
        return self.returncode


@pytest.mark.asyncio
async def test_client_lifecycle_clean(tmp_path: Path):
    """Full lifecycle: spawn, initialize, open, get clean diagnostics, shutdown."""
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n", encoding="utf-8")

    client = _client(tmp_path, "clean")
    await client.start()
    proc = client._proc
    assert proc is not None
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
    assert proc.returncode is not None


@pytest.mark.asyncio
async def test_client_receives_published_errors(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n", encoding="utf-8")

    client = _client(tmp_path, "errors")
    await client.start()
    proc = client._proc
    assert proc is not None
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
    assert proc.returncode is not None


@pytest.mark.asyncio
async def test_reader_exit_at_end_of_initialization_retires_client(tmp_path: Path):
    client = _client(tmp_path, "crash")

    try:
        await client.start()
    except LSPProtocolError:
        pass
    else:
        reader_task = client._reader_task
        if reader_task is not None:
            await asyncio.wait_for(asyncio.shield(reader_task), timeout=3.0)

    assert client.state == "error"
    assert not client.is_running
    assert client._proc is None
    await client.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("script", ["clean_eof", "malformed_frame"])
async def test_reader_failure_retires_client_and_rejects_later_work(
    tmp_path: Path, script: str
):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n", encoding="utf-8")

    client = _client(tmp_path, script)
    await client.start()
    proc = client._proc
    reader_task = client._reader_task
    assert proc is not None
    assert reader_task is not None
    try:
        version = await client.open_file(str(f), language_id="python")
        await asyncio.wait_for(asyncio.shield(reader_task), timeout=3.0)

        assert not client.is_running
        await asyncio.wait_for(proc.wait(), timeout=3.0)
        with pytest.raises(LSPProtocolError):
            await asyncio.wait_for(
                client.wait_for_diagnostics(str(f), version, timeout=3.0),
                timeout=0.5,
            )
        with pytest.raises(LSPProtocolError):
            await asyncio.wait_for(
                client.open_file(str(f), language_id="python"),
                timeout=0.5,
            )
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_cancelled_shutdown_waiter_cannot_abandon_process_cleanup(
    tmp_path: Path,
):
    client = _client(tmp_path, "clean")
    await client.start()
    proc = client._proc
    assert proc is not None

    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    original_cleanup = client._cleanup_process

    async def controlled_cleanup():
        cleanup_started.set()
        await allow_cleanup.wait()
        await original_cleanup()

    client._cleanup_process = controlled_cleanup  # type: ignore[method-assign]
    first_waiter = asyncio.create_task(client.shutdown())
    try:
        await cleanup_started.wait()
        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter

        assert client._shutdown_task is not None
        assert not client._shutdown_task.done()

        allow_cleanup.set()
        await client.shutdown()

        assert client._proc is None
        assert proc.returncode is not None
        assert client.state == "stopped"
    finally:
        allow_cleanup.set()
        await client.shutdown()


@pytest.mark.asyncio
async def test_cancelled_diagnostics_wait_drains_pull_and_push_children(
    tmp_path: Path,
    monkeypatch,
):
    client = _client(tmp_path)
    started = asyncio.Event()
    active: set[asyncio.Task] = set()
    children: list[asyncio.Task] = []

    async def block_child(*_args) -> None:
        task = asyncio.current_task()
        assert task is not None
        children.append(task)
        active.add(task)
        if len(active) == 2:
            started.set()
        try:
            await asyncio.Future()
        finally:
            active.discard(task)

    monkeypatch.setattr(client, "_connection_is_open", lambda: True)
    monkeypatch.setattr(client, "_pull_document_diagnostics", block_child)
    monkeypatch.setattr(client, "_wait_for_fresh_push", block_child)

    waiter = asyncio.create_task(
        client.wait_for_diagnostics(str(tmp_path / "x.py"), 1, timeout=30.0)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert len(children) == 2
        assert len(active) == 0
        assert all(task.done() and task.cancelled() for task in children)
    finally:
        waiter.cancel()
        for task in children:
            task.cancel()
        await asyncio.gather(waiter, *children, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminate_error", "wait_steps", "kill_error", "expected_error"),
    [
        (PermissionError("terminate denied"), [], None, PermissionError),
        (OSError("terminate failed"), [], None, OSError),
        (None, [OSError("wait failed")], None, OSError),
        (None, ["block"], PermissionError("kill denied"), PermissionError),
        (None, ["block", OSError("post-kill wait failed")], None, OSError),
    ],
)
async def test_cleanup_retains_process_until_exit_is_confirmed(
    tmp_path: Path,
    monkeypatch,
    terminate_error,
    wait_steps,
    kill_error,
    expected_error,
):
    monkeypatch.setattr("agent.lsp.client.SHUTDOWN_GRACE", 0.01)
    client = _client(tmp_path)
    proc = _ScriptedProcess(
        terminate_error=terminate_error,
        wait_steps=wait_steps,
        kill_error=kill_error,
    )
    client._proc = proc  # type: ignore[assignment]
    client._state = "error"

    with pytest.raises(expected_error):
        await client._cleanup_process()

    assert client._proc is proc
    assert client._cleanup_error is not None

    # Once exit is independently confirmed, the same handle can be cleared.
    proc.returncode = 1
    await client._cleanup_process()
    assert client._proc is None
    assert client._cleanup_error is None


@pytest.mark.asyncio
async def test_failed_client_shutdown_is_retried_once_for_concurrent_callers(
    tmp_path: Path,
):
    client = _client(tmp_path)
    allow_wait = asyncio.Event()
    wait_started = asyncio.Event()

    class TransientProcess(_ScriptedProcess):
        def __init__(self):
            super().__init__()
            self.fail_terminate = True

        def terminate(self) -> None:
            self.terminate_calls += 1
            if self.fail_terminate:
                self.fail_terminate = False
                raise PermissionError("transient terminate failure")

        async def wait(self) -> int:
            self.wait_calls += 1
            wait_started.set()
            await allow_wait.wait()
            self.returncode = 0
            return 0

    proc = TransientProcess()
    client._proc = proc  # type: ignore[assignment]
    client._state = "error"

    with pytest.raises(PermissionError):
        await client.shutdown()
    failed_owner = client._shutdown_task
    assert failed_owner is not None and failed_owner.done()
    assert client._proc is proc

    first = asyncio.create_task(client.shutdown())
    second = asyncio.create_task(client.shutdown())
    await wait_started.wait()

    retry_owner = client._shutdown_task
    assert retry_owner is not None and retry_owner is not failed_owner
    assert not retry_owner.done()
    assert proc.terminate_calls == 2

    allow_wait.set()
    await asyncio.gather(first, second)
    assert proc.terminate_calls == 2
    assert proc.wait_calls == 1
    assert client._proc is None


@pytest.mark.asyncio
async def test_shutdown_cancels_and_drains_reader_request_dispatch(monkeypatch, tmp_path):
    client = _client(tmp_path)
    proc = _ScriptedProcess()
    proc.returncode = 0
    client._proc = proc  # type: ignore[assignment]
    client._state = "running"

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

    async def fake_read_message(_stream):
        nonlocal reads
        reads += 1
        if reads == 1:
            return {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "workspace/configuration",
                "params": {"items": []},
            }
        await keep_reader_open.wait()
        return None

    client._request_handlers["workspace/configuration"] = controlled_handler
    monkeypatch.setattr("agent.lsp.client.read_message", fake_read_message)
    client._reader_task = asyncio.create_task(client._reader_loop())

    await handler_started.wait()
    dispatch_tasks = set(client._dispatch_tasks)
    assert len(dispatch_tasks) == 1

    await client.shutdown()

    assert handler_cancelled.is_set()
    assert all(task.done() for task in dispatch_tasks)
    assert client._dispatch_tasks == set()
    assert client._reader_task is None
    assert client._proc is None
