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
    f.write_text("print('hi')\n", encoding="utf-8")

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
    f.write_text("print('hi')\n", encoding="utf-8")

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
async def test_aborted_start_reports_exit_status_and_stderr_tail(tmp_path: Path):
    """A server that dies mid-initialize must not fail as an opaque protocol error.

    A Node language server that exhausts its heap aborts (SIGABRT) before answering
    ``initialize``; the failure the caller logs should carry the exit status and the
    stderr trace instead of a bare JSON-RPC error text.
    """
    client = _client(tmp_path, "oom_abort")

    with pytest.raises(LSPProtocolError) as excinfo:
        await client.start()

    assert client.state == "error"
    assert client._proc is None
    message = str(excinfo.value)
    # negative returncode rendered as a signal, not a bare code
    assert "signal" in message
    # stderr tail reached the failure report
    assert "JavaScript heap out of memory" in message
    await client.shutdown()


@pytest.mark.asyncio
async def test_failure_details_empty_for_live_server(tmp_path: Path):
    client = _client(tmp_path, "clean")
    await client.start()
    try:
        assert client.failure_details() == ""
    finally:
        await client.shutdown()


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
async def test_cancelled_start_terminates_spawned_server(tmp_path: Path):
    """An outer startup budget may cancel initialize before the manager registers the client."""
    client = _client(tmp_path, "slow")
    start = asyncio.create_task(client.start())
    while client._proc is None:
        await asyncio.sleep(0)
    proc = client._proc

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(start, timeout=0.05)

    assert proc is not None
    await asyncio.wait_for(proc.wait(), timeout=3.0)
    assert client.state == "error"
    assert client._proc is None


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_cancelled_start_hard_kills_sigterm_ignoring_descendant(tmp_path: Path):
    """A launcher exiting on SIGTERM must not let an ignoring server child escape cleanup."""
    import psutil

    child_pid_file = tmp_path / "child.pid"
    client = _client(tmp_path, "slow_tree")
    assert client._env is not None
    client._env["MOCK_LSP_CHILD_PID"] = str(child_pid_file)
    start = asyncio.create_task(client.start())
    child = None
    try:
        ready_deadline = asyncio.get_running_loop().time() + 3.0
        while not child_pid_file.exists():
            assert asyncio.get_running_loop().time() < ready_deadline
            await asyncio.sleep(0.01)
        child = psutil.Process(int(child_pid_file.read_text(encoding="utf-8")))

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(start, timeout=0.05)

        deadline = asyncio.get_running_loop().time() + 3.0
        while child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        assert client.state == "error"
        assert client._proc is None
    finally:
        if not start.done():
            start.cancel()
            await asyncio.gather(start, return_exceptions=True)
        if child is not None and child.is_running():
            child.kill()


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
async def test_shutdown_never_signals_a_server_that_honours_exit(tmp_path: Path):
    """A server that exits on the protocol ``exit`` must not be SIGTERMed on top of it (#72944:
    on Darwin the reaped PID can already belong to another process)."""
    client = _client(tmp_path, "clean")
    await client.start()
    proc = client._proc
    assert proc is not None
    signals: list[str] = []
    real_terminate, real_kill = proc.terminate, proc.kill
    proc.terminate = lambda: (signals.append("terminate"), real_terminate())  # type: ignore[method-assign]
    proc.kill = lambda: (signals.append("kill"), real_kill())  # type: ignore[method-assign]

    await client.shutdown()

    assert signals == []
    assert proc.returncode == 0


@pytest.mark.asyncio
async def test_docs_cache_is_lru_bounded_and_reopens_evicted(tmp_path: Path, monkeypatch):
    """`_docs` never exceeds MAX_TRACKED_FILES; an evicted file is didClose'd and re-didOpen'ed
    (version 0) with diagnostics flowing again, so the cap is invisible to callers."""
    import agent.lsp.client as client_mod

    monkeypatch.setattr(client_mod, "MAX_TRACKED_FILES", 3)
    files = [tmp_path / f"f{i}.py" for i in range(5)]
    for f in files:
        f.write_text("print('hi')\n", encoding="utf-8")

    client = _client(tmp_path, "errors")
    real_send = client._send_notification
    sent: list = []

    async def _spy(method, params):
        sent.append((method, params))
        await real_send(method, params)

    monkeypatch.setattr(client, "_send_notification", _spy)
    await client.start()
    try:
        for f in files:
            await client.open_file(str(f), language_id="python")
        assert len(client._docs) == 3
        assert str(files[0]) not in client._docs  # least recently touched went first
        # The server releases its mirror too: every evicted doc got a didClose on the wire.
        closed = [p["textDocument"]["uri"] for m, p in sent if m == "textDocument/didClose"]
        assert closed == [client_mod.file_uri(str(files[0])), client_mod.file_uri(str(files[1]))]
        version = await client.open_file(str(files[0]), language_id="python")
        assert version == 0  # fresh didOpen, not a didChange against dropped state
        assert len(client._docs) == 3
        await client.wait_for_diagnostics(str(files[0]), version, mode="document")
        assert client.diagnostics_for(str(files[0]))
    finally:
        await client.shutdown()
