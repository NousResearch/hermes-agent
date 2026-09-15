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


pytestmark = pytest.mark.live_system_guard_bypass


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
async def test_client_handles_large_stderr_line(tmp_path: Path):
    """LSP stderr drain must not crash or deadlock when the server emits a line
    larger than asyncio's old 64 KiB default StreamReader limit."""
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n", encoding="utf-8")

    client = _client(tmp_path, "large_stderr")
    await client.start()
    try:
        assert client.is_running
        version = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert diags == []
    finally:
        await client.shutdown()
    assert not client.is_running


@pytest.mark.asyncio
async def test_client_handles_stderr_line_over_stream_limit(tmp_path: Path):
    """An over-limit stderr line must not terminate the drain task."""
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n", encoding="utf-8")

    client = _client(tmp_path, "oversized_stderr")
    await client.start()
    try:
        assert client.is_running
        version = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), version, mode="document")
        assert client.diagnostics_for(str(f)) == []
    finally:
        await client.shutdown()
    assert not client.is_running
