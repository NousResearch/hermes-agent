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
async def test_timed_out_request_sends_cancel_notification(tmp_path: Path):
    """Abandoning a pending request must notify the server via $/cancelRequest.

    The ``hang_pull`` mock never answers ``textDocument/diagnostic``;
    when the client's wait_for times out, the server must receive a
    ``$/cancelRequest`` carrying the abandoned request's id (recorded
    into MOCK_LSP_TRACE by the mock).  Port of
    can1357/oh-my-pi#8153.
    """
    import asyncio

    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")
    trace = tmp_path / "cancel-trace.txt"

    env = {
        "MOCK_LSP_SCRIPT": "hang_pull",
        "MOCK_LSP_TRACE": str(trace),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    client = LSPClient(
        server_id="mock-hang-pull",
        workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(tmp_path),
    )
    await client.start()
    try:
        await client.open_file(str(f), language_id="python")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client._send_request(
                    "textDocument/diagnostic",
                    {"textDocument": {"uri": f"file://{f}"}},
                ),
                timeout=0.5,
            )
        # The cancel notification is written without drain; give the
        # transport + mock a moment to flush and record it.
        for _ in range(40):
            if trace.exists() and trace.read_text().strip():
                break
            await asyncio.sleep(0.05)
        assert trace.exists(), "mock never received $/cancelRequest"
        cancelled_ids = trace.read_text().split()
        assert cancelled_ids, "mock never received $/cancelRequest"
        # The abandoned diagnostic request must be among the cancelled ids.
        assert client._pending == {}
    finally:
        await client.shutdown()








