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
        f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        for f in files:
            await client.open_file(str(f), language_id="python")
        assert len(client._docs) == 3
        assert str(files[0]) not in client._docs  # least recently touched went first
        version = await client.open_file(str(files[0]), language_id="python")
        assert version == 0  # fresh didOpen, not a didChange against dropped state
        assert len(client._docs) == 3
        await client.wait_for_diagnostics(str(files[0]), version, mode="document")
        assert client.diagnostics_for(str(files[0]))
    finally:
        await client.shutdown()
