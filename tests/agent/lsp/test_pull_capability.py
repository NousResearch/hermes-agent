"""Pull diagnostics are requested only from a server that serves them (#126949).

A push-only server asked for ``textDocument/diagnostic`` answers -32601 at once, and the wait loop used
to re-issue the pull on every pass: thousands of rejected requests per edited file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _client(workspace: Path, script: str, **env_extra: str) -> LSPClient:
    env = {"MOCK_LSP_SCRIPT": script, "PYTHONPATH": os.environ.get("PYTHONPATH", ""), **env_extra}
    return LSPClient(server_id=f"mock-{script}", workspace_root=str(workspace),
                     command=[sys.executable, MOCK_SERVER], env=env, cwd=str(workspace))


async def _wait(client: LSPClient, path: Path, budget: float) -> tuple[bool, int]:
    """(fresh, requests sent during the wait), counted by the client's own JSON-RPC id counter."""
    version = await client.open_file(str(path))
    before = client._next_id
    fresh = await client.wait_for_diagnostics(str(path), version, timeout=budget)
    return fresh, client._next_id - before


@pytest.mark.asyncio
async def test_rejected_pull_is_not_reissued(tmp_path: Path):
    """``silent`` advertises pull, rejects it with -32601 and never pushes: the worst case."""
    path = tmp_path / "x.py"
    path.write_text("x = 1\n", encoding="utf-8")
    client = _client(tmp_path, "silent")
    await client.start()
    try:
        assert (await _wait(client, path, 1.0))[1] == 1
        assert (await _wait(client, path, 0.3))[1] == 0  # latched for the process
    finally:
        await client.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(("script", "pull_mode", "pulls"), [
    ("clean", "static", 1),    # diagnosticProvider at initialize
    ("clean", "none", 0),      # push-only: never asked
    ("late_pull", "static", 1),  # no static capability; registers mid-wait and never pushes: the wait must wake and pull
])
async def test_pull_follows_server_capability(tmp_path: Path, script: str, pull_mode: str, pulls: int):
    path = tmp_path / "x.py"
    path.write_text("x = 1\n", encoding="utf-8")
    client = _client(tmp_path, script, MOCK_LSP_PULL=pull_mode)
    await client.start()
    try:
        assert await _wait(client, path, 2.0) == (True, pulls)
    finally:
        await client.shutdown()
