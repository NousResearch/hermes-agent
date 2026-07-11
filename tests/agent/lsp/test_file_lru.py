"""LRU cap on the client's tracked-file text cache.

``LSPClient._files`` holds every opened file's full text (and the server
mirrors each open document in its own memory) with no per-file eviction —
a marathon coding run pinned the contents of every file it ever touched
until the whole client was idle-reaped. The cap evicts the least recently
opened files with a proper ``didClose``; evicted files transparently
re-``didOpen`` on their next touch.

Runs against the in-process mock LSP server, same harness as
test_client_e2e.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import agent.lsp.client as client_mod
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
async def test_open_files_evicted_beyond_cap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(client_mod, "MAX_TRACKED_FILES", 3)

    files = []
    for i in range(5):
        f = tmp_path / f"f{i}.py"
        f.write_text(f"print({i})\n")
        files.append(os.path.abspath(str(f)))

    client = _client(tmp_path)
    await client.start()
    try:
        for path in files:
            await client.open_file(path, language_id="python")

        # Cap holds; the two least-recently-opened files were evicted.
        assert len(client._files) == 3
        assert files[0] not in client._files
        assert files[1] not in client._files
        assert set(files[2:]).issubset(client._files)

        # An evicted file re-opens transparently as a fresh didOpen.
        version = await client.open_file(files[0], language_id="python")
        assert version == 0
        assert files[0] in client._files
        assert len(client._files) == 3

        # A still-tracked file keeps taking the didChange path.
        version = await client.open_file(files[4], language_id="python")
        assert version == 1
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_reopen_refreshes_lru_recency(tmp_path: Path, monkeypatch):
    """Touching a tracked file must move it to the back of the eviction
    queue — LRU, not FIFO-by-first-open."""
    monkeypatch.setattr(client_mod, "MAX_TRACKED_FILES", 2)

    a, b, c = (tmp_path / n for n in ("a.py", "b.py", "c.py"))
    for f in (a, b, c):
        f.write_text("print()\n")
    a_p, b_p, c_p = (os.path.abspath(str(f)) for f in (a, b, c))

    client = _client(tmp_path)
    await client.start()
    try:
        await client.open_file(a_p, language_id="python")
        await client.open_file(b_p, language_id="python")
        # Touch a again: b becomes least recently used.
        await client.open_file(a_p, language_id="python")
        await client.open_file(c_p, language_id="python")

        assert b_p not in client._files
        assert {a_p, c_p} == set(client._files)
    finally:
        await client.shutdown()


def test_delta_baseline_capped(monkeypatch):
    """The manager's per-path diagnostic baselines are bounded too — same
    grow-per-path-forever shape, smaller entries."""
    import agent.lsp.manager as manager_mod
    from agent.lsp.manager import LSPService

    monkeypatch.setattr(manager_mod, "_DELTA_BASELINE_CAP", 4)
    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="auto",
    )
    try:
        for i in range(10):
            svc._delta_baseline[f"/tmp/file{i}.py"] = [{"message": f"d{i}"}]
            svc._cap_delta_baseline()
        assert len(svc._delta_baseline) == 4
        assert "/tmp/file9.py" in svc._delta_baseline
        assert "/tmp/file0.py" not in svc._delta_baseline
    finally:
        svc.shutdown()
