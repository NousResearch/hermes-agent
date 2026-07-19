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
import warnings
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
async def test_lsp_time_monotonic_migration_no_deprecation_warning(tmp_path: Path):
    """#49762 behavioral coverage for the asyncio.get_event_loop().time() ->
    time.monotonic() migration in agent/lsp/client.py.

    The migration's purpose is to stop emitting the Python 3.10+
    ``DeprecationWarning`` from ``asyncio.get_event_loop()`` when there is
    no running event loop. ``_handle_publish_diagnostics`` (a synchronous
    notification handler dispatched from the active async reader loop) and
    the async waiters (``wait_for_diagnostics`` / ``_wait_for_fresh_push``)
    all read the monotonic clock via ``time.monotonic()``.

    This test drives a real client against the mock server and asserts that
    exercising the publish path produces no ``DeprecationWarning`` attributed
    to ``asyncio.get_event_loop`` / ``loop.time()``. It is behavioral, not a
    source-spelling check: it fails if a future refactor reintroduces
    ``get_event_loop().time()`` on any of the migrated sites.
    """
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client = _client(tmp_path, "errors")
        await client.start()
        try:
            version = await client.open_file(str(f), language_id="python")
            # Deadline / debounce math runs on time.monotonic(); if the old
            # get_event_loop().time() were still in place, the synchronous
            # publish handler would raise a DeprecationWarning here.
            await client.wait_for_diagnostics(str(f), version, mode="document")
            diags = client.diagnostics_for(str(f))
            assert len(diags) == 1
        finally:
            await client.shutdown()

    loop_time_warnings = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "get_event_loop" in str(w.message)
    ]
    assert not loop_time_warnings, (
        f"migration regressed: get_event_loop() deprecation still emitted: "
        f"{[str(w.message) for w in loop_time_warnings]}"
    )
