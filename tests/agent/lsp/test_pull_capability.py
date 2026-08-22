"""Tests for pull-diagnostics capability negotiation.

A server that does not advertise ``textDocument/diagnostic`` support
must never be asked for pull diagnostics — and a server that advertises
it but rejects the request with -32601 must be remembered as
pull-incapable after the first rejection.  Regression guard for a bug
where a push-only typescript server was asked once per diagnostics
call, forever (~58k -32601 errors over a week).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.lsp.client import LSPClient
from agent.lsp.protocol import LSPRequestError


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _client(workspace: Path, script: str, **env_extra: str) -> LSPClient:
    env = {
        "MOCK_LSP_SCRIPT": script,
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        **env_extra,
    }
    return LSPClient(
        server_id=f"mock-{script}",
        workspace_root=str(workspace),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(workspace),
    )


# ---------------------------------------------------------------------------
# Capability extraction (pure static)
# ---------------------------------------------------------------------------


def test_supports_pull_diagnostics_advertised():
    caps = {"textDocument": {"diagnostic": {"provider": {"workspaceDiagnostics": False}}}}
    assert LSPClient._supports_pull_diagnostics(caps) is True


def test_supports_pull_diagnostics_absent():
    assert LSPClient._supports_pull_diagnostics({}) is False
    assert LSPClient._supports_pull_diagnostics({"textDocument": {}}) is False


def test_supports_pull_diagnostics_false_provider():
    caps = {"textDocument": {"diagnostic": {"provider": False}}}
    assert LSPClient._supports_pull_diagnostics(caps) is False


def test_supports_pull_diagnostics_true_provider():
    caps = {"textDocument": {"diagnostic": {"provider": True}}}
    assert LSPClient._supports_pull_diagnostics(caps) is True


# ---------------------------------------------------------------------------
# Unadvertised capability: no pull request is ever sent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unadvertised_pull_never_requested(tmp_path: Path):
    """A push-only server (stale script: no provider advertisement) must
    not receive a single textDocument/diagnostic request."""
    client = _client(tmp_path, "stale")
    await client.start()
    try:
        assert client._pull_diagnostics_supported is False
        send = AsyncMock()
        client._send_request_with_retry = send
        await client._pull_document_diagnostics(str(tmp_path / "x.py"))
        send.assert_not_called()
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_advertised_pull_stays_enabled(tmp_path: Path):
    """A server that advertises the provider keeps the pull path live."""
    client = _client(tmp_path, "clean")
    await client.start()
    try:
        assert client._pull_diagnostics_supported is True
    finally:
        await client.shutdown()


# ---------------------------------------------------------------------------
# Lying server: advertised but -32601 — remembered after first rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_method_not_found_is_remembered(tmp_path: Path):
    client = _client(tmp_path, "clean")
    await client.start()
    try:
        assert client._pull_diagnostics_supported is True
        send = AsyncMock(
            side_effect=LSPRequestError(-32601, "method not found")
        )
        client._send_request_with_retry = send
        await client._pull_document_diagnostics(str(tmp_path / "x.py"))
        assert client._pull_diagnostics_supported is False
        # Second call short-circuits without another doomed request.
        await client._pull_document_diagnostics(str(tmp_path / "y.py"))
        send.assert_awaited_once()
    finally:
        await client.shutdown()
