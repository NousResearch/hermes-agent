"""A versionless publishDiagnostics read while didChange is still being written must count as fresh.

``open_or_change`` used to bump ``_DocState.version`` only after awaiting the send.  Servers that omit
``version`` are credited with ``doc.version`` at receipt, so a reply that landed during that await was
tagged with the OLD version and then rejected as stale once the send resumed.  The mock replies
versionless; the paused send wrapper holds the await open until its reply has been read.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


@pytest.mark.asyncio
async def test_versionless_push_read_during_didchange_send_is_fresh(tmp_path, monkeypatch):
    src = tmp_path / "x.py"
    src.write_text("bad\n", encoding="utf-8")
    client = LSPClient(
        server_id="mock-versionless", workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER], cwd=str(tmp_path),
        env={"MOCK_LSP_SCRIPT": "versionless", "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    await client.start()
    try:
        first = await client.open_file(str(src), language_id="python")
        assert await client.wait_for_diagnostics(str(src), first, timeout=5)
        real_send = client._send_notification

        async def send_and_let_reply_land(method, params):
            seen = client._push_counter
            await real_send(method, params)
            if method == "textDocument/didChange":
                while client._push_counter == seen:
                    await asyncio.sleep(0.001)

        monkeypatch.setattr(client, "_send_notification", send_and_let_reply_land)
        src.write_text("clean\n", encoding="utf-8")
        version = await client.open_file(str(src), language_id="python")
        assert await client.wait_for_diagnostics(str(src), version, timeout=1)
        assert client.diagnostics_for(str(src), fresh_only=True) == []
    finally:
        await client.shutdown()
