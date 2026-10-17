"""Seed-mode servers must not swallow the push that answers a bare didOpen.

``typescript-language-server`` is registered with ``seed=True`` because a versionless
publishDiagnostics that lands after our didChange describes the pre-edit content and must
never satisfy a waiter.  But tsserver publishes **exactly one** push per didOpen, so the
same swallow starved every open-only waiter: the post-write lint check opened the file,
waited, timed out, and the (server, root) pair was marked broken (#133602).  A push that
arrives while ``doc.version == 0`` answers the only content ever sent — the didOpen — and
must stay fresh; the baseline swallow only applies once a didChange is in flight.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient, _DocState, file_uri

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _seed_client(tmp_path: Path, script: str) -> LSPClient:
    return LSPClient(
        server_id="mock-seed", workspace_root=str(tmp_path),
        command=[sys.executable, MOCK_SERVER], cwd=str(tmp_path),
        env={"MOCK_LSP_SCRIPT": script, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        seed_diagnostics_on_first_push=True,
    )


@pytest.mark.asyncio
async def test_bare_didopen_push_satisfies_waiter_and_survives_didchange(tmp_path):
    """Versionless single-push-per-open server: both the didOpen push and the didChange
    push are credited, so each wait resolves inside budget (was: permanent timeout)."""
    src = tmp_path / "x.ts"
    src.write_text("bad\n", encoding="utf-8")
    client = _seed_client(tmp_path, "versionless")
    await client.start()
    try:
        opened = await client.open_file(str(src), language_id="typescript")
        assert await client.wait_for_diagnostics(str(src), opened, timeout=5)
        assert len(client.diagnostics_for(str(src), fresh_only=True)) == 1

        src.write_text("clean\n", encoding="utf-8")
        changed = await client.open_file(str(src), language_id="typescript")
        assert await client.wait_for_diagnostics(str(src), changed, timeout=5)
        assert client.diagnostics_for(str(src), fresh_only=True) == []
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_push_after_didchange_only_is_still_swallowed_as_baseline(tmp_path):
    """The race seed exists for: a versionless push for content we already replaced.
    It must stay untagged and never satisfy a waiter or bump the push counter."""
    client = LSPClient(
        server_id="seed-unit", workspace_root=str(tmp_path),
        command=["true"], cwd=str(tmp_path), seed_diagnostics_on_first_push=True,
    )
    path = os.path.abspath(str(tmp_path / "x.ts"))
    client._docs[path] = _DocState(version=1, text="edited")  # didChange sent, no push seen yet
    client._handle_publish_diagnostics(
        {"uri": file_uri(path), "diagnostics": [{"range": {}, "severity": 1, "message": "stale"}]}
    )
    doc = client._docs[path]
    assert doc.push_version == -1  # swallowed: no freshness tag
    assert client._push_counter == 0  # and no waiter wake-up
    assert len(doc.push) == 1  # still stored for baseline readers
