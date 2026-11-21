"""Regression tests for #126948: diagnostic insert sites must respect MAX_TRACKED_FILES.

``_handle_publish_diagnostics`` (push path) and the two insert sites in
``_pull_document_diagnostics`` (items + relatedDocuments) created brand-new
``_DocState`` entries with no cap. Feeding 500 distinct URIs with no
``open_file`` calls must stay bounded. Opened docs (version >= 0) are
server-mirrored and must never be dropped by the sync path, which cannot
send the didClose the full ``_evict_lru_docs`` performs.
"""
from __future__ import annotations

import sys

import pytest

from agent.lsp.client import MAX_TRACKED_FILES, LSPClient, _DocState, file_uri


def _client(tmp_path) -> LSPClient:
    return LSPClient(
        server_id="cap-probe",
        workspace_root=str(tmp_path),
        command=[sys.executable, "--version"],
        cwd=str(tmp_path),
    )


def test_push_diagnostics_bounded(tmp_path):
    client = _client(tmp_path)
    for i in range(500):
        client._handle_publish_diagnostics(
            {"uri": file_uri(str(tmp_path / f"f{i}.py")), "diagnostics": []}
        )
    assert len(client._docs) <= MAX_TRACKED_FILES


@pytest.mark.asyncio
async def test_pull_related_documents_bounded(tmp_path, monkeypatch):
    client = _client(tmp_path)
    related = {
        file_uri(str(tmp_path / f"g{i}.py")): {"items": []} for i in range(500)
    }

    async def fake_send(method, params, **kwargs):
        return {"items": [], "relatedDocuments": related}

    monkeypatch.setattr(client, "_send_request_with_retry", fake_send)
    target = tmp_path / "t.py"
    target.write_text("x = 1\n")
    await client._pull_document_diagnostics(str(target))
    assert len(client._docs) <= MAX_TRACKED_FILES


def test_sync_evict_never_drops_opened_docs(tmp_path):
    client = _client(tmp_path)
    opened = str(tmp_path / "keep.py")
    client._docs[opened] = _DocState(version=0, text="x = 1\n")
    for i in range(500):
        client._handle_publish_diagnostics(
            {"uri": file_uri(str(tmp_path / f"h{i}.py")), "diagnostics": []}
        )
    assert opened in client._docs
    # Only the pinned opened doc may stand over the cap; spillover is bounded.
    assert len(client._docs) <= MAX_TRACKED_FILES + 1
