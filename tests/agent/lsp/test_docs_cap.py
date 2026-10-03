"""Regression tests for #126948: every diagnostic-insert site must respect MAX_TRACKED_FILES.

`_handle_publish_diagnostics` (push path) and `_pull_document_diagnostics`'s
relatedDocuments walk both create brand-new ``_DocState`` entries.  Historically
they used ``setdefault`` and bypassed the cap that ``open_file`` enforced, so a
read-only lint session (pyright/clangd emitting relatedDocuments for a big tree,
no new opens) grew ``_docs`` without bound.  Both now route through ``_track_doc``,
which requests a deferred cap pass drained at the next await point — the reader
loop drains it after every message, and the async pull path drains it inline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient, MAX_TRACKED_FILES, _DocState, file_uri


def _client(tmp_path: Path) -> LSPClient:
    return LSPClient(
        server_id="cap-probe",
        workspace_root=str(tmp_path),
        command=[sys.executable, "--version"],
        cwd=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_push_diagnostics_bounded(tmp_path: Path):
    """500 distinct publishDiagnostics URIs with no open_file must stay bounded.

    ``_drain_docs_evict`` is what the reader loop runs after each inbound message;
    the push handler itself is synchronous and cannot await the didClose-sending
    eviction, so it only flags the pass.
    """
    client = _client(tmp_path)
    for i in range(500):
        client._handle_publish_diagnostics(
            {"uri": file_uri(str(tmp_path / f"f{i}.py")), "diagnostics": []}
        )
        await client._drain_docs_evict()
    assert len(client._docs) <= MAX_TRACKED_FILES


@pytest.mark.asyncio
async def test_pull_related_documents_bounded(tmp_path: Path, monkeypatch):
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


@pytest.mark.asyncio
async def test_new_never_opened_doc_survives_saturated_opened_docs(tmp_path: Path):
    """Mixed state (the zero-coverage dimension): the cap is full of server-mirrored
    (opened) docs, then a push arrives for a never-opened path.

    Eviction must drop the *oldest* entry, never the one just created.  A cap pass
    that scans for the first ``version < 0`` victim would pick the fresh entry — the
    only never-opened one — delete it, and silently drop its diagnostics
    (``wait_for_diagnostics``/``diagnostics_for`` treat a miss as "no verdict").
    """
    client = _client(tmp_path)
    for i in range(MAX_TRACKED_FILES):
        client._docs[str(tmp_path / f"open{i}.py")] = _DocState(version=0, text="x" * 100)

    new_path = str(tmp_path / "never_opened.py")
    client._handle_publish_diagnostics(
        {"uri": file_uri(new_path), "diagnostics": [{"message": "real error"}]}
    )
    await client._drain_docs_evict()

    assert new_path in client._docs  # not evicted by its own insert
    assert len(client.diagnostics_for(new_path)) == 1  # its diagnostics survived
    assert len(client._docs) <= MAX_TRACKED_FILES  # and the cap still holds
