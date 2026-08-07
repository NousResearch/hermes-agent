"""AnythingLLMProvider — REST adapter (Step 3).

Pure transport + mapping. No business logic: no caching, no reranking,
no retries (KnowledgeService owns those), no reasoning.

API surface used (AnythingLLM v1 developer API):
    GET  /api/v1/auth
    POST /api/v1/workspace/{slug}/vector-search
    POST /api/v1/workspace/{slug}/chat
    POST /api/v1/document/raw-text
    POST /api/v1/workspace/{slug}/update-embeddings
    DELETE /api/v1/system/remove-documents
    GET  /api/v1/documents
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from ..provider import KnowledgeProvider
from ..types import Chunk, Citation, Document, HealthStatus, IndexResult, SearchResult


class AnythingLLMError(RuntimeError):
    pass


class AnythingLLMProvider(KnowledgeProvider):
    name = "anythingllm"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        default_workspace: str = "default",
        timeout: float = 30.0,
        opener: Optional[Any] = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or os.getenv("ANYTHINGLLM_API_KEY", "")
        self.default_workspace = default_workspace
        self.timeout = timeout
        self._opener = opener  # injectable for tests

    # -- transport -----------------------------------------------------
    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        url = f"{self.base_url}/api/v1{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            opener = self._opener or urllib.request.urlopen
            with opener(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500] if hasattr(exc, "read") else ""
            raise AnythingLLMError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
        except Exception as exc:
            raise AnythingLLMError(f"{type(exc).__name__} {method} {path}: {exc}") from exc
        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}

    # -- mapping -------------------------------------------------------
    def _to_chunks(self, raw: List[dict], workspace: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        for i, item in enumerate(raw or []):
            meta = item.get("metadata") or {}
            score = float(item.get("score", item.get("similarity", 0.0)) or 0.0)
            path = meta.get("url") or meta.get("docpath") or meta.get("source") or ""
            path = path.replace("file://", "")
            doc_id = str(item.get("docId") or meta.get("id") or meta.get("docId") or path)
            chunk_id = str(item.get("id") or f"{doc_id}#{i}")
            title = meta.get("title") or (os.path.basename(path) if path else doc_id)
            chunks.append(Chunk(
                id=chunk_id,
                text=item.get("text") or item.get("pageContent") or "",
                score=score,
                document_id=doc_id,
                citation=Citation(
                    title=title,
                    file=os.path.basename(path) if path else title,
                    path=path,
                    score=round(score, 4),
                    workspace=workspace,
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    provider=self.name,
                ),
            ))
        return chunks

    # -- read ----------------------------------------------------------
    def search(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        t0 = time.perf_counter()
        ws = workspace or self.default_workspace
        payload: Dict[str, Any] = {"query": query, "topN": int(limit)}
        if filters and filters.get("score_threshold") is not None:
            payload["scoreThreshold"] = filters["score_threshold"]
        data = self._request("POST", f"/workspace/{ws}/vector-search", payload)
        chunks = self._to_chunks(data.get("results") or data.get("sources") or [], ws)
        return SearchResult(
            query=query, chunks=chunks[:limit], provider=self.name, workspace=ws,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            confidence=chunks[0].score if chunks else 0.0,
        )

    def retrieve(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        t0 = time.perf_counter()
        ws = workspace or self.default_workspace
        data = self._request(
            "POST", f"/workspace/{ws}/chat",
            {"message": query, "mode": "query"},
        )
        chunks = self._to_chunks(data.get("sources") or [], ws)
        return SearchResult(
            query=query,
            answer=data.get("textResponse") or "",
            chunks=chunks[:limit],
            provider=self.name,
            workspace=ws,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            confidence=chunks[0].score if chunks else 0.0,
            error=data.get("error") or "",
        )

    # -- write ---------------------------------------------------------
    def index(self, document: Document) -> IndexResult:
        ws = document.workspace or self.default_workspace
        data = self._request("POST", "/document/raw-text", {
            "textContent": document.content,
            "metadata": {
                "title": document.title,
                "docSource": document.source,
                "url": f"file://{document.path}" if document.path else "",
                "id": document.id,
                **(document.metadata or {}),
            },
        })
        docs = ((data.get("documents") or [{}])[0]) if isinstance(data, dict) else {}
        location = docs.get("location") or document.id
        self._request("POST", f"/workspace/{ws}/update-embeddings",
                      {"adds": [location]})
        return IndexResult(True, document.id, "indexed", location)

    def update(self, document: Document) -> IndexResult:
        self.delete(document.id, document.workspace)
        res = self.index(document)
        return IndexResult(res.ok, document.id, "updated", res.detail)

    def delete(self, document_id: str, workspace: Optional[str] = None) -> IndexResult:
        ws = workspace or self.default_workspace
        try:
            self._request("POST", f"/workspace/{ws}/update-embeddings",
                          {"deletes": [document_id]})
            self._request("DELETE", "/system/remove-documents",
                          {"names": [document_id]})
            return IndexResult(True, document_id, "deleted")
        except AnythingLLMError as exc:
            return IndexResult(False, document_id, "error", str(exc))

    def list_documents(self, workspace=None) -> List[Dict[str, Any]]:
        data = self._request("GET", "/documents")
        items = (data.get("localFiles") or {}).get("items") or []
        out: List[Dict[str, Any]] = []
        for folder in items:
            for f in folder.get("items", []) or []:
                out.append({
                    "id": f.get("id") or f.get("name"),
                    "title": (f.get("title") or f.get("name")),
                    "path": f.get("url", "").replace("file://", ""),
                    "checksum": f.get("published") or "",
                })
        return out

    # -- spec-named aliases (uploadDocument/updateDocument/deleteDocument) --
    # The sync worker talks to the provider through these names; they are thin
    # aliases so the interface contract (index/update/delete) stays canonical.
    uploadDocument = index
    updateDocument = update
    deleteDocument = delete

    def health(self) -> HealthStatus:
        t0 = time.perf_counter()
        try:
            data = self._request("GET", "/auth")
            ok = bool(data.get("authenticated", True))
            return HealthStatus(ok, self.name, f"{self.base_url} authenticated={ok}",
                                (time.perf_counter() - t0) * 1000)
        except AnythingLLMError as exc:
            return HealthStatus(False, self.name, str(exc),
                                (time.perf_counter() - t0) * 1000)
