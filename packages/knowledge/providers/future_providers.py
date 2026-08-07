"""Future backends (Step 9).

Each is a drop-in KnowledgeProvider. Adding a backend requires ONLY a new
class here plus one line in ``PROVIDER_REGISTRY`` — Hermes' reasoning engine,
the knowledge_search tool, and KnowledgeService stay untouched.

These ship as thin HTTP scaffolds: the vector-search wire calls are
implemented, so enabling one is a config change plus dependency/endpoint
availability, not a Hermes change.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, List, Optional

from ..embeddings import embed
from ..provider import KnowledgeProvider
from ..types import Chunk, Citation, Document, HealthStatus, IndexResult, SearchResult


class _HttpVectorProvider(KnowledgeProvider):
    """Shared plumbing for HTTP vector databases."""

    name = "http-vector"

    def __init__(self, base_url: str, api_key: str = "", collection: str = "hermes",
                 default_workspace: str = "default", timeout: float = 30.0,
                 opener: Optional[Any] = None):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.collection = collection
        self.default_workspace = default_workspace
        self.timeout = timeout
        self._opener = opener

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method,
        )
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("api-key", self.api_key)
            req.add_header("Authorization", f"Bearer {self.api_key}")
        opener = self._opener or urllib.request.urlopen
        with opener(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        return json.loads(body) if body.strip() else {}

    def _mk_chunks(self, points: List[dict], workspace: str) -> List[Chunk]:
        out = []
        for i, p in enumerate(points):
            payload = p.get("payload") or p.get("metadata") or {}
            score = float(p.get("score", 0.0) or 0.0)
            cid = str(p.get("id", f"{workspace}#{i}"))
            path = payload.get("path", "")
            out.append(Chunk(
                id=cid, text=payload.get("text", ""), score=score,
                document_id=str(payload.get("document_id", cid)),
                citation=Citation(
                    title=payload.get("title") or os.path.basename(path) or cid,
                    file=os.path.basename(path), path=path, score=round(score, 4),
                    workspace=workspace, chunk_id=cid,
                    document_id=str(payload.get("document_id", cid)),
                    provider=self.name,
                ),
            ))
        return out

    # subclasses override the four wire methods below
    def search(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        raise NotImplementedError

    def retrieve(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        res = self.search(query, limit=limit, workspace=workspace, filters=filters)
        res.answer = "\n\n".join(c.text for c in res.chunks)
        return res

    def index(self, document: Document) -> IndexResult:
        raise NotImplementedError(f"{self.name}.index not configured")

    def update(self, document: Document) -> IndexResult:
        return self.index(document)

    def delete(self, document_id: str, workspace: Optional[str] = None) -> IndexResult:
        raise NotImplementedError(f"{self.name}.delete not configured")

    def health(self) -> HealthStatus:
        t0 = time.perf_counter()
        try:
            self._request("GET", "/")
            return HealthStatus(True, self.name, self.base_url,
                                (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return HealthStatus(False, self.name, str(exc),
                                (time.perf_counter() - t0) * 1000)


class QdrantProvider(_HttpVectorProvider):
    name = "qdrant"

    def search(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        t0 = time.perf_counter()
        ws = workspace or self.default_workspace
        data = self._request(
            "POST", f"/collections/{self.collection}/points/search",
            {"vector": embed(query), "limit": int(limit), "with_payload": True},
        )
        chunks = self._mk_chunks(data.get("result") or [], ws)
        return SearchResult(query=query, chunks=chunks, provider=self.name, workspace=ws,
                            elapsed_ms=(time.perf_counter() - t0) * 1000,
                            confidence=chunks[0].score if chunks else 0.0)


class WeaviateProvider(_HttpVectorProvider):
    name = "weaviate"

    def search(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        t0 = time.perf_counter()
        ws = workspace or self.default_workspace
        gql = {
            "query": (
                "{Get{%s(limit:%d nearText:{concepts:[\"%s\"]})"
                "{text title path _additional{id certainty}}}}"
                % (self.collection, int(limit), query.replace('"', ' '))
            )
        }
        data = self._request("POST", "/v1/graphql", gql)
        raw = (((data.get("data") or {}).get("Get") or {}).get(self.collection)) or []
        points = [{
            "id": (r.get("_additional") or {}).get("id"),
            "score": (r.get("_additional") or {}).get("certainty", 0.0),
            "payload": {"text": r.get("text", ""), "title": r.get("title", ""),
                        "path": r.get("path", "")},
        } for r in raw]
        chunks = self._mk_chunks(points, ws)
        return SearchResult(query=query, chunks=chunks, provider=self.name, workspace=ws,
                            elapsed_ms=(time.perf_counter() - t0) * 1000,
                            confidence=chunks[0].score if chunks else 0.0)


class ChromaProvider(_HttpVectorProvider):
    name = "chroma"

    def search(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        t0 = time.perf_counter()
        ws = workspace or self.default_workspace
        data = self._request(
            "POST", f"/api/v1/collections/{self.collection}/query",
            {"query_embeddings": [embed(query)], "n_results": int(limit),
             "include": ["documents", "metadatas", "distances"]},
        )
        docs = (data.get("documents") or [[]])[0]
        metas = (data.get("metadatas") or [[]])[0]
        dists = (data.get("distances") or [[]])[0]
        ids = (data.get("ids") or [[]])[0]
        points = [{
            "id": ids[i] if i < len(ids) else str(i),
            "score": 1.0 - float(dists[i]) if i < len(dists) else 0.0,
            "payload": {**(metas[i] if i < len(metas) else {}), "text": docs[i]},
        } for i in range(len(docs))]
        chunks = self._mk_chunks(points, ws)
        return SearchResult(query=query, chunks=chunks, provider=self.name, workspace=ws,
                            elapsed_ms=(time.perf_counter() - t0) * 1000,
                            confidence=chunks[0].score if chunks else 0.0)


class PgVectorProvider(KnowledgeProvider):
    """pgvector backend. Requires psycopg (optional dependency)."""

    name = "pgvector"

    def __init__(self, dsn: str, table: str = "hermes_chunks",
                 default_workspace: str = "default", **_: Any):
        self.dsn = dsn
        self.table = table
        self.default_workspace = default_workspace

    def _conn(self):
        try:
            import psycopg  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pgvector provider requires 'psycopg'") from exc
        return psycopg.connect(self.dsn)

    def search(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        t0 = time.perf_counter()
        ws = workspace or self.default_workspace
        vec = embed(query)
        sql = (f"SELECT chunk_id, document_id, title, path, text, "
               f"1-(embedding <=> %s::vector) AS score FROM {self.table} "
               f"WHERE workspace=%s ORDER BY embedding <=> %s::vector LIMIT %s")
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (vec, ws, vec, int(limit)))
            rows = cur.fetchall()
        chunks = [Chunk(
            id=str(r[0]), text=r[4], score=float(r[5]), document_id=str(r[1]),
            citation=Citation(title=r[2] or "", file=os.path.basename(r[3] or ""),
                              path=r[3] or "", score=round(float(r[5]), 4),
                              workspace=ws, chunk_id=str(r[0]),
                              document_id=str(r[1]), provider=self.name),
        ) for r in rows]
        return SearchResult(query=query, chunks=chunks, provider=self.name, workspace=ws,
                            elapsed_ms=(time.perf_counter() - t0) * 1000,
                            confidence=chunks[0].score if chunks else 0.0)

    def retrieve(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        res = self.search(query, limit=limit, workspace=workspace, filters=filters)
        res.answer = "\n\n".join(c.text for c in res.chunks)
        return res

    def index(self, document: Document) -> IndexResult:
        raise NotImplementedError("pgvector.index not configured")

    def update(self, document: Document) -> IndexResult:
        return self.index(document)

    def delete(self, document_id: str, workspace: Optional[str] = None) -> IndexResult:
        raise NotImplementedError("pgvector.delete not configured")

    def health(self) -> HealthStatus:
        t0 = time.perf_counter()
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
            return HealthStatus(True, self.name, "connected",
                                (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return HealthStatus(False, self.name, str(exc),
                                (time.perf_counter() - t0) * 1000)
