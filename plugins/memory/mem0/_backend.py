"""Backend abstraction for Mem0 Platform and OSS modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Mem0Backend(ABC):
    """Unified interface over Platform (MemoryClient) and OSS (Memory) backends."""

    @abstractmethod
    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        ...

    @abstractmethod
    def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        ...

    @abstractmethod
    def update(self, memory_id: str, text: str) -> dict:
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> dict:
        ...

    def close(self) -> None:
        pass


def _unwrap_results(response: Any) -> list:
    """Normalize API response — extract results list from dict or pass through."""
    if isinstance(response, dict):
        return response.get("results", [])
    if isinstance(response, list):
        return response
    return []


class PlatformBackend(Mem0Backend):
    """Wraps mem0.MemoryClient for Mem0 Platform (cloud API)."""

    def __init__(self, api_key: str):
        from mem0 import MemoryClient
        self._client = MemoryClient(api_key=api_key)

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        response = self._client.search(query, filters=filters, top_k=top_k, rerank=rerank)
        return _unwrap_results(response)

    def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {"user_id": user_id, "agent_id": agent_id, "infer": infer}
        if metadata:
            kwargs["metadata"] = metadata
        return self._client.add(messages, **kwargs)

    def update(self, memory_id: str, text: str) -> dict:
        self._client.update(memory_id=memory_id, text=text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        self._client.delete(memory_id=memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}


class SelfHostedBackend(Mem0Backend):
    """Direct HTTP backend for a self-hosted Mem0 server (the FastAPI ``server/``).

    mem0.MemoryClient can't be reused for self-hosted: it is hardwired to the
    cloud API — ``Authorization: Token`` auth and a ``GET /v1/ping/`` validation
    call in ``__init__`` that the self-hosted server does not expose (it would
    404 before any real request). This client talks to that server directly,
    using its actual contract: ``X-API-Key`` auth and the ``/memories`` /
    ``/search`` routes.
    """

    def __init__(self, api_key: str, host: str, transport=None):
        import httpx

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key  # omitted only for AUTH_DISABLED servers
        # Connect-level retries smooth over transient blips so a single
        # dropped SYN doesn't count toward the provider failure breaker.
        # ``transport`` is injectable for tests (httpx.MockTransport).
        if transport is None:
            transport = httpx.HTTPTransport(retries=2)
        self._client = httpx.Client(
            base_url=host.rstrip("/"), headers=headers, timeout=30.0,
            transport=transport,
        )

    def _json(self, method: str, path: str, **kwargs) -> Any:
        resp = self._client.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        # rerank is a platform-only feature; the self-hosted /search ignores it.
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if filters:
            body["filters"] = filters  # user_id belongs in filters (top-level is deprecated)
        return _unwrap_results(self._json("POST", "/search", json=body))

    def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "messages": messages,
            "user_id": user_id,
            "agent_id": agent_id,
            "infer": infer,
        }
        if metadata:
            body["metadata"] = metadata
        return self._json("POST", "/memories", json=body)

    def update(self, memory_id: str, text: str) -> dict:
        self._json("PUT", f"/memories/{memory_id}", json={"text": text})
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        self._json("DELETE", f"/memories/{memory_id}")
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class OSSBackend(Mem0Backend):
    """Wraps mem0.Memory for self-hosted (OSS) mode."""

    def __init__(self, oss_config: dict):
        import os
        from mem0 import Memory

        def _provider_block(name: str) -> dict:
            block = dict(oss_config[name])
            provider = str(block.get("provider") or "").strip().lower()
            provider_config = dict(block.get("config", {}))
            legacy_base = provider_config.pop("api_base", None)
            if legacy_base:
                from ._oss_providers import EMBEDDER_PROVIDERS, LLM_PROVIDERS

                provider_def = (
                    LLM_PROVIDERS if name == "llm" else EMBEDDER_PROVIDERS
                ).get(provider, {})
                canonical_key = provider_def.get("base_url_key")
                if canonical_key:
                    provider_config.setdefault(canonical_key, legacy_base)
            block["config"] = provider_config
            return block

        vector_store = dict(oss_config["vector_store"])
        vs_config = dict(vector_store.get("config", {}))

        if "path" in vs_config:
            vs_config["path"] = os.path.expanduser(vs_config["path"])

        embedder_config = oss_config.get("embedder", {}).get("config", {})
        dims = embedder_config.get("embedding_dims")
        if not dims:
            from ._oss_providers import KNOWN_DIMS
            model = embedder_config.get("model", "")
            dims = KNOWN_DIMS.get(model)

        provider = vector_store.get("provider", "qdrant") or "qdrant"

        if dims:
            vs_config["embedding_model_dims"] = dims
            # pgvector (and other non-qdrant stores) are safe to check before
            # Memory is built — they hold no local file lock. Qdrant is handled
            # after construction via Memory's own client, so we never open a
            # second QdrantClient against the same on-disk path (which
            # deadlocks on Windows file locks).
            if provider != "qdrant":
                self._recreate_collection_if_dims_changed(provider, vs_config, dims)

        vector_store["config"] = vs_config

        config = {
            "vector_store": vector_store,
            "llm": _provider_block("llm"),
            "embedder": _provider_block("embedder"),
            "version": "v1.1",
        }
        self._memory = Memory.from_config(config)

        if dims and provider == "qdrant":
            self._recreate_qdrant_if_dims_changed(dims)

    def _recreate_qdrant_if_dims_changed(self, expected_dims: int) -> None:
        """Recreate the Qdrant collection when its stored embedding dims differ.

        Runs after ``Memory.from_config`` and uses the Memory's own
        ``vector_store`` client, rather than opening a second QdrantClient
        against the same local path (which deadlocks on Windows file locks).

        ``Memory.from_config`` already created the collection (or left an
        existing one untouched, since ``create_col`` skips when it exists). If
        that existing collection has the wrong dims we delete it and rebuild it
        *only through the vector store's ``create_col``*, which restores BM25
        sparse vectors and filter indexes.  A bare ``create_collection`` would
        drop them, so we never delete unless we know we can properly rebuild.
        """
        import logging
        log = logging.getLogger(__name__)
        try:
            vs = getattr(self._memory, "vector_store", None)
            if vs is None:
                return
            client = getattr(vs, "client", None)
            if client is None:
                return
            collection_name = self._memory.collection_name
            if not client.collection_exists(collection_name):
                return
            info = client.get_collection(collection_name)
            vectors = info.config.params.vectors
            # Named-vector collections expose a dict; unnamed expose an object with .size.
            if isinstance(vectors, dict):
                first = next(iter(vectors.values()), None)
                current_dims = first.size if first else None
            else:
                current_dims = getattr(vectors, "size", None)
            if current_dims is None or current_dims == expected_dims:
                return

            # Only proceed if the vector store exposes a proper create_col.
            # A bare client.create_collection cannot restore BM25 sparse vectors,
            # filter indexes, on_disk, or named-vector config — the resulting
            # degraded collection is worse than the original dim mismatch.
            if not hasattr(vs, "create_col"):
                log.warning(
                    "Qdrant collection %r has dims %d != expected %d, but "
                    "vector store has no create_col — skipping recreate.",
                    collection_name, current_dims, expected_dims,
                )
                return

            on_disk = getattr(vs, "on_disk", False)
            client.delete_collection(collection_name)
            vs.create_col(expected_dims, on_disk)
        except Exception:
            log.exception(
                "Failed to recreate Qdrant collection %r with dims %d",
                getattr(self._memory, "collection_name", "mem0"),
                expected_dims,
            )

    @staticmethod
    def _recreate_collection_if_dims_changed(provider: str, vs_config: dict, expected_dims: int) -> None:
        """Delete a stale pgvector table when embedding dimensions change.

        Qdrant is handled separately by ``_recreate_qdrant_if_dims_changed`` on
        the already-constructed Memory, to avoid a second QdrantClient.
        """
        collection_name = vs_config.get("collection_name", "mem0")
        if provider == "pgvector":
            try:
                import psycopg2
                from psycopg2 import sql as pgsql
                conn_params = {}
                for k in ("host", "port", "user", "password", "dbname"):
                    if vs_config.get(k):
                        conn_params[k] = vs_config[k]
                if vs_config.get("sslmode"):
                    conn_params["sslmode"] = vs_config["sslmode"]
                conn = psycopg2.connect(**conn_params)
                conn.autocommit = True
                try:
                    cur = conn.cursor()
                    try:
                        cur.execute(
                            "SELECT atttypmod FROM pg_attribute "
                            "WHERE attrelid = %s::regclass AND attname = 'vector'",
                            (collection_name,),
                        )
                        row = cur.fetchone()
                        if row and row[0] > 0 and row[0] != expected_dims:
                            cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(
                                pgsql.Identifier(collection_name)
                            ))
                    finally:
                        cur.close()
                finally:
                    conn.close()
            except Exception:
                pass

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        response = self._memory.search(query, filters=filters, top_k=top_k)
        return _unwrap_results(response)

    def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {"user_id": user_id, "agent_id": agent_id, "infer": infer}
        if metadata:
            kwargs["metadata"] = metadata
        return self._memory.add(messages, **kwargs)

    def update(self, memory_id: str, text: str) -> dict:
        self._memory.update(memory_id, data=text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        self._memory.delete(memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self):
        try:
            telemetry = getattr(self._memory, "telemetry", None)
            if telemetry and hasattr(telemetry, "posthog"):
                try:
                    telemetry.posthog.shutdown()
                except Exception:
                    pass
            if hasattr(self._memory, "close"):
                self._memory.close()
            vs = getattr(self._memory, "vector_store", None)
            if vs and hasattr(vs, "close"):
                vs.close()
            client = getattr(vs, "client", None)
            if client and hasattr(client, "close"):
                client.close()
        except Exception:
            pass
