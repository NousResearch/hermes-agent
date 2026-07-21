"""Backend abstraction for Mem0 Platform and OSS modes."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


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
        run_id: str | None = None,
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
        run_id: str | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {"user_id": user_id, "agent_id": agent_id, "infer": infer}
        if metadata:
            kwargs["metadata"] = metadata
        if run_id is not None:
            try:
                return self._client.add(messages, run_id=run_id, **kwargs)
            except TypeError:
                # Older mem0ai versions don't support run_id — fold it into metadata.
                merged = {**(kwargs.get("metadata") or {}), "run_id": run_id}
                kwargs["metadata"] = merged
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
        run_id: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "messages": messages,
            "user_id": user_id,
            "agent_id": agent_id,
            "infer": infer,
        }
        if metadata:
            body["metadata"] = metadata
        if run_id is not None:
            body["run_id"] = run_id
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
        if dims:
            vs_config["embedding_model_dims"] = dims
            self._recreate_collection_if_dims_changed(
                vector_store.get("provider", "qdrant"), vs_config, dims,
            )

        vector_store["config"] = vs_config

        config = {
            "vector_store": vector_store,
            "llm": _provider_block("llm"),
            "embedder": _provider_block("embedder"),
            "version": "v1.1",
        }
        self._memory = Memory.from_config(config)
        try:
            from ._entity_sidecar import init_db
            init_db()
        except Exception:
            pass

    @staticmethod
    def _recreate_collection_if_dims_changed(provider: str, vs_config: dict, expected_dims: int) -> None:
        """Delete stale vector collection when embedding dimensions change."""
        collection_name = vs_config.get("collection_name", "mem0")
        if provider == "qdrant":
            try:
                from qdrant_client import QdrantClient
                path = vs_config.get("path")
                url = vs_config.get("url")
                if path:
                    client = QdrantClient(path=path)
                elif url:
                    client = QdrantClient(url=url, api_key=vs_config.get("api_key"))
                else:
                    return
                try:
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
                    if current_dims is not None and current_dims != expected_dims:
                        client.delete_collection(collection_name)
                finally:
                    client.close()
            except Exception:
                pass
        elif provider == "pgvector":
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
        run_id: str | None = None,
    ) -> dict:
        # Structural dedup: for explicit adds (infer=False), check the Postgres sidecar
        # before writing. Prevents exact-text duplicates from accumulating in Qdrant.
        sidecar_key: str | None = None
        if not infer:
            content = next(
                (m.get("content", "") for m in messages if m.get("role") == "user"), ""
            )
            if content:
                try:
                    from ._entity_sidecar import delete_stale, lookup, normalize, upsert
                    sidecar_key = normalize(content)
                    existing_id = lookup(user_id, sidecar_key)
                    if existing_id:
                        try:
                            return self.update(existing_id, content)
                        except Exception as exc:
                            # Stale entry — Qdrant point was deleted externally
                            err = str(exc).lower()
                            if "not found" in err or "404" in err:
                                delete_stale(user_id, sidecar_key)
                            else:
                                logger.warning("sidecar update failed (%s), falling through to add", exc)
                            # Fall through to normal add
                except ImportError:
                    sidecar_key = None

        kwargs: dict[str, Any] = {"user_id": user_id, "agent_id": agent_id, "infer": infer}
        if metadata:
            kwargs["metadata"] = metadata
        if run_id is not None:
            try:
                result = self._memory.add(messages, run_id=run_id, **kwargs)
            except TypeError:
                # Older mem0ai versions don't support run_id — fold it into metadata.
                merged = {**(kwargs.get("metadata") or {}), "run_id": run_id}
                kwargs["metadata"] = merged
                result = self._memory.add(messages, **kwargs)
        else:
            result = self._memory.add(messages, **kwargs)

        # Register new point IDs in the sidecar after a successful explicit add.
        # Accept any item with an "id" — don't gate on event name (mem0 event
        # vocabulary varies by version and the ID is what we actually need).
        if sidecar_key and isinstance(result, dict):
            try:
                from ._entity_sidecar import upsert
                for item in result.get("results", []):
                    if isinstance(item, dict) and item.get("id"):
                        upsert(user_id, sidecar_key, item["id"])
                        break
            except Exception as exc:
                logger.warning("sidecar post-add upsert failed: %s", exc)

        return result

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
