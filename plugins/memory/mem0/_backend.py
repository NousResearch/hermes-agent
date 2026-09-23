"""Backend abstraction for Mem0 Platform and OSS modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import closing, suppress
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _widen_qdrant_timeout(vector_store: dict) -> None:
    """Give remote Qdrant enough time for Mem0's entity-boost fan-out."""
    cfg = vector_store.get("config") or {}
    if str(vector_store.get("provider") or "qdrant").lower() != "qdrant":
        return
    if cfg.get("client") is not None or cfg.get("path"):
        return
    try:
        from qdrant_client import QdrantClient
    except Exception:
        return
    kwargs: dict[str, Any] = {"timeout": 30}
    if cfg.get("url"):
        kwargs["url"] = cfg["url"]
    else:
        if not cfg.get("host"):
            return
        kwargs["host"] = cfg["host"]
        if cfg.get("port"):
            kwargs["port"] = cfg["port"]
    if cfg.get("api_key"):
        kwargs["api_key"] = cfg["api_key"]
    if cfg.get("https") is not None:
        kwargs["https"] = cfg["https"]
    try:
        cfg["client"] = QdrantClient(**kwargs)
    except Exception:
        return


def _add_kwargs(user_id: str, agent_id: str, infer: bool, metadata: dict | None) -> dict[str, Any]:
    return {"user_id": user_id, "agent_id": agent_id, "infer": infer, **({"metadata": metadata} if metadata else {})}


def _unwrap_results(response: Any) -> list:
    """Normalize API response — extract results list from dict or pass through."""
    return response.get("results", []) if isinstance(response, dict) else response if isinstance(response, list) else []


class Mem0Backend(ABC):
    """Unified interface over Platform (MemoryClient), self-hosted (HTTP) and OSS (Memory) backends.
    update()/delete() are template methods: subclasses implement raw ``_update``/``_delete``."""

    @abstractmethod
    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]: ...
    @abstractmethod
    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict: ...
    @abstractmethod
    def _update(self, memory_id: str, text: str) -> None: ...
    @abstractmethod
    def _delete(self, memory_id: str) -> None: ...

    def update(self, memory_id: str, text: str) -> dict:
        self._update(memory_id, text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        self._delete(memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self) -> None:
        pass


class PlatformBackend(Mem0Backend):
    """Wraps mem0.MemoryClient for Mem0 Platform (cloud API)."""

    def __init__(self, api_key: str):
        from mem0 import MemoryClient
        self._client = MemoryClient(api_key=api_key)

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        return _unwrap_results(self._client.search(query, filters=filters, top_k=top_k, rerank=rerank))

    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict:
        return self._client.add(messages, **_add_kwargs(user_id, agent_id, infer, metadata))

    def _update(self, memory_id: str, text: str) -> None:
        self._client.update(memory_id=memory_id, text=text)

    def _delete(self, memory_id: str) -> None:
        self._client.delete(memory_id=memory_id)


class SelfHostedBackend(Mem0Backend):
    """Direct HTTP backend for a self-hosted Mem0 server (the FastAPI ``server/``).
    mem0.MemoryClient is hardwired to the cloud API (``Authorization: Token``, ``GET /v1/ping/`` in ``__init__``),
    so this speaks the server's real contract: ``X-API-Key`` auth and the ``/memories`` / ``/search`` routes."""

    def __init__(self, api_key: str, host: str, transport=None):
        import httpx
        headers = {"Content-Type": "application/json", **({"X-API-Key": api_key} if api_key else {})}  # key omitted only for AUTH_DISABLED servers
        # Connect-level retries keep one dropped SYN from counting toward the breaker. ``transport`` is injectable for tests.
        self._client = httpx.Client(base_url=host.rstrip("/"), headers=headers, timeout=30.0, transport=transport or httpx.HTTPTransport(retries=2))

    def _json(self, method: str, path: str, **kwargs) -> Any:
        resp = self._client.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        # rerank is platform-only; the self-hosted /search ignores it. user_id belongs in filters (top-level is deprecated).
        return _unwrap_results(self._json("POST", "/search", json={"query": query, "top_k": top_k, **({"filters": filters} if filters else {})}))

    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict:
        return self._json("POST", "/memories", json={"messages": messages, **_add_kwargs(user_id, agent_id, infer, metadata)})

    def _update(self, memory_id: str, text: str) -> None:
        self._json("PUT", f"/memories/{memory_id}", json={"text": text})

    def _delete(self, memory_id: str) -> None:
        self._json("DELETE", f"/memories/{memory_id}")

    def close(self) -> None:
        with suppress(Exception):
            self._client.close()


_DIRECT_OPENAI_PROVIDER = "hermes_openai"
_DIRECT_OPENAI_CLASS_PATH = "plugins.memory.mem0._openai_llm.DirectOpenAILLM"
_CLOUDFLARE_EMBED_MODEL = "@cf/baai/bge-m3"
_CLOUDFLARE_RERANK_MODEL = "@cf/baai/bge-reranker-base"


def _cloudflare_credentials(config: dict) -> tuple[str, str]:
    """Resolve Workers AI credentials only from the active profile secret scope."""
    from agent.secret_scope import get_secret

    if config.get("api_key"):
        raise ValueError("Cloudflare Workers AI token must come from the profile secret scope, not mem0.json")
    account_id = str(config.get("account_id") or get_secret("CLOUDFLARE_ACCOUNT_ID") or "")
    api_key = str(get_secret("CLOUDFLARE_WORKERS_AI_TOKEN") or "")
    if not account_id or not api_key:
        raise ValueError("Cloudflare Workers AI account ID and token are required")
    return account_id, api_key


def _cloudflare_embedder_block(block: dict) -> dict:
    """Use Mem0's OpenAI embedder against Cloudflare's compatible endpoint."""
    provider_config = dict(block.get("config", {}))
    account_id, api_key = _cloudflare_credentials(provider_config)
    provider_config.update(
        {
            "api_key": api_key,
            "model": provider_config.get("model") or _CLOUDFLARE_EMBED_MODEL,
            "embedding_dims": int(provider_config.get("embedding_dims") or 1024),
            "openai_base_url": f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        }
    )
    provider_config.pop("account_id", None)
    return {"provider": "openai", "config": provider_config}


def _register_direct_openai_provider() -> None:
    """Register Hermes' OpenAI-only Mem0 LLM provider once per factory."""
    from mem0.configs.llms.openai import OpenAIConfig
    from mem0.utils.factory import LlmFactory
    provider_map = getattr(LlmFactory, "provider_to_class", None)
    register_provider = getattr(LlmFactory, "register_provider", None)
    if not isinstance(provider_map, dict) or not callable(register_provider):
        raise RuntimeError("mem0 LlmFactory does not support the provider registration required for the Hermes OpenAI OSS backend")
    if provider_map.get(_DIRECT_OPENAI_PROVIDER) != (_DIRECT_OPENAI_CLASS_PATH, OpenAIConfig):
        register_provider(_DIRECT_OPENAI_PROVIDER, _DIRECT_OPENAI_CLASS_PATH, OpenAIConfig)


class OSSBackend(Mem0Backend):
    """Wraps mem0.Memory for self-hosted (OSS) mode."""

    def __init__(self, oss_config: dict):
        import os
        from mem0 import Memory
        from ._oss_providers import EMBEDDER_PROVIDERS, KNOWN_DIMS, LLM_PROVIDERS

        def _provider_block(name: str, registry: dict) -> dict:
            """Copy of oss_config[name] with the legacy ``api_base`` key mapped to the provider's canonical base-URL key."""
            block = dict(oss_config[name])
            provider_config = dict(block.get("config", {}))
            legacy_base = provider_config.pop("api_base", None)
            canonical_key = registry.get(str(block.get("provider") or "").strip().lower(), {}).get("base_url_key")
            if legacy_base and canonical_key:
                provider_config.setdefault(canonical_key, legacy_base)
            block["config"] = provider_config
            return block

        vector_store = dict(oss_config["vector_store"])
        vs_config = dict(vector_store.get("config", {}))
        if "path" in vs_config:
            vs_config["path"] = os.path.expanduser(vs_config["path"])
        raw_embedder = _provider_block("embedder", EMBEDDER_PROVIDERS)
        embedder = _cloudflare_embedder_block(raw_embedder) if str(raw_embedder.get("provider") or "").strip().lower() == "cloudflare" else raw_embedder
        embedder_config = embedder.get("config", {})
        dims = embedder_config.get("embedding_dims") or KNOWN_DIMS.get(embedder_config.get("model", ""))
        if dims:
            vs_config["embedding_model_dims"] = dims
            self._recreate_collection_if_dims_changed(vector_store.get("provider", "qdrant"), vs_config, dims)
        vector_store["config"] = vs_config
        _widen_qdrant_timeout(vector_store)
        config = {"vector_store": vector_store, "llm": _provider_block("llm", LLM_PROVIDERS), "embedder": embedder, "version": "v1.1"}
        if oss_config.get("history_db_path"):
            config["history_db_path"] = os.path.expanduser(str(oss_config["history_db_path"]))
        self._reranker = None
        reranker_config = dict(oss_config.get("reranker") or {})
        if str(reranker_config.get("provider") or "").strip().lower() == "cloudflare":
            from ._cloudflare import CloudflareReranker

            cf_config = dict(reranker_config.get("config") or {})
            account_id, api_key = _cloudflare_credentials(cf_config)
            self._reranker = CloudflareReranker(
                account_id=account_id,
                api_key=api_key,
                model=cf_config.get("model") or _CLOUDFLARE_RERANK_MODEL,
                timeout=float(cf_config.get("timeout") or 30.0),
            )
        if str(config["llm"].get("provider") or "").strip().lower() == "openai":
            # mem0 validates LlmConfig.provider before its factory lookup: build the supported OpenAI config, then swap the provider.
            _register_direct_openai_provider()
            from mem0.configs.base import MemoryConfig
            memory_config = MemoryConfig(**config)
            try:
                memory_config.llm.provider = _DIRECT_OPENAI_PROVIDER
            except (AttributeError, TypeError) as exc:
                raise RuntimeError("mem0 MemoryConfig does not expose a mutable llm.provider for the Hermes OpenAI OSS backend") from exc
            self._memory = Memory(memory_config)
        else:
            self._memory = Memory.from_config(config)

    @staticmethod
    def _recreate_collection_if_dims_changed(provider: str, vs_config: dict, expected_dims: int) -> None:
        """Delete stale vector collection when embedding dimensions change."""
        collection_name = vs_config.get("collection_name", "mem0")
        with suppress(Exception):
            if provider == "qdrant":
                from qdrant_client import QdrantClient
                path, url = vs_config.get("path"), vs_config.get("url")
                if path:
                    client = QdrantClient(path=path)
                elif url:
                    client = QdrantClient(url=url, api_key=vs_config.get("api_key"))
                else:
                    return
                with closing(client):
                    if not client.collection_exists(collection_name):
                        return
                    vectors = client.get_collection(collection_name).config.params.vectors
                    # Named-vector collections expose a dict; unnamed expose an object with .size.
                    if isinstance(vectors, dict):
                        vectors = next(iter(vectors.values()), None)
                    current_dims = getattr(vectors, "size", None)
                    if current_dims is not None and current_dims != expected_dims:
                        client.delete_collection(collection_name)
            elif provider == "pgvector":
                import psycopg2
                from psycopg2 import sql as pgsql
                conn_params = {k: vs_config[k] for k in ("host", "port", "user", "password", "dbname", "sslmode") if vs_config.get(k)}
                with closing(psycopg2.connect(**conn_params)) as conn:
                    conn.autocommit = True
                    with closing(conn.cursor()) as cur:
                        cur.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'vector'", (collection_name,))
                        row = cur.fetchone()
                        if row and row[0] > 0 and row[0] != expected_dims:
                            cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(pgsql.Identifier(collection_name)))

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        candidate_count = max(top_k * 5, 50) if rerank and self._reranker else top_k
        results = _unwrap_results(self._memory.search(query, filters=filters, top_k=candidate_count))
        if not (rerank and self._reranker and results):
            return results[:top_k]
        try:
            return self._reranker.rerank(query, results, top_k)
        except Exception:
            return results[:top_k]

    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict:
        return self._memory.add(messages, **_add_kwargs(user_id, agent_id, infer, metadata))

    def add_idempotent(self, operation_id: str, payload: dict) -> None:
        """Replay explicit journal entries idempotently into Qdrant and history."""
        import hashlib
        import uuid
        from datetime import datetime, timezone

        point_id = str(uuid.UUID(str(operation_id)))
        text = payload["content"]
        user_id = payload.get("user_id")
        agent_id = payload.get("agent_id")
        now_iso = payload.get("observed_at") or datetime.now(timezone.utc).isoformat()
        vs = getattr(self._memory, "vector_store", None)
        if vs is None:
            raise NotImplementedError("Vector store is not configured; cannot replay")
        db = getattr(self._memory, "db", None)
        if db is None or not (hasattr(db, "add_history") and hasattr(db, "get_history")):
            raise RuntimeError("History database is required for durable replay")
        existing_points = []
        if hasattr(vs, "get") and callable(vs.get):
            existing = vs.get(point_id)
            if existing:
                existing_points = [existing]
        elif hasattr(vs, "retrieve") and callable(vs.retrieve):
            existing_points = vs.retrieve(collection_name=getattr(vs, "collection_name", "mem0"), ids=[point_id], with_payload=True)
        elif hasattr(vs, "client") and hasattr(vs.client, "retrieve") and callable(vs.client.retrieve):
            existing_points = vs.client.retrieve(collection_name=getattr(vs, "collection_name", "mem0"), ids=[point_id], with_payload=True)
        else:
            raise NotImplementedError("Vector store does not support point retrieval by ID; cannot verify idempotent replay")
        if existing_points:
            point = existing_points[0]
            existing_payload: Any = getattr(point, "payload", None)
            if existing_payload is None and isinstance(point, dict):
                existing_payload = point.get("payload") if "payload" in point else point
            if not isinstance(existing_payload, dict):
                raise ValueError(f"Existing point for operation_id {operation_id} has invalid or missing payload")
            existing_data = existing_payload.get("data") or existing_payload.get("content")
            if not existing_data:
                raise ValueError(f"Existing point for operation_id {operation_id} is missing memory text in payload")
            if existing_data != text or any(existing_payload.get(field) != payload.get(field) for field in ("user_id", "agent_id", "channel")):
                raise ValueError(f"Conflicting payload for existing operation_id {operation_id}")
            if not db.get_history(memory_id=point_id):
                db.add_history(point_id, None, text, "ADD", created_at=existing_payload.get("created_at") or now_iso, updated_at=existing_payload.get("updated_at") or now_iso, actor_id=existing_payload.get("actor_id") or user_id, role="user")
            return
        try:
            from mem0.memory.main import lemmatize_for_bm25
            lemmatized = lemmatize_for_bm25(text)
        except Exception:
            lemmatized = text
        vector = self._memory.embedding_model.embed(text, memory_action="add")
        metadata = {key: payload[key] for key in ("user_id", "agent_id", "channel") if key in payload}
        metadata.update(data=text, hash=hashlib.md5(text.encode()).hexdigest(), role="user", origin_type=payload.get("origin_type", "agent_write"), verification_status="unverified", observed_at=payload.get("observed_at"), pending_operation_id=str(operation_id), created_at=now_iso, updated_at=now_iso, text_lemmatized=lemmatized)
        vs.insert(ids=[point_id], vectors=[vector], payloads=[metadata])
        if not db.get_history(memory_id=point_id):
            db.add_history(point_id, None, text, "ADD", created_at=metadata.get("created_at"), updated_at=metadata.get("updated_at"), actor_id=metadata.get("actor_id") or user_id, role="user")

    def _update(self, memory_id: str, text: str) -> None:
        self._memory.update(memory_id, data=text)

    def _delete(self, memory_id: str) -> None:
        self._memory.delete(memory_id)

    def close(self):
        # Mem0's vector client and history DB can be shared inside a multiplexed
        # gateway. Closing either during one agent teardown breaks later writes.
        with suppress(Exception):
            telemetry = getattr(self._memory, "telemetry", None)
            if telemetry and hasattr(telemetry, "posthog"):
                telemetry.posthog.shutdown()
        if self._reranker and hasattr(self._reranker, "close"):
            with suppress(Exception):
                self._reranker.close()
