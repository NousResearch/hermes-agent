from __future__ import annotations

import hashlib
import os
from typing import Any, Optional

from agent.auxiliary_client import resolve_provider_client
from hermes_cli.config import load_config

_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_EMBEDDING_PROVIDER = "openrouter"
_CHUNK_ID_PREFIX = "memory-chunk"


class MemoryEmbedder:
    """Thin embedding wrapper for memory ingestion and retrieval.

    Provider/model resolution comes from explicit constructor overrides first,
    then ``config.yaml`` ``memory.embedding_*`` keys when present, and finally
    the ``MEMORY_EMBEDDING_*`` environment variables.
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        client: Any = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._config = config if config is not None else load_config()
        memory_cfg = self._config.get("memory", {}) if isinstance(self._config, dict) else {}
        self.provider = (
            provider
            or (memory_cfg.get("embedding_provider") if isinstance(memory_cfg, dict) else None)
            or os.getenv("MEMORY_EMBEDDING_PROVIDER")
            or _DEFAULT_EMBEDDING_PROVIDER
        )
        requested_model = (
            model
            or (memory_cfg.get("embedding_model") if isinstance(memory_cfg, dict) else None)
            or os.getenv("MEMORY_EMBEDDING_MODEL")
            or _DEFAULT_EMBEDDING_MODEL
        )
        self.client, resolved_model = (client, requested_model)
        if self.client is None:
            self.client, resolved_model = resolve_provider_client(self.provider, requested_model)
        if self.client is None:
            raise RuntimeError(
                f"Unable to initialize memory embedding client for provider {self.provider!r}"
            )
        self.model = resolved_model or requested_model

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def chunk_id_for_text(cls, text: str, *, prefix: str = _CHUNK_ID_PREFIX) -> str:
        return f"{prefix}:{cls.content_hash(text)}"

    def chunk_ids_for_texts(self, texts: list[str], *, prefix: str = _CHUNK_ID_PREFIX) -> list[str]:
        return [self.chunk_id_for_text(text, prefix=prefix) for text in texts]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        if not response.data:
            raise RuntimeError("Embedding provider returned no vectors for query")
        return list(response.data[0].embedding)
