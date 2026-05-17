from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


class PineconeMemoryClient:
    """Fail-open wrapper around the Pinecone index client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
        top_k: int = 5,
        fail_open: bool = True,
        sdk_client: Any | None = None,
        index: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PINECONE_API_KEY", "")
        self.index_name = index_name or os.getenv("PINECONE_INDEX", "")
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE", "hermes")
        self.top_k = top_k
        self.fail_open = fail_open
        self._sdk_client = sdk_client
        self._index = index

    def is_configured(self) -> bool:
        return bool(self.api_key and self.index_name)

    def upsert_records(self, records: Iterable[dict[str, Any]]) -> int:
        normalized = [self._normalize_record(record) for record in records]
        if not normalized:
            return 0
        if not self.is_configured():
            logger.warning("Pinecone upsert skipped: missing configuration")
            return 0
        try:
            index = self._get_index()
            response = index.upsert(vectors=normalized, namespace=self.namespace)
            if isinstance(response, dict):
                return int(response.get("upserted_count", len(normalized)))
            return int(getattr(response, "upserted_count", len(normalized)))
        except Exception as exc:
            if self.fail_open:
                logger.warning("Pinecone upsert failed; continuing fail-open: %s", exc)
                return 0
            raise

    def query(self, vector: Sequence[float], *, top_k: int | None = None, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self.is_configured():
            logger.warning("Pinecone query skipped: missing configuration")
            return []
        try:
            index = self._get_index()
            response = index.query(
                vector=list(vector),
                top_k=top_k or self.top_k,
                namespace=self.namespace,
                include_metadata=True,
                filter=filter,
            )
            matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
            return [self._normalize_match(match) for match in matches]
        except Exception as exc:
            if self.fail_open:
                logger.warning("Pinecone query failed; continuing fail-open: %s", exc)
                return []
            raise

    def delete_by_source(self, *, source_kind: str, source_id: str) -> int:
        if not self.is_configured():
            logger.warning("Pinecone delete skipped: missing configuration")
            return 0
        try:
            index = self._get_index()
            index.delete(
                namespace=self.namespace,
                filter={"source_kind": {"$eq": source_kind}, "source_id": {"$eq": source_id}},
            )
            return 1
        except Exception as exc:
            if self.fail_open:
                logger.warning("Pinecone delete failed; continuing fail-open: %s", exc)
                return 0
            raise

    def _get_index(self) -> Any:
        if self._index is not None:
            return self._index
        if self._sdk_client is None:
            self._sdk_client = self._build_sdk_client()
        index_factory = getattr(self._sdk_client, "Index")
        self._index = index_factory(self.index_name)
        return self._index

    def _build_sdk_client(self) -> Any:
        from pinecone import Pinecone  # type: ignore

        return Pinecone(api_key=self.api_key)

    @staticmethod
    def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        if "id" not in record or "values" not in record or "metadata" not in record:
            raise ValueError("Pinecone record must include id, values, and metadata")
        return {
            "id": str(record["id"]),
            "values": [float(value) for value in record["values"]],
            "metadata": dict(record["metadata"]),
        }

    @staticmethod
    def _normalize_match(match: Any) -> dict[str, Any]:
        if isinstance(match, dict):
            return {
                "id": match.get("id", ""),
                "score": match.get("score"),
                "metadata": dict(match.get("metadata") or {}),
            }
        return {
            "id": getattr(match, "id", ""),
            "score": getattr(match, "score", None),
            "metadata": dict(getattr(match, "metadata", {}) or {}),
        }


__all__ = ["PineconeMemoryClient"]
