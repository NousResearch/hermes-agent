"""Cloudflare Workers AI reranking for Mem0 OSS recall."""

from __future__ import annotations

from typing import Any

import httpx

_RERANK_MODEL = "@cf/baai/bge-reranker-base"


class CloudflareReranker:
    """Rerank Mem0 result dictionaries without changing their payloads."""

    def __init__(
        self,
        *,
        account_id: str,
        api_key: str,
        model: str = _RERANK_MODEL,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not account_id or not api_key:
            raise ValueError("Cloudflare Workers AI account ID and token are required")
        self.model = model
        self._client = httpx.Client(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            transport=transport or httpx.HTTPTransport(retries=2),
        )

    def rerank(self, query: str, memories: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if not memories or limit <= 0:
            return []
        documents = [str(row.get("memory") or row.get("text") or "") for row in memories]
        response = self._client.post(
            f"/{self.model}",
            json={"query": query, "contexts": [{"text": document} for document in documents], "top_k": len(documents)},
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            errors = body.get("errors") or []
            raise RuntimeError(f"Cloudflare rerank failed: {errors}")
        ranked = body.get("result", {}).get("response") or body.get("result", {}).get("data") or []
        rerank_scores = {
            index: float(item["score"])
            for item in ranked
            if isinstance((index := item.get("id", item.get("index"))), int)
            and 0 <= index < len(memories)
            and isinstance(item.get("score"), (int, float))
        }
        # Cloudflare's cross-encoder can return near-ties at zero for broad or
        # cross-language prompts. Blend it with Mem0's semantic score rather
        # than letting an arbitrary zero-score order destroy a strong match.
        combined: list[tuple[float, int, dict[str, Any]]] = []
        for index, original in enumerate(memories):
            row = dict(original)
            semantic_score = float(row.get("score") or 0.0)
            rerank_score = rerank_scores.get(index, 0.0)
            score = 0.7 * semantic_score + 0.3 * rerank_score
            row["score"] = score
            combined.append((score, index, row))
        combined.sort(key=lambda item: (-item[0], item[1]))
        return [row for _, _, row in combined[:limit]]

    def close(self) -> None:
        self._client.close()
