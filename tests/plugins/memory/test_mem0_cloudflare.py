"""Tests for Cloudflare Workers AI reranking."""

import httpx
import pytest

from plugins.memory.mem0._cloudflare import CloudflareReranker


def test_reranker_preserves_payload_and_uses_cloudflare_order():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer token"
        assert request.url.path.endswith("/@cf/baai/bge-reranker-base")
        body = __import__("json").loads(request.content)
        assert body["query"] == "bagage"
        assert body["contexts"] == [{"text": "rode tas"}, {"text": "paarse koffer"}]
        return httpx.Response(200, json={"success": True, "result": {"response": [{"id": 1, "score": 0.99}, {"id": 0, "score": 0.2}]}})

    reranker = CloudflareReranker(
        account_id="account",
        api_key="token",
        transport=httpx.MockTransport(handler),
    )
    rows = [
        {"id": "a", "memory": "rode tas", "score": 0.1, "metadata": {"source": "x"}},
        {"id": "b", "memory": "paarse koffer", "score": 0.2, "metadata": {"source": "y"}},
    ]
    result = reranker.rerank("bagage", rows, 2)
    assert [row["id"] for row in result] == ["b", "a"]
    assert result[0]["metadata"] == {"source": "y"}
    assert result[0]["score"] == pytest.approx(0.7 * 0.2 + 0.3 * 0.99)
    assert rows[1]["score"] == 0.2


def test_reranker_rejects_failed_cloudflare_envelope():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"success": False, "errors": [{"message": "no"}]}))
    reranker = CloudflareReranker(account_id="account", api_key="token", transport=transport)
    with pytest.raises(RuntimeError, match="Cloudflare rerank failed"):
        reranker.rerank("q", [{"memory": "m"}], 1)
