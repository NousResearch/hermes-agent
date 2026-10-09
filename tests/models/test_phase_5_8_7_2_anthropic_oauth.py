"""Anthropic source-contract regression: OAuth catalogue headers and retry are plugin-owned."""
from __future__ import annotations

import io
import urllib.error

from providers import get_provider_profile


def test_oauth_long_context_subscription_retry_preserves_pagination(monkeypatch):
    # The normal common-beta set does not request 1M context. Simulate a
    # catalogue call that actually sent the beta before checking its removal.
    import agent.anthropic_adapter as adapter

    monkeypatch.setattr(
        adapter, "_COMMON_BETAS", [*adapter._COMMON_BETAS, adapter._CONTEXT_1M_BETA]
    )
    profile = get_provider_profile("anthropic")
    assert profile is not None
    requests = []

    def request(url, *, timeout, headers):
        requests.append((url, dict(headers)))
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                url, 400, "unsupported beta", None,
                io.BytesIO(b"long context beta is not yet available for this subscription"),
            )
        if len(requests) == 2:
            return {"data": [{"id": "claude-a"}], "has_more": True, "last_id": "claude-a"}
        return {"data": [{"id": "claude-b"}], "has_more": False}

    models = profile.fetch_catalog_models(
        api_key="oauth-token", oauth=True, request_json=request, timeout=1.0
    )
    assert models == ["claude-a", "claude-b"]
    assert len(requests) == 3
    assert requests[0][1]["Authorization"] == "Bearer oauth-token"
    assert "x-api-key" not in requests[0][1]
    from agent.anthropic_adapter import _COMMON_BETAS, _CONTEXT_1M_BETA
    if _CONTEXT_1M_BETA in _COMMON_BETAS:
        assert requests[0][1]["anthropic-beta"] != requests[1][1]["anthropic-beta"]
        assert _CONTEXT_1M_BETA not in requests[1][1]["anthropic-beta"]
    assert "after_id=claude-a" in requests[2][0]


def test_unexpected_oauth_error_is_not_retried():
    profile = get_provider_profile("anthropic")
    attempts = []

    def request(url, *, timeout, headers):
        attempts.append(url)
        raise urllib.error.HTTPError(url, 400, "invalid", None, io.BytesIO(b"unrelated error"))

    assert profile.fetch_catalog_models(
        api_key="oauth-token", oauth=True, request_json=request,
    ) is None
    assert len(attempts) == 1
