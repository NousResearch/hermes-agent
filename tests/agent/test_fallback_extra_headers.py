"""Unit tests for fallback client extra_headers preservation (#87098)."""

from unittest.mock import MagicMock, patch
from agent.auxiliary_client import _resolve_fallback_entry


def test_resolve_fallback_entry_merges_extra_headers_and_coerces_str():
    """Verify that _resolve_fallback_entry merges extra_headers into client.default_headers
    and coerces values to str."""
    fake_client = MagicMock()
    fake_client.default_headers = {"X-Existing": "base"}

    entry = {
        "provider": "openrouter",
        "model": "deepseek-chat",
        "extra_headers": {
            "HTTP-Referer": "https://hermes.ai",
            "X-Title": "Hermes",
            "X-Rate-Limit": 100,  # int should be coerced to str
        },
    }

    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fake_client, "deepseek-chat")):
        client, model = _resolve_fallback_entry(entry)

    assert client is fake_client
    assert model == "deepseek-chat"
    assert client.default_headers["X-Existing"] == "base"
    assert client.default_headers["HTTP-Referer"] == "https://hermes.ai"
    assert client.default_headers["X-Title"] == "Hermes"
    assert client.default_headers["X-Rate-Limit"] == "100"
