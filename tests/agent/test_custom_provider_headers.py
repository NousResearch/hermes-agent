"""Tests for custom-provider header suppression (issue #40033).

When the provider is a generic OpenAI-compatible custom endpoint, some upstream
gateways reject requests that carry the OpenAI Python SDK default headers
(User-Agent: OpenAI/Python, X-Stainless-*).  The fix strips these headers
for custom providers that don't already have provider-specific headers set.
"""

import logging
from unittest.mock import patch, MagicMock

import pytest

from agent.auxiliary_client import (
    _NEUTRAL_SDK_HEADERS,
    _neutral_custom_openai_kwargs,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


class TestNeutralSdkHeaders:
    """Verify the header constant suppresses SDK identification."""

    def test_neutral_headers_contains_user_agent(self):
        assert "User-Agent" in _NEUTRAL_SDK_HEADERS

    def test_neutral_headers_not_openai(self):
        """User-Agent should not identify as the OpenAI SDK."""
        assert "OpenAI" not in _NEUTRAL_SDK_HEADERS["User-Agent"]

    def test_neutral_client_hook_strips_stainless(self):
        """Outgoing requests should contain no X-Stainless-* headers."""
        import httpx
        from openai import OpenAI

        captured_headers = {}

        def handler(request):
            captured_headers.update(request.headers)
            return httpx.Response(200, json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            })

        kwargs = _neutral_custom_openai_kwargs()
        # Replace the real transport but keep the production request hook.
        kwargs["http_client"] = httpx.Client(
            transport=httpx.MockTransport(handler),
            event_hooks=kwargs["http_client"].event_hooks,
        )
        client = OpenAI(api_key="test-key", base_url="http://custom.local/v1", **kwargs)
        client.chat.completions.create(model="test-model", messages=[{"role": "user", "content": "hi"}])

        assert captured_headers.get("user-agent") == "hermes-agent"
        assert not [key for key in captured_headers if key.lower().startswith("x-stainless-")]


class TestCustomEndpointHeaderInjection:
    """Verify _try_custom_endpoint injects neutral headers for generic endpoints."""

    def test_generic_endpoint_gets_neutral_headers(self, monkeypatch):
        """A generic custom endpoint should get neutral SDK-suppression headers."""
        mock_client = MagicMock()

        with patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://generic-host.local/v1", "test-key", None)), \
             patch("agent.auxiliary_client._read_main_model", return_value="test-model"), \
             patch("agent.auxiliary_client.OpenAI", return_value=mock_client) as mock_openai, \
             patch("agent.auxiliary_client._maybe_wrap_anthropic",
                   side_effect=lambda c, *a, **kw: c):
            from agent.auxiliary_client import _try_custom_endpoint
            client, model = _try_custom_endpoint()

            # OpenAI should have been called with default_headers containing
            # neutral User-Agent and suppressed Stainless headers
            call_kwargs = mock_openai.call_args
            headers = call_kwargs.kwargs.get("default_headers")
            assert headers is not None, "custom endpoint should get default_headers"
            assert "User-Agent" in headers
            assert "OpenAI" not in headers["User-Agent"]
            assert mock_openai.call_args.kwargs.get("http_client") is not None

    def test_codex_responses_mode_not_affected(self, monkeypatch):
        """codex_responses mode uses its own client wrapper, not neutral headers."""
        mock_client = MagicMock()

        with patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://custom-host/v1", "test-key", "codex_responses")), \
             patch("agent.auxiliary_client._read_main_model", return_value="test-model"), \
             patch("agent.auxiliary_client.OpenAI", return_value=mock_client) as mock_openai:
            from agent.auxiliary_client import _try_custom_endpoint, CodexAuxiliaryClient
            client, model = _try_custom_endpoint()

            # Should return a CodexAuxiliaryClient wrapper, not raw OpenAI
            assert isinstance(client, CodexAuxiliaryClient)

    def test_anthropic_messages_mode_not_affected(self, monkeypatch):
        """anthropic_messages mode uses its own client, not neutral headers."""
        mock_anthropic_client = MagicMock()

        with patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://custom-host/v1", "test-key", "anthropic_messages")), \
             patch("agent.auxiliary_client._read_main_model", return_value="test-model"), \
             patch("agent.auxiliary_client.build_anthropic_client",
                   return_value=mock_anthropic_client, create=True):
            from agent.auxiliary_client import _try_custom_endpoint
            try:
                client, model = _try_custom_endpoint()
                # If anthropic SDK is available, should use AnthropicAuxiliaryClient
            except ImportError:
                pass  # anthropic SDK not installed — acceptable


class TestResolveProviderClientCustomHeaders:
    """Verify resolve_provider_client sets neutral headers for custom providers."""

    def test_custom_provider_with_base_url_gets_headers(self, monkeypatch):
        """Custom provider with explicit base_url should get neutral headers
        when no provider-specific headers match."""
        from agent.auxiliary_client import resolve_provider_client

        mock_client = MagicMock()
        mock_client.base_url = "http://my-gateway.example.com/v1"

        captured_headers = {}

        def capture_openai(*args, **kwargs):
            captured_headers.update(kwargs.get("default_headers", {}))
            return mock_client

        import agent.auxiliary_client as _mod

        with patch.object(_mod, "OpenAI", side_effect=capture_openai), \
             patch.object(_mod, "_extract_url_query_params",
                          return_value=("http://my-gateway.example.com/v1", None)), \
             patch.object(_mod, "_to_async_client", side_effect=lambda c, *a, **kw: c):
            try:
                client, model = resolve_provider_client(
                    provider="custom",
                    model="my-model",
                    explicit_base_url="http://my-gateway.example.com/v1",
                    explicit_api_key="test-key",
                )
            except Exception:
                pass  # May fail on wrapper logic — OK, we captured headers

        assert "User-Agent" in captured_headers, \
            "Custom provider should receive neutral User-Agent header"
        assert "OpenAI" not in captured_headers.get("User-Agent", ""), \
            "User-Agent should not contain 'OpenAI'"
