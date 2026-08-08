"""Tests that an explicit ``api_mode`` on a fallback chain entry is honored.

The primary model path already respects a user-declared ``api_mode`` (via
``hermes_cli.runtime_provider._parse_api_mode``).  Fallback activation used to
ignore it and re-derive the wire protocol from provider name / base URL / model
instead, so a chain entry could silently come up on a different protocol than
the one the user configured.

The concrete failure this guards against: a self-hosted OpenAI-compatible
gateway fronting Claude, declared as

    fallback_providers:
      - provider: custom
        model: claude-opus-4
        base_url: http://gateway.internal:3000/v1
        api_mode: anthropic_messages

matches none of the anthropic detection branches (provider is not
``anthropic``, the URL neither ends in ``/anthropic`` nor resolves to
``api.anthropic.com``), so inference fell through to ``chat_completions``.
Requests then went out over ``/chat/completions``, which drops the
``cache_control`` blocks the Anthropic path attaches — zeroing prompt caching
for the remainder of the conversation with no error surfaced anywhere.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────

def _make_agent(provider="custom", model="primary-model",
                base_url="http://gateway.internal:3000/v1",
                api_mode="chat_completions"):
    """Minimal AIAgent-like stub carrying only the fields fallback touches."""
    agent = MagicMock()
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.api_mode = api_mode
    agent.api_key = "primary-key"
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._unavailable_fallback_keys = set()
    agent._primary_runtime = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_mode": api_mode,
        "api_key": "primary-key",
        "client_kwargs": {"api_key": "primary-key", "base_url": base_url},
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "anthropic_api_key": "",
        "anthropic_base_url": "",
    }
    agent._config_context_length = None
    agent._credential_pool = None
    agent._rate_limited_until = 0
    agent._transport_cache = {}
    agent._client_kwargs = {"api_key": "primary-key", "base_url": base_url}
    agent._buffer_status = MagicMock()
    agent._is_azure_openai_url.return_value = False
    agent._is_direct_openai_url.return_value = False
    agent._provider_model_requires_responses_api.return_value = False
    agent._anthropic_prompt_cache_policy.return_value = (True, True)
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    agent._replace_primary_openai_client = MagicMock()
    agent.context_compressor = None
    return agent


def _activate(agent, fb_base_url="http://gateway.internal:3000/v1"):
    """Drive the real fallback activator with network calls stubbed out."""
    from agent.chat_completion_helpers import try_activate_fallback

    fallback_client = SimpleNamespace(
        api_key="gateway-key",
        base_url=fb_base_url,
        _custom_headers={},
    )
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fallback_client, agent._fallback_chain[0]["model"]),
    ), patch(
        "agent.credential_pool.load_pool",
        return_value=None,
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=MagicMock(),
    ):
        return try_activate_fallback(agent)


# ── Tests ────────────────────────────────────────────────────────────

class TestFallbackExplicitApiMode:
    """An explicit chain-entry ``api_mode`` must win over inference."""

    def test_explicit_anthropic_messages_on_custom_gateway(self):
        """provider=custom + non-anthropic URL still honors anthropic_messages."""
        agent = _make_agent()
        agent._fallback_chain = [{
            "provider": "custom",
            "model": "claude-opus-4",
            "base_url": "http://gateway.internal:3000/v1",
            "api_mode": "anthropic_messages",
        }]

        assert _activate(agent) is True
        assert agent.api_mode == "anthropic_messages", (
            "Explicit api_mode from config was overridden by inference — "
            "requests would go out over /chat/completions and lose prompt caching"
        )

    def test_explicit_mode_builds_native_anthropic_client(self):
        """Honoring the mode must also wire up the native Anthropic client."""
        agent = _make_agent()
        agent._fallback_chain = [{
            "provider": "custom",
            "model": "claude-opus-4",
            "base_url": "http://gateway.internal:3000/v1",
            "api_mode": "anthropic_messages",
        }]

        assert _activate(agent) is True
        assert agent._anthropic_client is not None
        assert agent._anthropic_base_url == "http://gateway.internal:3000/v1"

    def test_explicit_chat_completions_overrides_anthropic_host(self):
        """The override works in both directions, not just toward anthropic."""
        agent = _make_agent()
        agent._fallback_chain = [{
            "provider": "anthropic",
            "model": "claude-opus-4",
            "base_url": "https://api.anthropic.com",
            "api_mode": "chat_completions",
        }]

        assert _activate(agent, fb_base_url="https://api.anthropic.com") is True
        assert agent.api_mode == "chat_completions"

    @pytest.mark.parametrize("bogus", ["", "   ", "rest_api", "ANTHROPIC", None, 42])
    def test_invalid_or_absent_mode_falls_back_to_inference(self, bogus):
        """Unparseable values must not short-circuit the inference chain."""
        agent = _make_agent()
        entry = {
            "provider": "custom",
            "model": "claude-opus-4",
            "base_url": "http://gateway.internal:3000/v1",
        }
        if bogus is not None:
            entry["api_mode"] = bogus
        agent._fallback_chain = [entry]

        assert _activate(agent) is True
        # Inference path for a plain custom OpenAI-compatible gateway.
        assert agent.api_mode == "chat_completions"

    def test_inference_still_detects_anthropic_host_without_explicit_mode(self):
        """Pre-existing hostname detection (#32243, #49247) must keep working."""
        agent = _make_agent()
        agent._fallback_chain = [{
            "provider": "custom",
            "model": "claude-opus-4",
            "base_url": "https://api.anthropic.com",
        }]

        assert _activate(agent, fb_base_url="https://api.anthropic.com") is True
        assert agent.api_mode == "anthropic_messages"

    def test_case_and_whitespace_are_normalized(self):
        """_parse_api_mode lowercases and strips, so config typos still work."""
        agent = _make_agent()
        agent._fallback_chain = [{
            "provider": "custom",
            "model": "claude-opus-4",
            "base_url": "http://gateway.internal:3000/v1",
            "api_mode": "  Anthropic_Messages  ",
        }]

        assert _activate(agent) is True
        assert agent.api_mode == "anthropic_messages"
