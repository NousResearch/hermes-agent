"""Tests for hermes_cli.runtime_provider._detect_api_mode_for_url.

The helper maps base URLs to api_modes for four cases:
  * api.openai.com    → codex_responses
  * api.x.ai          → codex_responses
  * api.anthropic.com → anthropic_messages (Pro/Max OAuth is only billed
                                            against /v1/messages; the
                                            chat_completions shim counts
                                            against a separate empty
                                            "extra usage" pool, see #32243)
  * */anthropic       → anthropic_messages (third-party gateways like MiniMax,
                                            Zhipu GLM, LiteLLM proxies)

Consolidating the /anthropic detection in this helper (instead of three
inline ``endswith`` checks spread across _resolve_runtime_from_pool_entry,
the explicit-provider path, and the api-key-provider path) means every
future update to the detection logic lives in one place.
"""

from __future__ import annotations

from hermes_cli.runtime_provider import _detect_api_mode_for_url


class TestCodexResponsesDetection:
    def test_openai_api_returns_codex_responses(self):
        assert _detect_api_mode_for_url("https://api.openai.com/v1") == "codex_responses"


class TestDirectAnthropicHost:
    """Native api.anthropic.com → /v1/messages. Pinned for issue #32243.

    The Anthropic OpenAI-compat ``/chat/completions`` shim on the same
    host bills against a separate "extra usage" pool that Pro/Max OAuth
    subscriptions don't fund, so a fresh OAuth credential 400s with
    "out of extra usage" the moment a request lands there. The detector
    must keep ``api.anthropic.com`` on the native Messages API.
    """


    def test_lookalike_subdomain_does_not_match(self):
        # ``api.anthropic.com.attacker.test`` is an attacker-controlled
        # host; the registrable label is ``attacker``, not Anthropic.
        # Must NOT be routed to anthropic_messages — leaking an
        # Anthropic OAuth token there is the worst case.
        assert (
            _detect_api_mode_for_url("https://api.anthropic.com.attacker.test/v1")
            is None
        )


class TestAnthropicMessagesDetection:
    """Third-party gateways that speak the Anthropic protocol under /anthropic."""


class TestDefaultCase:


    def test_localhost_returns_none(self):
        assert _detect_api_mode_for_url("http://localhost:11434/v1") is None
