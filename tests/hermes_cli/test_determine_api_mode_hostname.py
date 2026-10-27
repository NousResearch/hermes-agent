"""Regression tests for ``determine_api_mode`` hostname handling.

Companion to tests/hermes_cli/test_detect_api_mode_for_url.py — the same
false-positive class (custom URLs containing ``api.openai.com`` /
``api.anthropic.com`` as a path segment or host suffix) must be rejected
by ``determine_api_mode`` as well, since it's the code path used by
custom/unknown providers in ``resolve_custom_provider``.
"""

from __future__ import annotations

from hermes_cli.providers import determine_api_mode


class TestOpenAIHostHardening:
    def test_native_openai_url_is_codex_responses(self):
        assert determine_api_mode("", "https://api.openai.com/v1") == "codex_responses"


class TestAnthropicHostHardening:


    def test_anthropic_path_suffix_still_wins(self):
        # Third-party Anthropic-compatible gateways (MiniMax, Zhipu GLM, LiteLLM
        # proxies) expose the Anthropic protocol under a ``/anthropic`` suffix.
        # That convention must still resolve to anthropic_messages.
        assert determine_api_mode("", "https://api.minimax.io/anthropic") == "anthropic_messages"
