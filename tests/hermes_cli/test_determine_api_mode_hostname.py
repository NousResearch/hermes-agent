"""Regression tests for canonical invocation-route hostname handling."""

from __future__ import annotations

from providers.routing import InvocationRequest, resolve_invocation_route


def _mode(provider: str = "", base_url: str = "", model: str = "") -> str:
    return resolve_invocation_route(
        InvocationRequest(provider=provider, base_url=base_url, model=model)
    ).api_mode


class TestOpenAIHostHardening:
    def test_native_openai_url_is_codex_responses(self):
        assert _mode("", "https://api.openai.com/v1") == "codex_responses"


class TestAnthropicHostHardening:
    def test_anthropic_path_suffix_still_wins(self):
        assert _mode("", "https://api.minimax.io/anthropic") == "anthropic_messages"
