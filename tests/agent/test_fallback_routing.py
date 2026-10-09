"""Fallback route facts are acquired in agent code and interpreted by providers.routing."""

from unittest.mock import patch

from agent.fallback_routing import resolve_fallback_invocation_route


def test_explicit_fallback_mode_wins_over_endpoint_mandate():
    route = resolve_fallback_invocation_route(
        "custom", "gpt-5.2", "https://api.openai.com/v1",
        explicit_api_mode="chat_completions",
    )
    assert route.provider == "custom"
    assert route.api_mode == "chat_completions"


def test_named_custom_inherits_configured_base_and_wire():
    configured = {
        "name": "ai-proxy",
        "base_url": "https://proxy.example/v1",
        "api_mode": "anthropic_messages",
    }
    with patch("agent.configured_provider_resolution.get_configured_provider_entry", return_value=configured):
        route = resolve_fallback_invocation_route("custom:ai-proxy", "claude-opus-4-6")
    assert route.provider == "custom:ai-proxy"
    assert route.base_url == "https://proxy.example/v1"
    assert route.api_mode == "anthropic_messages"


def test_opencode_model_route_overrides_stale_configured_wire():
    configured = {
        "name": "go-bridge",
        "base_url": "https://opencode.ai/zen/go/v1",
        "api_mode": "chat_completions",
    }
    with (
        patch("agent.configured_provider_resolution.get_configured_provider_entry", return_value=configured),
        patch(
            "agent.fallback_routing.opencode_transport",
            return_value=("codex_responses", "https://opencode.ai/zen/go/v1"),
        ),
    ):
        route = resolve_fallback_invocation_route("custom:go-bridge", "muse-spark-1.3-contributor")
    assert route.api_mode == "codex_responses"
def test_original_endpoint_hint_preserves_wire_after_client_url_rewrite():
    route = resolve_fallback_invocation_route(
        "custom",
        "MiniMax-M2.5",
        "https://api.minimax.io/v1",
        route_base_url_hint="https://api.minimax.io/anthropic",
    )
    assert route.base_url == "https://api.minimax.io/v1"
    assert route.api_mode == "anthropic_messages"