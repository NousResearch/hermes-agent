"""Behavior tests for deterministic per-turn model routing."""

from agent.smart_model_routing import resolve_smart_model_route


RUNTIME = {
    "provider": "openai-codex",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "api_mode": "chat_completions",
    "api_key": "test-key",
}

CONFIG = {
    "enabled": True,
    "platforms": ["discord"],
    "max_simple_chars": 160,
    "max_simple_words": 28,
    "max_balanced_chars": 512,
    "max_balanced_words": 96,
    "cheap_model": {"model": "gpt-5.6-luna"},
    "balanced_model": {"model": "gpt-5.6-terra"},
}


def test_short_plain_gateway_message_uses_cheap_tier():
    route = resolve_smart_model_route(
        "hi",
        model="gpt-5.6-sol",
        runtime=RUNTIME,
        config=CONFIG,
        platform="discord",
    )

    assert route == {"model": "gpt-5.6-luna", "label": "cheap"}


def test_explanation_request_uses_balanced_tier():
    route = resolve_smart_model_route(
        "Explain the practical differences between webhooks and polling.",
        model="gpt-5.6-sol",
        runtime=RUNTIME,
        config=CONFIG,
        platform="discord",
    )

    assert route == {"model": "gpt-5.6-terra", "label": "balanced"}


def test_code_request_stays_on_primary_model():
    route = resolve_smart_model_route(
        "Write a Python script that parses JSONL and reports invalid rows.",
        model="gpt-5.6-sol",
        runtime=RUNTIME,
        config=CONFIG,
        platform="discord",
    )

    assert route == {"model": "gpt-5.6-sol", "label": "primary"}


def test_disabled_or_unconfigured_platform_stays_on_primary_model():
    disabled = {**CONFIG, "enabled": False}
    assert resolve_smart_model_route(
        "hi", model="gpt-5.6-sol", runtime=RUNTIME, config=disabled, platform="discord"
    ) == {"model": "gpt-5.6-sol", "label": "primary"}
    assert resolve_smart_model_route(
        "hi", model="gpt-5.6-sol", runtime=RUNTIME, config=CONFIG, platform="telegram"
    ) == {"model": "gpt-5.6-sol", "label": "primary"}


def test_tier_configured_for_a_different_provider_is_ignored():
    config = {**CONFIG, "cheap_model": {"model": "gpt-5.6-luna", "provider": "openrouter"}}

    assert resolve_smart_model_route(
        "hi", model="gpt-5.6-sol", runtime=RUNTIME, config=config, platform="discord"
    ) == {"model": "gpt-5.6-sol", "label": "primary"}
