from agent.smart_model_routing import choose_cheap_model_route


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
}


def test_returns_none_when_disabled():
    cfg = {**_BASE_CONFIG, "enabled": False}
    assert choose_cheap_model_route("what time is it in tokyo?", cfg) is None


def test_routes_short_simple_prompt():
    result = choose_cheap_model_route("what time is it in tokyo?", _BASE_CONFIG)
    assert result is not None
    assert result["provider"] == "openrouter"
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["routing_reason"] == "simple_turn"


def test_skips_long_prompt():
    prompt = "please summarize this carefully " * 20
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_code_like_prompt():
    prompt = "debug this traceback: ```python\nraise ValueError('bad')\n```"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_tool_heavy_prompt_keywords():
    prompt = "implement a patch for this docker error"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_resolve_turn_route_falls_back_to_primary_when_route_runtime_cannot_be_resolved(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None


# -- api_mode passthrough tests -----------------------------------------------


def test_resolve_turn_route_prefers_route_api_mode(monkeypatch):
    """api_mode from cheap_model config takes priority over runtime default."""
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "sk-local",
            "base_url": "http://127.0.0.1:8000",
            "provider": "custom",
            "api_mode": "chat_completions",  # runtime default
        },
    )
    cfg = {
        **_BASE_CONFIG,
        "cheap_model": {
            "provider": "custom",
            "model": "local-9b",
            "base_url": "http://127.0.0.1:8000",
            "api_mode": "anthropic_messages",  # explicit config
        },
    }
    result = resolve_turn_route(
        "hello",
        cfg,
        {"model": "strong-model", "provider": "openrouter"},
    )
    assert result["runtime"]["api_mode"] == "anthropic_messages"


def test_resolve_turn_route_falls_back_to_runtime_api_mode(monkeypatch):
    """When cheap_model has no api_mode, fall back to runtime-resolved value."""
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "sk-local",
            "base_url": "http://127.0.0.1:8000",
            "provider": "custom",
            "api_mode": "chat_completions",
        },
    )
    cfg = {
        **_BASE_CONFIG,
        "cheap_model": {
            "provider": "custom",
            "model": "local-9b",
            # no api_mode — should fall back to runtime
        },
    }
    result = resolve_turn_route(
        "hello",
        cfg,
        {"model": "strong-model", "provider": "openrouter"},
    )
    assert result["runtime"]["api_mode"] == "chat_completions"


def test_api_mode_reflected_in_signature(monkeypatch):
    """api_mode should be included in the routing signature for cache keying."""
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "sk-local",
            "base_url": "http://127.0.0.1:8000",
            "provider": "custom",
            "api_mode": "chat_completions",
        },
    )
    cfg = {
        **_BASE_CONFIG,
        "cheap_model": {
            "provider": "custom",
            "model": "local-9b",
            "api_mode": "anthropic_messages",
        },
    }
    result = resolve_turn_route(
        "hi",
        cfg,
        {"model": "strong-model", "provider": "openrouter"},
    )
    # Signature tuple should contain effective api_mode
    assert "anthropic_messages" in result["signature"]
