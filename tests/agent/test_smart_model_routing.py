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


def test_routes_coding_prompt_to_dedicated_intent_route(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": kwargs["requested"],
            "base_url": kwargs.get("explicit_base_url") or "https://route.example/v1",
            "api_key": "route-key",
            "api_mode": "chat_completions",
        },
    )

    cfg = {
        "enabled": True,
        "cheap_model": {
            "provider": "openrouter",
            "model": "google/gemini-2.5-flash",
        },
        "intent_routes": {
            "coding": {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4",
            }
        },
    }
    primary = {
        "model": "openai/gpt-4.1",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "api_key": "sk-primary",
    }

    result = resolve_turn_route("please debug this traceback:\n```python\nraise ValueError('bad')\n```", cfg, primary)

    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] == "smart route[coding] → anthropic/claude-sonnet-4 (openrouter)"


def test_routes_thinking_prompt_to_dedicated_intent_route(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": kwargs["requested"],
            "base_url": "https://route.example/v1",
            "api_key": "route-key",
            "api_mode": "chat_completions",
        },
    )

    cfg = {
        "enabled": True,
        "cheap_model": {
            "provider": "openrouter",
            "model": "google/gemini-2.5-flash",
        },
        "intent_routes": {
            "thinking": {
                "provider": "openrouter",
                "model": "openai/gpt-4.1-mini",
            }
        },
    }
    primary = {
        "model": "openai/gpt-4.1",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "api_key": "sk-primary",
    }

    result = resolve_turn_route("can you analyze the tradeoffs here?", cfg, primary)

    assert result["model"] == "openai/gpt-4.1-mini"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] == "smart route[thinking] → openai/gpt-4.1-mini (openrouter)"


def test_routes_tool_prompt_to_dedicated_intent_route(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": kwargs["requested"],
            "base_url": "https://route.example/v1",
            "api_key": "route-key",
            "api_mode": "chat_completions",
        },
    )

    cfg = {
        "enabled": True,
        "cheap_model": {
            "provider": "openrouter",
            "model": "google/gemini-2.5-flash",
        },
        "intent_routes": {
            "tool": {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4",
            }
        },
    }
    primary = {
        "model": "openai/gpt-4.1",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "api_key": "sk-primary",
    }

    result = resolve_turn_route("run this command and inspect the files", cfg, primary)

    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] == "smart route[tool] → anthropic/claude-sonnet-4 (openrouter)"


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
