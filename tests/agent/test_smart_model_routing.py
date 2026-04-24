from agent.smart_model_routing import choose_cheap_model_route, choose_complex_model_route


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
}

_BIDIRECTIONAL_CONFIG = {
    "enabled": True,
    "complex_keywords": ["research", "review", "audit", "研究", "審查", "審閱"],
    "cheap_model": {
        "provider": "openai-codex",
        "model": "gpt-5.4-mini",
    },
    "complex_model": {
        "provider": "openai-codex",
        "model": "gpt-5.4",
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


def test_routes_complex_prompt_when_complex_model_configured():
    prompt = "please review this PR diff carefully"
    result = choose_complex_model_route(prompt, _BIDIRECTIONAL_CONFIG)
    assert result is not None
    assert result["provider"] == "openai-codex"
    assert result["model"] == "gpt-5.4"
    assert result["routing_reason"] == "complex_turn"


def test_routes_chinese_review_prompt_when_complex_model_configured():
    prompt = "幫我研究這個競品，順便做審查"
    result = choose_complex_model_route(prompt, _BIDIRECTIONAL_CONFIG)
    assert result is not None
    assert result["model"] == "gpt-5.4"
    assert choose_cheap_model_route(prompt, _BIDIRECTIONAL_CONFIG) is None


def test_does_not_route_non_review_complex_prompt_to_complex_model():
    prompt = "implement a patch for this docker error"
    assert choose_complex_model_route(prompt, _BIDIRECTIONAL_CONFIG) is None


def test_does_not_route_simple_prompt_to_complex_model():
    assert choose_complex_model_route("hi there", _BIDIRECTIONAL_CONFIG) is None


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
            "api_key": "***",
        },
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None


def test_resolve_turn_route_uses_complex_model_when_message_is_complex(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "***",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
            "credential_pool": None,
        },
    )
    result = resolve_turn_route(
        "please review this PR diff carefully",
        _BIDIRECTIONAL_CONFIG,
        {
            "model": "gpt-5.4-mini",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "api_key": "***",
        },
    )
    assert result["model"] == "gpt-5.4"
    assert result["runtime"]["provider"] == "openai-codex"
    assert result["label"] == "smart route → gpt-5.4 (openai-codex)"


def test_resolve_turn_route_keeps_primary_for_non_review_complex_message(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    result = resolve_turn_route(
        "implement a patch for this docker error",
        _BIDIRECTIONAL_CONFIG,
        {
            "model": "gpt-5.4-mini",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "api_key": "***",
        },
    )
    assert result["model"] == "gpt-5.4-mini"
    assert result["runtime"]["provider"] == "openai-codex"
    assert result["label"] is None
