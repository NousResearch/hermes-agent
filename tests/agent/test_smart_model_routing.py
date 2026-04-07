from agent.smart_model_routing import choose_cheap_model_route


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
    },
}


def test_returns_none_when_disabled():
    cfg = {**_BASE_CONFIG, "enabled": False}
    assert choose_cheap_model_route("what time is it in tokyo?", cfg) is None


def test_routes_short_simple_prompt():
    result = choose_cheap_model_route("what time is it in tokyo?", _BASE_CONFIG)
    assert result is not None
    assert result["provider"] == "anthropic"
    assert result["model"] == "claude-haiku-4-5-20251001"
    assert result["routing_reason"] == "simple_turn"


def test_routes_medium_length_simple_prompt():
    """Messages up to 300 chars / 50 words should route to cheap model."""
    prompt = "Can you explain what the compression threshold config option does and what a good default value would be?"
    result = choose_cheap_model_route(prompt, _BASE_CONFIG)
    assert result is not None
    assert result["model"] == "claude-haiku-4-5-20251001"


def test_skips_long_prompt():
    prompt = "please summarize this carefully " * 20
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_code_like_prompt():
    prompt = "debug this traceback: ```python\nraise ValueError('bad')\n```"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_complex_action_keywords():
    prompt = "implement a patch for this"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_allows_topic_words_without_action_verbs():
    """Words like 'test', 'tool', 'review' alone no longer trigger primary model."""
    assert choose_cheap_model_route("what does this test do?", _BASE_CONFIG) is not None
    assert choose_cheap_model_route("which tool handles file reads?", _BASE_CONFIG) is not None
    assert choose_cheap_model_route("show me the review comments", _BASE_CONFIG) is not None


def test_skips_complex_phrases():
    """Two-word phrases that signal real work should still route to primary."""
    assert choose_cheap_model_route("write tests for the router", _BASE_CONFIG) is None
    assert choose_cheap_model_route("run pytest on this module", _BASE_CONFIG) is None
    assert choose_cheap_model_route("do a code review of this PR", _BASE_CONFIG) is None
    assert choose_cheap_model_route("create plan for the migration", _BASE_CONFIG) is None


def test_allows_inline_code_backtick():
    """Single backticks (inline code references) should not block routing."""
    assert choose_cheap_model_route("what is max_tokens?", _BASE_CONFIG) is not None


def test_skips_many_newlines():
    prompt = "line1\nline2\nline3\nline4\nline5"
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
