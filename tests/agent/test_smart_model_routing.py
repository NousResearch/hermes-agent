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


# -- CJK keyword matching tests -----------------------------------------------


def test_chinese_simple_greeting_routes_to_cheap():
    """Simple Chinese greetings should route to the cheap model."""
    assert choose_cheap_model_route("你好", _BASE_CONFIG) is not None
    assert choose_cheap_model_route("晚安", _BASE_CONFIG) is not None
    assert choose_cheap_model_route("谢谢", _BASE_CONFIG) is not None


def test_chinese_task_keywords_route_to_strong():
    """Chinese messages with task/action keywords should route to strong model."""
    # 帮我 (help me) — task delegation
    assert choose_cheap_model_route("帮我查一下附近的奶茶店", _BASE_CONFIG) is None
    # 记住 (remember) — memory operation
    assert choose_cheap_model_route("记住我喜欢喝红茶玛奇朵", _BASE_CONFIG) is None
    # 设置 (configure) — tool invocation
    assert choose_cheap_model_route("帮我设置一个提醒", _BASE_CONFIG) is None


def test_chinese_search_keywords_route_to_strong():
    """Chinese search/lookup keywords should route to strong model."""
    assert choose_cheap_model_route("搜索最近的天气预报", _BASE_CONFIG) is None
    assert choose_cheap_model_route("查一下明天的航班", _BASE_CONFIG) is None


def test_chinese_technical_keywords_route_to_strong():
    """Chinese technical keywords should route to strong model."""
    assert choose_cheap_model_route("分析这个问题的原因", _BASE_CONFIG) is None
    assert choose_cheap_model_route("研究下这个方案", _BASE_CONFIG) is None


def test_english_behavior_unchanged():
    """English routing should not be affected by CJK additions."""
    # Simple English still routes cheap
    assert choose_cheap_model_route("hello", _BASE_CONFIG) is not None
    assert choose_cheap_model_route("good morning", _BASE_CONFIG) is not None
    # Complex English still routes strong
    assert choose_cheap_model_route("debug this error", _BASE_CONFIG) is None
    assert choose_cheap_model_route("implement a feature", _BASE_CONFIG) is None


def test_mixed_cjk_english_with_keyword():
    """Mixed CJK+English with a complex keyword routes to strong."""
    assert choose_cheap_model_route("帮我debug这个bug", _BASE_CONFIG) is None
