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


# -- CJK keyword matching tests (#8516) ---------------------------------------


def test_chinese_simple_greeting_routes_to_cheap():
    """Simple Chinese greetings without task keywords should use cheap model."""
    result = choose_cheap_model_route("你好", _BASE_CONFIG)
    assert result is not None, "Simple greeting '你好' should route to cheap model"
    assert result["routing_reason"] == "simple_turn"


def test_chinese_goodnight_routes_to_cheap():
    """Simple Chinese well-wishes should use cheap model."""
    result = choose_cheap_model_route("晚安", _BASE_CONFIG)
    assert result is not None, "Simple '晚安' should route to cheap model"


def test_chinese_task_keyword_routes_to_primary():
    """Chinese messages with task keywords should route to primary model."""
    assert choose_cheap_model_route("帮我查一下附近的奶茶店", _BASE_CONFIG) is None


def test_chinese_memory_keyword_routes_to_primary():
    """Chinese messages asking to remember should route to primary model."""
    assert choose_cheap_model_route("记住我喜欢喝红茶玛奇朵", _BASE_CONFIG) is None


def test_chinese_search_keyword_routes_to_primary():
    """Chinese search requests should route to primary model."""
    assert choose_cheap_model_route("搜索最新的AI论文", _BASE_CONFIG) is None


def test_chinese_setup_keyword_routes_to_primary():
    """Chinese setup/config requests should route to primary model."""
    assert choose_cheap_model_route("帮我设置一个提醒", _BASE_CONFIG) is None


def test_chinese_code_keyword_routes_to_primary():
    """Chinese technical requests should route to primary model."""
    assert choose_cheap_model_route("帮我修改这段代码", _BASE_CONFIG) is None


def test_japanese_simple_routes_to_cheap():
    """Simple Japanese greetings should route to cheap model."""
    result = choose_cheap_model_route("こんにちは", _BASE_CONFIG)
    assert result is not None, "Simple Japanese greeting should route to cheap"


def test_japanese_task_keyword_routes_to_primary():
    """Japanese task requests should route to primary model."""
    assert choose_cheap_model_route("このバグをデバッグして", _BASE_CONFIG) is None


def test_korean_simple_routes_to_cheap():
    """Simple Korean greetings should route to cheap model."""
    result = choose_cheap_model_route("안녕하세요", _BASE_CONFIG)
    assert result is not None, "Simple Korean greeting should route to cheap"


def test_korean_task_keyword_routes_to_primary():
    """Korean task requests should route to primary model."""
    assert choose_cheap_model_route("이 패키지를 설치해주세요", _BASE_CONFIG) is None


def test_english_routing_unchanged_by_cjk_addition():
    """English routing should not be affected by CJK keyword additions."""
    # Simple English still routes to cheap
    result = choose_cheap_model_route("hello there", _BASE_CONFIG)
    assert result is not None

    # Complex English still routes to primary
    assert choose_cheap_model_route("debug this error", _BASE_CONFIG) is None


def test_mixed_cjk_english_with_task_keyword():
    """Mixed CJK+English messages with CJK task keywords route to primary."""
    assert choose_cheap_model_route("帮我 debug this", _BASE_CONFIG) is None


# -- api_mode propagation tests (#8515) ----------------------------------------


def test_resolve_turn_route_preserves_cheap_model_api_mode(monkeypatch):
    """cheap_model api_mode should take precedence over runtime-resolved value."""
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "sk-cheap",
            "base_url": "http://localhost:8000",
            "provider": "custom",
            "api_mode": "chat_completions",  # runtime returns wrong api_mode
        },
    )

    cfg = {
        "enabled": True,
        "cheap_model": {
            "provider": "custom",
            "model": "local-model",
            "base_url": "http://localhost:8000",
            "api_mode": "anthropic_messages",  # user explicitly configured this
        },
    }
    result = resolve_turn_route(
        "hello",
        cfg,
        {"model": "claude-sonnet-4", "provider": "openrouter"},
    )
    assert result["model"] == "local-model"
    assert result["runtime"]["api_mode"] == "anthropic_messages"
    assert result["signature"][3] == "anthropic_messages"


def test_resolve_turn_route_falls_back_to_runtime_api_mode(monkeypatch):
    """When cheap_model has no api_mode, use the runtime-resolved value."""
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "sk-or",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
        },
    )

    result = resolve_turn_route(
        "hello",
        _BASE_CONFIG,
        {"model": "claude-sonnet-4", "provider": "openrouter"},
    )
    assert result["runtime"]["api_mode"] == "chat_completions"


def test_resolve_turn_route_preserves_cheap_model_command_and_args(monkeypatch):
    """cheap_model command and args should take precedence over runtime."""
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": None,
            "base_url": None,
            "provider": "custom",
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )

    cfg = {
        "enabled": True,
        "cheap_model": {
            "provider": "custom",
            "model": "local-model",
            "command": "/usr/local/bin/llama-server",
            "args": ["--model", "qwen-9b"],
        },
    }
    result = resolve_turn_route(
        "hi",
        cfg,
        {"model": "claude-sonnet-4", "provider": "openrouter"},
    )
    assert result["runtime"]["command"] == "/usr/local/bin/llama-server"
    assert result["runtime"]["args"] == ["--model", "qwen-9b"]
