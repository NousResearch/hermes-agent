from agent.smart_model_routing import choose_cheap_model_route
from agent.tiny_router import HeadPrediction, RouterOutput


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
    "tier_routes": {"low": "cheap", "medium": "primary", "high": "primary"},
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


def test_resolve_turn_route_prefers_tiny_router_when_active(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "cheap-key",
            "source": "env/config",
        },
    )
    result = resolve_turn_route(
        "please summarize this chat quickly",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
        tiny_router_config={"enabled": True, "behavior_mode": "active"},
        router_output=RouterOutput(
            relation_to_previous=HeadPrediction("follow_up", 0.7),
            actionability=HeadPrediction("none", 0.9),
            retention=HeadPrediction("ephemeral", 0.8),
            urgency=HeadPrediction("low", 0.9),
            overall_confidence=0.9,
            source="heuristic",
        ),
    )
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"].startswith("tiny-router")


def test_resolve_turn_route_uses_named_low_tier_route(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-key")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "local-key",
            "source": "env/config",
        },
    )
    cfg = {
        "enabled": True,
        "tier_routes": {"low": "local_fast", "medium": "primary", "high": "primary"},
        "routes": {
            "local_fast": {
                "provider": "custom",
                "model": "qwen2.5:7b-instruct",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_env": "LOCAL_LLM_API_KEY",
            }
        },
    }
    result = resolve_turn_route(
        "what time is it in tokyo?",
        cfg,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
    )
    assert result["model"] == "qwen2.5:7b-instruct"
    assert result["runtime"]["provider"] == "custom"
    assert "local_fast" in (result["label"] or "")


def test_tiny_router_high_stakes_uses_named_route_and_blocks_simple_downgrade(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "strong-key",
            "source": "env/config",
        },
    )
    cfg = {
        "enabled": True,
        "tier_routes": {"low": "cheap", "medium": "primary", "high": "strong_remote"},
        "cheap_model": {"provider": "openrouter", "model": "google/gemini-2.5-flash"},
        "routes": {
            "strong_remote": {
                "provider": "openrouter",
                "model": "anthropic/claude-opus-4.1",
            }
        },
    }
    result = resolve_turn_route(
        "what time is it in tokyo?",
        cfg,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
        tiny_router_config={
            "enabled": True,
            "behavior_mode": "active",
            "confidence_thresholds": {"overall": 0.4, "actionability": 0.5, "urgency": 0.5},
        },
        router_output=RouterOutput(
            relation_to_previous=HeadPrediction("follow_up", 0.9),
            actionability=HeadPrediction("act", 0.9),
            retention=HeadPrediction("ephemeral", 0.8),
            urgency=HeadPrediction("high", 0.9),
            overall_confidence=0.9,
            source="heuristic",
        ),
    )
    assert result["model"] == "anthropic/claude-opus-4.1"
    assert result["runtime"]["provider"] == "openrouter"
    assert "tiny-router" in (result["label"] or "")


def test_high_tier_budget_cap_falls_back_to_medium(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "resolved-key",
            "source": "env/config",
        },
    )

    cfg = {
        "enabled": True,
        "tier_routes": {"low": "cheap", "medium": "medium_remote", "high": "strong_remote"},
        "max_high_tier_calls_per_session": 1,
        "cheap_model": {"provider": "openrouter", "model": "google/gemini-2.5-flash"},
        "routes": {
            "medium_remote": {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4",
            },
            "strong_remote": {
                "provider": "openrouter",
                "model": "anthropic/claude-opus-4.1",
            },
        },
    }
    primary = {
        "model": "anthropic/claude-haiku-3.5",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "api_key": "primary-key",
    }

    first = resolve_turn_route(
        "do this critical action now",
        cfg,
        primary,
        tiny_router_config={"enabled": True, "behavior_mode": "active"},
        router_output=RouterOutput(
            relation_to_previous=HeadPrediction("follow_up", 0.9),
            actionability=HeadPrediction("act", 0.9),
            retention=HeadPrediction("ephemeral", 0.8),
            urgency=HeadPrediction("high", 0.9),
            overall_confidence=0.9,
            source="heuristic",
        ),
        routing_context={"session_id": "session-budget-1"},
    )
    second = resolve_turn_route(
        "another critical action",
        cfg,
        primary,
        tiny_router_config={"enabled": True, "behavior_mode": "active"},
        router_output=RouterOutput(
            relation_to_previous=HeadPrediction("follow_up", 0.9),
            actionability=HeadPrediction("act", 0.9),
            retention=HeadPrediction("ephemeral", 0.8),
            urgency=HeadPrediction("high", 0.9),
            overall_confidence=0.9,
            source="heuristic",
        ),
        routing_context={"session_id": "session-budget-1"},
    )

    assert first["model"] == "anthropic/claude-opus-4.1"
    assert first["route_tier"] == "high"
    assert second["model"] == "anthropic/claude-sonnet-4"
    assert second["route_tier"] == "medium"


def test_low_tier_route_failure_falls_back_to_primary(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    def _runtime_resolve(**kwargs):
        requested = kwargs.get("requested")
        if requested == "broken":
            raise RuntimeError("route unavailable")
        return {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "resolved-key",
            "source": "env/config",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _runtime_resolve)

    cfg = {
        "enabled": True,
        "tier_routes": {"low": "cheap", "medium": "medium_remote", "high": "primary"},
        "cheap_model": {"provider": "broken", "model": "cheap-model"},
        "routes": {
            "medium_remote": {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4",
            }
        },
    }

    result = resolve_turn_route(
        "what time is it in tokyo?",
        cfg,
        {
            "model": "anthropic/claude-haiku-3.5",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "primary-key",
        },
    )

    assert result["model"] == "anthropic/claude-haiku-3.5"
    assert result["route_tier"] == "low"
