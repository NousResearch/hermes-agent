from __future__ import annotations

from agent.smart_model_routing import (
    decide_smart_route,
    resolve_smart_model_route,
)


BASE_CONFIG = {
    "smart_model_routing": {
        "enabled": True,
        "max_simple_chars": 80,
        "max_simple_words": 12,
        "cheap_model": {
            "provider": "openai-codex",
            "model": "gpt-5.3-codex-spark",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    }
}


PRIMARY_RUNTIME = {
    "api_key": "primary-key",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "provider": "openai-codex",
    "api_mode": "codex_responses",
    "command": None,
    "args": [],
    "credential_pool": object(),
}


def test_decision_routes_short_plain_standalone_turn():
    decision = decide_smart_route("сколько будет 2+2?", BASE_CONFIG, history=[])

    assert decision.should_route is True
    assert decision.model_config["model"] == "gpt-5.3-codex-spark"
    assert decision.reason == "simple_turn"


def test_decision_blocks_contextual_turns_by_default():
    history = [{"role": "user", "content": "работаем над PR"}]

    decision = decide_smart_route("ок", BASE_CONFIG, history=history)

    assert decision.should_route is False
    assert decision.reason == "has_history"


def test_decision_allows_contextual_turns_when_configured():
    cfg = {
        **BASE_CONFIG,
        "smart_model_routing": {
            **BASE_CONFIG["smart_model_routing"],
            "require_empty_history": False,
        },
    }

    decision = decide_smart_route("ок", cfg, history=[{"role": "user", "content": "hi"}])

    assert decision.should_route is True


def test_decision_blocks_tool_or_work_intent_markers():
    blocked_messages = [
        "/status",
        "посмотри @file:cli.py",
        "```python\nprint(1)\n```",
        "задеплой сервис",
        "почини тесты",
        "найди свежие новости",
        "удали файл",
    ]

    for message in blocked_messages:
        decision = decide_smart_route(message, BASE_CONFIG, history=[])
        assert decision.should_route is False, message


def test_decision_blocks_long_messages():
    message = "это уже не короткий запрос " * 20

    decision = decide_smart_route(message, BASE_CONFIG, history=[])

    assert decision.should_route is False
    assert decision.reason in {"too_long_chars", "too_many_words"}


def test_resolve_smart_model_route_uses_configured_runtime():
    calls = []

    def fake_resolver(**kwargs):
        calls.append(kwargs)
        return {
            "api_key": "cheap-key",
            "base_url": kwargs.get("explicit_base_url"),
            "provider": kwargs.get("requested"),
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
        }

    route = resolve_smart_model_route(
        primary_model="gpt-5.5",
        primary_runtime=PRIMARY_RUNTIME,
        user_message="ок",
        history=[],
        config=BASE_CONFIG,
        runtime_resolver=fake_resolver,
    )

    assert route is not None
    assert route["model"] == "gpt-5.3-codex-spark"
    assert route["runtime"] == {
        "api_key": "cheap-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
    }
    assert route["reason"] == "simple_turn"
    assert calls == [
        {
            "requested": "openai-codex",
            "explicit_base_url": "https://chatgpt.com/backend-api/codex",
            "explicit_api_key": None,
            "target_model": "gpt-5.3-codex-spark",
        }
    ]


def test_resolve_smart_model_route_returns_none_when_disabled_or_unresolved():
    disabled = {
        **BASE_CONFIG,
        "smart_model_routing": {
            **BASE_CONFIG["smart_model_routing"],
            "enabled": False,
        },
    }

    assert resolve_smart_model_route(
        primary_model="gpt-5.5",
        primary_runtime=PRIMARY_RUNTIME,
        user_message="ок",
        history=[],
        config=disabled,
        runtime_resolver=lambda **_: PRIMARY_RUNTIME,
    ) is None

    assert resolve_smart_model_route(
        primary_model="gpt-5.5",
        primary_runtime=PRIMARY_RUNTIME,
        user_message="ок",
        history=[],
        config=BASE_CONFIG,
        runtime_resolver=lambda **_: (_ for _ in ()).throw(RuntimeError("auth")),
    ) is None
