"""Tests for optional gateway smart model routing."""

import gateway.run as gateway_run


def _runner():
    return object.__new__(gateway_run.GatewayRunner)


def test_model_routing_disabled_keeps_session_model(monkeypatch):
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {})

    routed = _runner()._route_model_for_turn("debug this failing test", "gpt-5.4-mini")

    assert routed == "gpt-5.4-mini"


def test_model_routing_promotes_hard_turn_when_enabled(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "model_routing": {
                "enabled": True,
                "base_model": "gpt-5.4-mini",
                "premium_model": "gpt-5.5",
            }
        },
    )

    routed = _runner()._route_model_for_turn("debug this failing test traceback", "gpt-5.4-mini")

    assert routed == "gpt-5.5"


def test_model_routing_does_not_promote_routine_turn(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "model_routing": {
                "enabled": True,
                "base_model": "gpt-5.4-mini",
                "premium_model": "gpt-5.5",
            }
        },
    )

    routed = _runner()._route_model_for_turn("what time is it?", "gpt-5.4-mini")

    assert routed == "gpt-5.4-mini"


def test_model_routing_respects_explicit_cheap_model_request(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "model_routing": {
                "enabled": True,
                "base_model": "gpt-5.4-mini",
                "premium_model": "gpt-5.5",
            }
        },
    )

    runner = _runner()

    assert runner._route_model_for_turn(
        "use mini and debug this failing test traceback",
        "gpt-5.4-mini",
    ) == "gpt-5.4-mini"
    assert runner._route_model_for_turn(
        "do not use gpt-5.5 for this architecture question",
        "gpt-5.4-mini",
    ) == "gpt-5.4-mini"
    assert runner._route_model_for_turn(
        "avoid the best model and debug this",
        "gpt-5.4-mini",
    ) == "gpt-5.4-mini"


def test_model_routing_only_promotes_configured_base_model(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "model_routing": {
                "enabled": True,
                "base_model": "gpt-5.4-mini",
                "premium_model": "gpt-5.5",
            }
        },
    )

    routed = _runner()._route_model_for_turn("debug this failing test", "claude-sonnet-4")

    assert routed == "claude-sonnet-4"


def test_resolve_turn_agent_config_uses_routed_model_in_signature(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {
            "model_routing": {
                "enabled": True,
                "base_model": "gpt-5.4-mini",
                "premium_model": "gpt-5.5",
            }
        },
    )

    route = _runner()._resolve_turn_agent_config(
        "think harder about this architecture",
        "gpt-5.4-mini",
        {"provider": "openai-api", "api_key": "***"},
    )

    assert route["model"] == "gpt-5.5"
    assert route["signature"][0] == "gpt-5.5"


def test_model_routing_match_reason_covers_long_ambiguous_strategy():
    message = " ".join(["word"] * 80) + " what should we recommend as the strategy?"

    reason = gateway_run.GatewayRunner._model_routing_match_reason(message)

    assert reason == "long ambiguous strategy request"
