"""Gateway integration tests for smart model routing."""

from gateway import run as gateway_run


class _Runner:
    _service_tier = None


RUNTIME = {
    "provider": "openai-codex",
    "base_url": "https://chatgpt.com/backend-api/codex",
    "api_mode": "chat_completions",
    "api_key": "test-key",
    "args": [],
}


def test_turn_route_uses_configured_smart_model_tier(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "smart_model_routing": {
                "enabled": True,
                "platforms": ["discord"],
                "cheap_model": {"model": "gpt-5.6-luna"},
            }
        },
    )

    base_route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        _Runner(), "hi", "gpt-5.6-terra", RUNTIME
    )
    route = gateway_run.GatewayRunner._apply_smart_model_route(
        _Runner(), "hi", base_route, platform="discord"
    )

    assert route["model"] == "gpt-5.6-luna"
    assert route["signature"][0] == "gpt-5.6-luna"
    assert route["routing_label"] == "cheap"


def test_turn_route_keeps_primary_without_a_matching_platform(monkeypatch):
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "smart_model_routing": {
                "enabled": True,
                "platforms": ["telegram"],
                "cheap_model": {"model": "gpt-5.6-luna"},
            }
        },
    )

    base_route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        _Runner(), "hi", "gpt-5.6-terra", RUNTIME
    )
    route = gateway_run.GatewayRunner._apply_smart_model_route(
        _Runner(), "hi", base_route, platform="discord"
    )

    assert route["model"] == "gpt-5.6-terra"
    assert route["routing_label"] == "primary"
