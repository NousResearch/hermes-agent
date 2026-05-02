from __future__ import annotations

from types import SimpleNamespace


def _runtime(provider="primary"):
    return {
        "api_key": f"{provider}-key",
        "base_url": f"https://{provider}.example/v1",
        "provider": provider,
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
    }


def test_cli_turn_config_uses_smart_route_for_default_model(monkeypatch):
    import cli
    from cli import HermesCLI

    calls = []

    def fake_route(**kwargs):
        calls.append(kwargs)
        return {
            "model": "cheap-model",
            "runtime": _runtime("cheap"),
            "reason": "simple_turn",
        }

    monkeypatch.setattr("agent.smart_model_routing.resolve_smart_model_route", fake_route)
    monkeypatch.setattr(cli, "CLI_CONFIG", {"smart_model_routing": {"enabled": True}})

    shell = SimpleNamespace(
        api_key="primary-key",
        base_url="https://primary.example/v1",
        provider="primary",
        api_mode="codex_responses",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        model="primary-model",
        _model_is_default=True,
        conversation_history=[],
        service_tier=None,
    )

    route = HermesCLI._resolve_turn_agent_config(shell, "ок")

    assert route["model"] == "cheap-model"
    assert route["runtime"]["provider"] == "cheap"
    assert route["signature"][:4] == (
        "cheap-model",
        "cheap",
        "https://cheap.example/v1",
        "codex_responses",
    )
    assert route["route_reason"] == "smart:simple_turn"
    assert calls[0]["history"] == []


def test_cli_turn_config_preserves_explicit_model_pin(monkeypatch):
    from cli import HermesCLI

    def fake_route(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("smart routing must not run for explicit model pins")

    monkeypatch.setattr("agent.smart_model_routing.resolve_smart_model_route", fake_route)

    shell = SimpleNamespace(
        api_key="primary-key",
        base_url="https://primary.example/v1",
        provider="primary",
        api_mode="codex_responses",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        model="pinned-model",
        _model_is_default=False,
        conversation_history=[],
        service_tier=None,
    )

    route = HermesCLI._resolve_turn_agent_config(shell, "ок")

    assert route["model"] == "pinned-model"
    assert route["route_reason"] == "primary"


def test_gateway_turn_config_uses_route_context(monkeypatch):
    from gateway.run import GatewayRunner

    calls = []

    def fake_route(**kwargs):
        calls.append(kwargs)
        return {
            "model": "cheap-model",
            "runtime": _runtime("cheap"),
            "reason": "simple_turn",
        }

    monkeypatch.setattr("agent.smart_model_routing.resolve_smart_model_route", fake_route)

    runner = SimpleNamespace(
        _service_tier=None,
        _smart_model_route_context={
            "history": [],
            "user_config": {"smart_model_routing": {"enabled": True}},
            "disable": False,
        },
    )

    route = GatewayRunner._resolve_turn_agent_config(
        runner,
        "ок",
        "primary-model",
        _runtime("primary"),
    )

    assert route["model"] == "cheap-model"
    assert route["runtime"]["provider"] == "cheap"
    assert route["route_reason"] == "smart:simple_turn"
    assert calls[0]["history"] == []
    assert calls[0]["config"] == {"smart_model_routing": {"enabled": True}}


def test_gateway_turn_config_preserves_session_override_context(monkeypatch):
    from gateway.run import GatewayRunner

    def fake_route(**_kwargs):  # pragma: no cover - should not be called
        raise AssertionError("smart routing must not run when disabled by session override")

    monkeypatch.setattr("agent.smart_model_routing.resolve_smart_model_route", fake_route)

    runner = SimpleNamespace(
        _service_tier=None,
        _smart_model_route_context={"disable": True},
    )

    route = GatewayRunner._resolve_turn_agent_config(
        runner,
        "ок",
        "session-model",
        _runtime("session-provider"),
    )

    assert route["model"] == "session-model"
    assert route["runtime"]["provider"] == "session-provider"
    assert route["route_reason"] == "primary"
