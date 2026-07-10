from types import SimpleNamespace


ROUTER_CONFIG = {
    "model_router": {
        "enabled": True,
        "default": "standard",
        "routes": {
            "quick": {"model": "gpt-5.6-luna"},
            "standard": {"model": "gpt-5.6-terra"},
            "complex": {"model": "gpt-5.6-sol"},
        },
    }
}


def test_select_model_routes_quick_standard_and_complex():
    from hermes_cli.model_router import select_model_for_turn

    assert select_model_for_turn("what time is it?", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-luna",
        "quick",
    )
    assert select_model_for_turn("implement the auth route", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-terra",
        "standard",
    )
    assert select_model_for_turn("draft an architectural migration plan", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-sol",
        "complex",
    )


def test_select_model_supports_explicit_route_overrides():
    from hermes_cli.model_router import select_model_for_turn

    assert select_model_for_turn("/sol advise on the data model", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-sol",
        "complex",
    )
    assert select_model_for_turn("route:luna summarize this", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-luna",
        "quick",
    )


def test_select_model_disabled_keeps_base_model():
    from hermes_cli.model_router import select_model_for_turn

    assert select_model_for_turn(
        "architect the platform",
        "gpt-5.5",
        {"model_router": {"enabled": False}},
    ) == ("gpt-5.5", None)


def test_cli_turn_route_uses_model_router(monkeypatch):
    import cli as cli_mod
    import hermes_cli.model_router as model_router

    monkeypatch.setattr(
        model_router,
        "select_model_for_turn",
        lambda message, base_model: ("gpt-5.6-sol", "complex"),
    )
    stub = SimpleNamespace(
        model="gpt-5.6-terra",
        api_key="primary-key",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        api_mode="codex_responses",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        service_tier="",
    )

    route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "architect this")

    assert route["model"] == "gpt-5.6-sol"
    assert route["router_tier"] == "complex"
    assert route["signature"][0] == "gpt-5.6-sol"
    assert route["runtime"]["provider"] == "openai-codex"


def test_gateway_turn_route_uses_model_router(monkeypatch):
    import gateway.run as gateway_run
    import hermes_cli.model_router as model_router

    monkeypatch.setattr(
        model_router,
        "select_model_for_turn",
        lambda message, base_model: ("gpt-5.6-luna", "quick"),
    )
    runner = SimpleNamespace(_service_tier="")
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner,
        "what is the status?",
        "gpt-5.6-terra",
        runtime_kwargs,
    )

    assert route["model"] == "gpt-5.6-luna"
    assert route["router_tier"] == "quick"
    assert route["signature"][0] == "gpt-5.6-luna"
    assert route["runtime"]["provider"] == "openai-codex"
