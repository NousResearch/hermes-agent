from types import SimpleNamespace


def test_fallback_chain_uses_configured_model_tier():
    from hermes_cli.fallback_config import get_fallback_chain

    cfg = {
        "model": {"tier": "cheap"},
        "fallback_providers": [{"provider": "anthropic", "model": "global"}],
        "fallback_tiers": {
            "cheap": [{"provider": "openai", "model": "gpt-5-nano"}],
            "hard": [{"provider": "anthropic", "model": "claude-opus"}],
        },
    }

    assert get_fallback_chain(cfg) == [{"provider": "openai", "model": "gpt-5-nano"}]


def test_routing_context_can_select_different_fallback_tier():
    from hermes_cli.fallback_config import get_fallback_chain
    from hermes_cli.model_routing import classify_task_context, get_route_override

    cfg = {
        "routing": {
            "chat": {"provider": "openrouter", "model": "small"},
            "tool_use": {"provider": "anthropic", "model": "sonnet", "tier": "hard"},
        },
        "fallback_tiers": {
            "hard": [{"provider": "anthropic", "model": "opus"}],
        },
    }

    context = classify_task_context(cfg, "please edit the file and run tests")

    assert context == "tool_use"
    assert get_route_override(cfg, context)["model"] == "sonnet"
    assert get_fallback_chain(cfg, task_context=context) == [
        {"provider": "anthropic", "model": "opus"}
    ]


def test_cli_turn_route_applies_routing_override(monkeypatch):
    import cli
    from cli import HermesCLI
    import hermes_cli.runtime_provider as runtime_provider

    cfg = {
        "routing": {
            "chat": {"provider": "openrouter", "model": "small-chat"},
            "tool_use": {"provider": "anthropic", "model": "claude-sonnet"},
        },
        "fallback_tiers": {
            "hard": [{"provider": "anthropic", "model": "claude-opus"}],
        },
    }
    cfg["routing"]["tool_use"]["tier"] = "hard"
    monkeypatch.setattr(cli, "CLI_CONFIG", cfg)
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kw: {
            "api_key": "anthropic-key",
            "base_url": "https://api.anthropic.com/v1/anthropic",
            "provider": kw["requested"],
            "api_mode": "anthropic_messages",
            "command": None,
            "args": [],
            "credential_pool": None,
        },
    )

    shell = SimpleNamespace(
        model="small-chat",
        api_key="openrouter-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        service_tier=None,
    )

    route = HermesCLI._resolve_turn_agent_config(shell, "edit file.py")

    assert route["model"] == "claude-sonnet"
    assert route["runtime"]["provider"] == "anthropic"
    assert route["fallback_model"] == [
        {"provider": "anthropic", "model": "claude-opus"}
    ]


def test_gateway_turn_route_applies_routing_override(monkeypatch):
    import gateway.run as gateway_run
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "routing": {
                "chat": {"provider": "openrouter", "model": "small-chat"},
                "tool_use": {"provider": "openai", "model": "gpt-5", "tier": "mid"},
            },
            "fallback_tiers": {
                "mid": [{"provider": "openai", "model": "gpt-5-mini"}],
            },
        },
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kw: {
            "api_key": "openai-key",
            "base_url": "https://api.openai.com/v1",
            "provider": kw["requested"],
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
            "credential_pool": "pool",
        },
    )
    runner = SimpleNamespace(_service_tier=None)
    runtime_kwargs = {
        "api_key": "openrouter-key",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner,
        "run pytest",
        "small-chat",
        runtime_kwargs,
    )

    assert route["model"] == "gpt-5"
    assert route["runtime"]["provider"] == "openai"
    assert route["runtime"]["credential_pool"] == "pool"
    assert route["fallback_model"] == [{"provider": "openai", "model": "gpt-5-mini"}]


def test_gateway_cached_agent_prefers_routed_fallback_chain():
    import gateway.run as gateway_run

    agent = SimpleNamespace(
        _fallback_chain=[],
        _fallback_model=None,
        _fallback_activated=False,
        _fallback_index=0,
    )
    routed_chain = [{"provider": "openai", "model": "gpt-5-mini"}]
    global_chain = [{"provider": "anthropic", "model": "claude-haiku"}]

    gateway_run.GatewayRunner._apply_fallback_chain_to_agent(
        agent,
        routed_chain or global_chain,
    )

    assert agent._fallback_chain == routed_chain
    assert agent._fallback_model == routed_chain[0]
