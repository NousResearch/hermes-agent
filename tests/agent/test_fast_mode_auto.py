"""Bounded /fast auto|cold windows and the shared route-aware gate."""

from types import SimpleNamespace

from agent import fast_mode


def _agent(**kw):
    base = dict(
        service_tier="auto",
        model="gpt-5.4",
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_mode="chat_completions",
        request_overrides={"extra_body": {"keep": 1}},
        fast_auto_seconds=60,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_bounded_fast_window_policy(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(fast_mode.time, "monotonic", lambda: clock[0])

    # auto: window open -> fast override layered over existing overrides
    agent = _agent()
    fast_mode.begin_turn(agent, conversation_history=[])
    assert fast_mode.effective_request_overrides(agent) == {
        "extra_body": {"keep": 1},
        "service_tier": "priority",
    }
    assert agent.request_overrides == {"extra_body": {"keep": 1}}  # never mutated

    # window expired -> override absent
    clock[0] += 61
    assert fast_mode.effective_request_overrides(agent) == {"extra_body": {"keep": 1}}

    # auto re-opens on the next turn
    fast_mode.begin_turn(agent, conversation_history=[{"role": "user", "content": "x"}])
    assert "service_tier" in fast_mode.effective_request_overrides(agent)

    # cold: prior history -> no window at all
    cold = _agent(service_tier="cold")
    fast_mode.begin_turn(cold, conversation_history=[{"role": "user", "content": "x"}])
    assert "service_tier" not in fast_mode.effective_request_overrides(cold)
    fast_mode.begin_turn(cold, conversation_history=None)
    assert fast_mode.effective_request_overrides(cold)["service_tier"] == "priority"

    # Anthropic route uses the speed param
    anth = _agent(
        service_tier="auto",
        model="claude-opus-5",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
    )
    fast_mode.begin_turn(anth, conversation_history=[])
    assert fast_mode.effective_request_overrides(anth)["speed"] == "fast"

    # unsupported routes never get fast params, in auto or static mode
    from hermes_cli.models import resolve_fast_mode_overrides

    for provider, base_url in (
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("nous", "https://inference-api.nousresearch.com/v1"),
        ("copilot", "https://api.githubcopilot.com"),
        ("azure", "https://foo.openai.azure.com"),
        ("custom", "http://10.0.0.1:8000/v1"),
        ("openai", "https://proxy.example.com/v1"),
    ):
        proxied = _agent(provider=provider, base_url=base_url)
        fast_mode.begin_turn(proxied, conversation_history=[])
        assert "service_tier" not in fast_mode.effective_request_overrides(proxied), provider
        assert resolve_fast_mode_overrides("gpt-5.4", provider=provider, base_url=base_url) is None
    assert resolve_fast_mode_overrides(
        "claude-opus-5", provider="bedrock", base_url="https://bedrock-runtime.us-east-1.amazonaws.com"
    ) is None
    # first-party routes (and the legacy model-only call) still resolve
    assert resolve_fast_mode_overrides("gpt-5.4", provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex")
    assert resolve_fast_mode_overrides("grok-4.6", provider="xai", base_url="https://api.x.ai/v1")
    assert resolve_fast_mode_overrides("gpt-5.4") == {"service_tier": "priority"}

    # normal / static modes are untouched by the window logic
    static = _agent(service_tier="priority", request_overrides={"service_tier": "priority"})
    fast_mode.begin_turn(static, conversation_history=[])
    assert fast_mode.effective_request_overrides(static) == {"service_tier": "priority"}
    off = _agent(service_tier=None)
    fast_mode.begin_turn(off, conversation_history=[])
    assert fast_mode.effective_request_overrides(off) == {"extra_body": {"keep": 1}}


def test_fast_auto_and_cold_parse_and_slash_command(monkeypatch):
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {"success": True}
    import cli as cli_mod
    from gateway.run import GatewayRunner
    from hermes_cli.commands import COMMAND_REGISTRY
    from hermes_cli.config import DEFAULT_CONFIG

    # config parsing: CLI, gateway, TUI all accept auto/cold; default stays off
    for raw, expected in (("auto", "auto"), ("COLD", "cold"), ("fast", "priority"), ("", None), ("bogus", None)):
        assert cli_mod._parse_service_tier_config(raw) == expected
        monkeypatch.setattr(
            "gateway.run._load_gateway_runtime_config", lambda: {"agent": {"service_tier": raw}}
        )
        assert GatewayRunner._load_service_tier() == expected
    assert DEFAULT_CONFIG["agent"]["service_tier"] == ""
    assert DEFAULT_CONFIG["agent"]["fast_auto_seconds"] == 60

    # /fast auto — session-scoped, agent rebuilt, status reports the mode
    fast_cmd = next(c for c in COMMAND_REGISTRY if c.name == "fast")
    assert {"auto", "cold"} <= set(fast_cmd.subcommands)
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: printed.append(" ".join(map(str, a))))
    monkeypatch.setattr(cli_mod, "save_config_value", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no config write")))
    stub = SimpleNamespace(
        service_tier=None, model="gpt-5.4", agent=object(), _fast_command_available=lambda: True
    )
    cli_mod.HermesCLI._handle_fast_command(stub, "/fast auto")
    assert stub.service_tier == "auto"
    assert stub.agent is None
    cli_mod.HermesCLI._handle_fast_command(stub, "/fast status")
    assert any("auto" in line for line in printed)
    cli_mod.HermesCLI._handle_fast_command(stub, "/fast cold")
    assert stub.service_tier == "cold"

    # auto/cold do NOT pin a static override into the turn route
    route_stub = SimpleNamespace(
        model="gpt-5.4", api_key="k", base_url="https://api.openai.com/v1", provider="openai",
        api_mode="chat_completions", acp_command=None, acp_args=[], _credential_pool=None,
        service_tier="auto",
    )
    assert cli_mod.HermesCLI._resolve_turn_agent_config(route_stub, "hi")["request_overrides"] is None
    route_stub.service_tier = "priority"
    assert cli_mod.HermesCLI._resolve_turn_agent_config(route_stub, "hi")["request_overrides"] == {
        "service_tier": "priority"
    }
    route_stub.base_url = "https://openrouter.ai/api/v1"
    route_stub.provider = "openrouter"
    assert cli_mod.HermesCLI._resolve_turn_agent_config(route_stub, "hi")["request_overrides"] is None
