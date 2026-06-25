from hermes_cli.turn_router import route_turn


def _cfg(**overrides):
    base = {
        "enabled": True,
        "providers_allowlist": ["openai-codex"],
        "default_preset": "high_standard",
    }
    base.update(overrides)
    return base


def test_router_disabled_is_noop():
    decision = route_turn("debug this sql", router_config={"enabled": False}, current_model="gpt-5.5", provider="openai-codex")
    assert not decision.enabled
    assert decision.reasoning_config is None
    assert decision.request_overrides == {}


def test_provider_not_allowlisted_is_noop():
    decision = route_turn("debug this sql", router_config=_cfg(), current_model="claude-sonnet-4", provider="anthropic")
    assert not decision.enabled
    assert decision.reason == "provider not allowlisted"


def test_explicit_overrides_win():
    decision = route_turn(
        "please do a deep refactor",
        router_config=_cfg(),
        current_model="gpt-5.5",
        provider="openai-codex",
        explicit_reasoning_config={"enabled": True, "effort": "low"},
    )
    assert not decision.enabled
    assert decision.source == "explicit"


def test_grammar_routes_low_standard():
    decision = route_turn("corrige a gramática deste texto", router_config=_cfg(), current_model="gpt-5.5", provider="openai-codex")
    assert decision.enabled
    assert decision.preset == "low_standard"
    assert decision.reasoning_config == {"enabled": True, "effort": "low"}
    assert decision.request_overrides == {}


def test_sql_debug_routes_high_standard():
    decision = route_turn("debug this SQL join and filter", router_config=_cfg(), current_model="gpt-5.5", provider="openai-codex")
    assert decision.preset == "high_standard"
    assert decision.reasoning_config == {"enabled": True, "effort": "high"}


def test_architecture_routes_xhigh_standard():
    decision = route_turn("analyze the architecture and cached agent gateway parity risk", router_config=_cfg(), current_model="gpt-5.5", provider="openai-codex")
    assert decision.preset == "xhigh_standard"
    assert decision.reasoning_config == {"enabled": True, "effort": "xhigh"}


def test_quick_routes_xhigh_fast():
    decision = route_turn("quick fix, preciso rápido e barato", router_config=_cfg(), current_model="gpt-5.5", provider="openai-codex")
    assert decision.preset == "xhigh_fast"
    assert decision.model == "gpt-5.4"


def test_emergency_priority_requires_config_gate():
    decision = route_turn("emergency production incident", router_config=_cfg(), current_model="gpt-5.5", provider="openai-codex")
    assert decision.preset == "xhigh_standard"
    assert decision.service_tier is None
    priority = route_turn("emergency production incident", router_config=_cfg(allow_auto_priority=True), current_model="gpt-5.5", provider="openai-codex")
    assert priority.preset == "high_priority"
    assert priority.service_tier == "priority"
    assert priority.request_overrides == {"service_tier": "priority"}


def test_prefix_routes_and_strips_api_message():
    decision = route_turn("!deep investigate this migration", router_config=_cfg(), current_model="gpt-5.5", provider="openai-codex")
    assert decision.source == "prefix"
    assert decision.preset == "xhigh_standard"
    assert decision.message_override == "investigate this migration"


def test_slash_commands_bypass_router():
    decision = route_turn("/model gpt-5.5", router_config=_cfg(), current_model="gpt-5.5", provider="openai-codex")
    assert not decision.enabled
    assert decision.reason == "slash command"


def test_cli_mixin_resolve_turn_route_enriches_codex_turn():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    class Dummy(CLIAgentSetupMixin):
        pass

    dummy = Dummy()
    dummy.api_key = "key"
    dummy.base_url = "https://example.invalid"
    dummy.provider = "openai-codex"
    dummy.api_mode = "responses"
    dummy.acp_command = None
    dummy.acp_args = []
    dummy.model = "gpt-5.5"
    dummy.reasoning_config = None
    dummy.service_tier = None
    dummy.request_router_config = _cfg()
    route = dummy._resolve_turn_agent_config("quick fix, preciso rápido e barato")
    assert route["model"] == "gpt-5.4"
    assert route["signature"][0] == "gpt-5.4"
    assert route["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
    assert route["router_decision"].preset == "xhigh_fast"


def test_cli_mixin_resolve_turn_route_honors_explicit_reasoning_override():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    class Dummy(CLIAgentSetupMixin):
        pass

    dummy = Dummy()
    dummy.api_key = "key"
    dummy.base_url = "https://example.invalid"
    dummy.provider = "openai-codex"
    dummy.api_mode = "responses"
    dummy.acp_command = None
    dummy.acp_args = []
    dummy.model = "gpt-5.5"
    dummy.reasoning_config = {"enabled": True, "effort": "low"}
    dummy.service_tier = None
    dummy.request_router_config = _cfg()
    route = dummy._resolve_turn_agent_config("please deeply refactor this architecture")
    assert route["model"] == "gpt-5.5"
    assert route["reasoning_config"] == {"enabled": True, "effort": "low"}
    assert not route["router_decision"].enabled
    assert route["router_decision"].source == "explicit"



def test_empty_allowlist_still_codex_only():
    allowed = route_turn(
        "debug this SQL",
        router_config={"enabled": True, "providers_allowlist": []},
        current_model="gpt-5.5",
        provider="openai-codex",
    )
    blocked = route_turn(
        "debug this SQL",
        router_config={"enabled": True, "providers_allowlist": []},
        current_model="claude-sonnet-4",
        provider="anthropic",
    )

    assert allowed.enabled
    assert not blocked.enabled
    assert blocked.reason == "provider not allowlisted"


def test_explicit_model_override_wins():
    decision = route_turn(
        "quick fix, preciso rápido e barato",
        router_config=_cfg(),
        current_model="gpt-5.5",
        provider="openai-codex",
        explicit_model_override=True,
    )

    assert not decision.enabled
    assert decision.source == "explicit"
