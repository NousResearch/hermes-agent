"""``/model <name> --reasoning <level>`` — one request carries a model pick AND its effort.

The parser is the single owner (hermes_cli.model_switch.parse_model_switch_args); the CLI and
TUI-gateway commit steps apply the effort AFTER the agent swap, because ``agent.switch_model``
re-resolves ``reasoning_config`` from config.yaml and would clobber an earlier write.
"""

from types import SimpleNamespace

from hermes_cli.model_switch import (
    MODEL_SWITCH_ERR_BAD_REASONING,
    ModelSwitchResult,
    parse_model_switch_args,
)


def test_reasoning_flag_rides_with_the_pick_and_validates():
    req = parse_model_switch_args("sonnet --provider anthropic --reasoning high --session")
    assert req.target == "sonnet"
    assert req.explicit_provider == "anthropic"
    assert req.reasoning_effort == "high"
    assert req.scope == "session"
    assert req.errors == ()

    bad = parse_model_switch_args("sonnet --reasoning turbo")
    assert MODEL_SWITCH_ERR_BAD_REASONING in bad.errors
    # Unicode dash normalization (Telegram/iOS) covers the new flag too.
    assert parse_model_switch_args("sonnet \u2014reasoning low").reasoning_effort == "low"


def test_cli_commit_applies_effort_after_the_agent_swap(monkeypatch):
    """The agent's switch_model resets reasoning_config from config; the ride-along effort must
    win over that reset, on both the CLI and the live agent."""
    import cli as cli_mod
    from hermes_cli import cli_model_switch_mixin as mixin

    class _Agent:
        reasoning_config = {"enabled": True, "effort": "medium"}

        def switch_model(self, **_kw):
            # Mirrors agent_runtime_helpers._switch_model: re-resolve from config.yaml.
            self.reasoning_config = {"enabled": True, "effort": "medium"}

    agent = _Agent()
    cli = SimpleNamespace(
        model="old", provider="nous", requested_provider="nous", _explicit_api_key="", _explicit_base_url="",
        api_key="", base_url="", api_mode="", agent=agent, reasoning_config=None,
        _pending_one_turn_model_restore=None, _pending_model_switch_note="",
        _snapshot_model_runtime=lambda: {}, _persist_model_switch_to_session=lambda *_a: None)
    cli._stage_and_swap_model = lambda result, old: cli_mod.HermesCLI._stage_and_swap_model(cli, result, old)
    monkeypatch.setattr(mixin, "_print_switch_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_mod.HermesCLI, "_persist_model_switch_to_session", lambda *_a: None)
    saved = {}
    monkeypatch.setattr(cli_mod, "save_config_value", lambda k, v: saved.setdefault(k, v) or True)

    result = ModelSwitchResult(success=True, new_model="new", target_provider="nous")
    mixin._commit_model_switch(cli, result, persist_global=False, reasoning_effort="high")

    assert agent.reasoning_config == {"enabled": True, "effort": "high"}
    assert cli.reasoning_config == {"enabled": True, "effort": "high"}
    assert "agent.reasoning_effort" not in saved  # session scope: no config write

    mixin._commit_model_switch(cli, result, persist_global=True, reasoning_effort="none")
    assert saved.get("agent.reasoning_effort") == "none"
    assert agent.reasoning_config == {"enabled": False}
