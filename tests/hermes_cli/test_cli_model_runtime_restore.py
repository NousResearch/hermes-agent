from __future__ import annotations

from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin


class _Agent:
    def __init__(self) -> None:
        self.model = "one-turn-model"
        self.provider = "openai"
        self.calls = []

    def switch_model(self, *, once=False, restore=False, **_kwargs) -> bool:
        self.calls.append((once, restore))
        return True



def test_successful_model_switch_only_cancels_intent_for_explicit_choice(monkeypatch) -> None:
    """#119195: user switches supersede startup recovery; internal switches do not."""
    from types import SimpleNamespace
    from agent import agent_runtime_helpers as helpers

    monkeypatch.setattr(helpers, "_resolve_switch_destination", lambda *args: ("chat_completions", "https://example.test/v1", None))
    monkeypatch.setattr(helpers, "_snapshot_switch_state", lambda agent: {})
    monkeypatch.setattr(helpers, "_swap_switch_runtime", lambda agent, model, provider, *args: (setattr(agent, "model", model), setattr(agent, "provider", provider)))
    monkeypatch.setattr(helpers, "_resolve_switch_context_length", lambda agent, snapshot: (None, None))
    monkeypatch.setattr(helpers, "_build_primary_runtime_snapshot", lambda agent, mode: {"model": agent.model})
    monkeypatch.setattr(helpers, "_finish_switch", lambda *args: None)
    monkeypatch.setattr(helpers, "_persist_switch_billing_route", lambda agent: None)
    monkeypatch.setattr("agent.chat_completion_helpers._reset_stale_streak", lambda agent: None)

    for internal in (False, True):
        intent = {"model": "configured-primary"}
        agent = SimpleNamespace(
            model="fallback", provider="deepseek", base_url="https://fallback.test/v1",
            _pre_agent_primary=intent,
            _anthropic_prompt_cache_policy=lambda **kwargs: (False, False),
        )
        kwargs = {"supersede_pre_agent_primary": False} if internal else {}
        helpers.switch_model(agent, "chosen-model", "openai", **kwargs)
        assert agent.model == "chosen-model"
        assert agent._pre_agent_primary is (intent if internal else None)

def test_one_turn_runtime_restore_uses_gateway_nonpersistent_restore() -> None:
    """The classic CLI now uses GatewayAgent; internal restore must not commit a model switch."""
    cli = CLIModelSwitchMixin()
    agent = _Agent()
    cli.agent = agent
    cli._restore_model_runtime_snapshot(
        {"agent_primary_runtime": None, "model": "previous-model", "provider": "openai"}
    )
    assert agent.calls == [(True, True)]
