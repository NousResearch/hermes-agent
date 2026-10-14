from __future__ import annotations

from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin


class _Agent:
    def __init__(self) -> None:
        self._pre_agent_primary = {"model": "configured-primary"}
        self.seen_supersede = None

    def switch_model(self, *, supersede_pre_agent_primary=True, **_kwargs) -> None:
        self.seen_supersede = supersede_pre_agent_primary
        if supersede_pre_agent_primary:
            self._pre_agent_primary = None



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

def test_one_turn_runtime_restore_preserves_pending_startup_primary_intent() -> None:
    cli = CLIModelSwitchMixin()
    agent = _Agent()
    cli.agent = agent

    intent = agent._pre_agent_primary
    cli._restore_model_runtime_snapshot(
        {
            "agent_primary_runtime": None,
            "model": "temporary-fallback",
            "provider": "openai",
        }
    )

    assert agent.seen_supersede is False
    assert agent._pre_agent_primary is intent
