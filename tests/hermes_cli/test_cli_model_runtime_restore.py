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
