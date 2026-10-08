"""One-turn model overrides must not erase automatic bootstrap fallback provenance."""

from unittest.mock import MagicMock, patch

import pytest

from agent.manual_fallback import prepare_turn_runtime
from tests.agent.test_manual_provider_fallback import _agent


@pytest.mark.parametrize("surface", ["cli", "tui"])
@pytest.mark.parametrize("use_switch_restore", [False, True])
def test_model_once_restores_bootstrap_provenance(surface, use_switch_restore):
    agent = _agent(auto=True)
    agent._fallback_bootstrap_active = True
    if surface == "tui":
        from tui_gateway import server
        snapshot = server._snapshot_agent_model_runtime(agent)
        restore = lambda: server._restore_agent_model_runtime(agent, snapshot)
    else:
        from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
        shell = CLIModelSwitchMixin.__new__(CLIModelSwitchMixin)
        shell.agent = agent
        for key in ("model", "provider", "api_key", "base_url", "api_mode", "reasoning_config"):
            setattr(shell, key, getattr(agent, key))
        snapshot = shell._snapshot_model_runtime()
        restore = lambda: shell._restore_model_runtime_snapshot(snapshot)
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.get_model_context_length", return_value=200_000):
        agent.switch_model(new_model="temporary", new_provider="custom", api_key="test-key",
                           base_url="https://temporary.example/v1", api_mode="chat_completions")
        assert agent._fallback_bootstrap_active is False  # an explicit permanent switch clears it
        if use_switch_restore:
            with patch.object(agent, "_restore_primary_runtime", return_value=False):
                restore()
        else:
            restore()
    assert agent.model == "primary"
    assert agent._fallback_bootstrap_active is True
    agent._fallback_auto_activate = False
    publish = MagicMock()
    with pytest.raises(RuntimeError, match="primary was unavailable at startup"):
        prepare_turn_runtime(agent, publish)
    publish.assert_not_called()
