from types import SimpleNamespace
from unittest.mock import patch


def test_reduced_authority_agent_skips_plugin_hooks():
    from agent.plugin_hook_policy import invoke_agent_hook

    agent = SimpleNamespace(_skip_plugin_hooks=True)
    with patch("hermes_cli.plugins.invoke_hook") as invoke_hook:
        result = invoke_agent_hook(agent, "pre_llm_call", session_id="session-1")

    assert result == []
    invoke_hook.assert_not_called()


def test_normal_agent_invokes_plugin_hooks():
    from agent.plugin_hook_policy import invoke_agent_hook

    agent = SimpleNamespace(_skip_plugin_hooks=False)
    with patch("hermes_cli.plugins.invoke_hook", return_value=[{"context": "safe"}]) as invoke_hook:
        result = invoke_agent_hook(agent, "pre_llm_call", session_id="session-1")

    assert result == [{"context": "safe"}]
    invoke_hook.assert_called_once_with("pre_llm_call", session_id="session-1")
