"""An optional built-in import failure must not disable a configured provider."""
import builtins
from unittest.mock import patch

import pytest


class RecordingProvider:
    name = "bootstrap-recording"

    def __init__(self):
        self.sessions = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.sessions.append(session_id)

    def get_tool_schemas(self):
        return [{"name": "bootstrap_recall", "description": "Fixture recall",
                 "parameters": {"type": "object", "properties": {}}}]

    def shutdown(self):
        pass


@pytest.mark.parametrize("fail_builtin_import", [False, True])
@pytest.mark.parametrize("skip_memory", [False, True])
def test_public_agent_preserves_provider_selection(monkeypatch, fail_builtin_import, skip_memory):
    from run_agent import AIAgent
    from hermes_constants import get_hermes_home
    import run_agent

    home = get_hermes_home()
    monkeypatch.setattr(run_agent, "_hermes_home", home)
    provider = RecordingProvider()
    config = {"memory": {"provider": provider.name}}
    real_import = builtins.__import__
    attempts = []

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (fail_builtin_import and name == "tools.memory_tool"
                and (globals or {}).get("__name__") == "agent.agent_init"):
            attempts.append(name)
            raise ModuleNotFoundError("fixture optional built-in dependency unavailable")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("plugins.memory.load_memory_provider", return_value=provider) as load,
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata_http.get") as metadata,
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("tools.env_probe.warm_environment_probe_async"),
        patch("builtins.__import__", side_effect=guarded_import),
    ):
        metadata.return_value.json.return_value = {
            "data": [{"id": "test-model", "context_length": 204800}]}
        agent = AIAgent(model="test-model", provider="openrouter",
                        api_mode="chat_completions", api_key="test-key",
                        base_url="https://openrouter.ai/api/v1",
                        enabled_toolsets=["memory"], quiet_mode=True,
                        skip_context_files=True, skip_memory=skip_memory,
                        skip_background_review=True, save_trajectories=False,
                        session_id="bootstrap-session")
        try:
            if skip_memory:
                load.assert_not_called()
                assert agent._memory_manager is None
            else:
                assert agent._memory_manager is not None, "built-in import disabled the external provider"
                load.assert_called_once_with(provider.name)
                assert provider.sessions == ["bootstrap-session"]
                assert agent._memory_manager.has_tool("bootstrap_recall")
                assert "bootstrap_recall" in agent.valid_tool_names
            if fail_builtin_import:
                assert attempts
                assert agent._memory_store is None
            else:
                assert agent._memory_store is not None
        finally:
            agent.close()


@pytest.mark.parametrize("memory", [None, {}, "malformed", {"provider": ""}, {"provider": "builtin"}])
def test_builtin_failure_does_not_invent_provider_or_plugin_error(caplog, memory):
    from types import SimpleNamespace
    from agent import agent_init
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tools.memory_tool" and (globals or {}).get("__name__") == "agent.agent_init":
            raise ModuleNotFoundError("fixture built-in import failure")
        return real_import(name, globals, locals, fromlist, level)

    agent = SimpleNamespace(enabled_toolsets=["memory"], disabled_toolsets=[],
                            tools=[], valid_tool_names=set())
    with patch("builtins.__import__", side_effect=guarded_import), patch("plugins.memory.load_memory_provider") as load:
        agent_init._init_memory(agent, {"memory": memory}, False, None)
    load.assert_not_called()
    assert agent._memory_store is None
    assert agent._memory_manager is None
    assert not any("Memory provider plugin init failed" in r.getMessage() for r in caplog.records)
