"""Service ceilings reach real agent initialization and provider schema injection."""

import json

import pytest


@pytest.mark.parametrize("source", ["environment", "config"])
@pytest.mark.parametrize("requested", [None, ["terminal", "memory", "context_engine"]])
def test_provider_schemas_follow_service_ceiling(tmp_path, monkeypatch, source, requested):
    import plugins.context_engine as engines
    from run_agent import AIAgent

    home = tmp_path / "home"
    provider = home / "plugins" / "ceiling_memory"
    provider.mkdir(parents=True)
    provider.joinpath("__init__.py").write_text('''
from agent.memory_provider import MemoryProvider
class CeilingMemory(MemoryProvider):
    name = "ceiling_memory"
    def is_available(self): return True
    def initialize(self, session_id, **kwargs): pass
    def get_tool_schemas(self):
        return [{"name": "ceiling_recall", "description": "Recall", "parameters": {"type": "object", "properties": {}}}]
''')
    engine = tmp_path / "engines" / "ceiling_context"
    engine.mkdir(parents=True)
    engine.joinpath("__init__.py").write_text('''
from agent.context_engine import ContextEngine
class CeilingContext(ContextEngine):
    name = "ceiling_context"
    def update_from_response(self, usage): pass
    def should_compress(self, prompt_tokens=None): return False
    def compress(self, messages, current_tokens=None): return messages
    def get_tool_schemas(self):
        return [{"name": "ceiling_expand", "description": "Expand", "parameters": {"type": "object", "properties": {}}}]
''')
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(engines, "_CONTEXT_ENGINE_PLUGINS_DIR", engine.parent)
    monkeypatch.delenv("HERMES_ALLOWED_TOOLSETS", raising=False)
    config = {"memory": {"provider": "ceiling_memory"}, "context": {"engine": "ceiling_context"}}

    # Denied → permitted → denied exercises both real injectors and schema-cache reuse.
    for allowed in (["terminal"], ["terminal", "memory", "context_engine"], ["terminal"]):
        if source == "environment":
            monkeypatch.setenv("HERMES_ALLOWED_TOOLSETS", ",".join(allowed))
        else:
            config["agent"] = {"allowed_toolsets": allowed}
        home.joinpath("config.yaml").write_text(json.dumps(config))
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            enabled_toolsets=requested, quiet_mode=True, skip_context_files=True,
        )
        try:
            assert agent._memory_manager.providers
            assert agent.context_compressor.name == "ceiling_context"
            names = {tool["function"]["name"] for tool in agent.tools}
            assert ("ceiling_recall" in names) == ("memory" in allowed)
            assert ("ceiling_expand" in names) == ("context_engine" in allowed)
            assert names == agent.valid_tool_names
            assert set(agent.enabled_toolsets) <= set(allowed)
        finally:
            agent.close()
