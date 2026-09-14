"""tools.disabled_functions: remove single tools below toolset granularity (#31375)."""

import json
import os
import time
from types import SimpleNamespace

import pytest

import model_tools


def _write_config(tmp_path, text: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    # A distinct mtime per write, so the fingerprint cache sees every change.
    stamp = time.time() + len(text)
    os.utime(path, (stamp, stamp))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(model_tools, "_DISABLED_FUNCTIONS_CACHE", (None, frozenset()))
    return tmp_path


def test_reads_the_list_and_follows_config_edits(home):
    assert model_tools.disabled_function_names() == frozenset()
    _write_config(home, "tools:\n  disabled_functions:\n    - web_extract\n    - ' skill_manage '\n")
    assert model_tools.disabled_function_names() == frozenset({"web_extract", "skill_manage"})
    _write_config(home, "tools:\n  disabled_functions: []\n")
    assert model_tools.disabled_function_names() == frozenset()


def test_a_legacy_tool_name_disables_the_tool_it_resolves_to(home):
    _write_config(home, "tools:\n  disabled_functions: [todo]\n")
    assert model_tools.disabled_function_names() == frozenset({"todo_list"})


def test_accepts_the_json_string_form_hermes_config_set_writes(home):
    _write_config(home, "tools:\n  disabled_functions: '[\"web_extract\"]'\n")
    assert model_tools.disabled_function_names() == frozenset({"web_extract"})


def test_a_disabled_tool_leaves_its_toolset_but_its_siblings_stay(home):
    before = model_tools._select_tool_names(["web"], None, quiet_mode=True)
    assert {"web_search", "web_extract"} <= before
    _write_config(home, "tools:\n  disabled_functions: [web_extract]\n")
    after = model_tools._select_tool_names(["web"], None, quiet_mode=True)
    assert "web_extract" not in after
    assert "web_search" in after


def test_direct_dispatch_of_a_disabled_tool_is_refused(home):
    _write_config(home, "tools:\n  disabled_functions: [web_extract]\n")
    result = json.loads(model_tools.handle_function_call("web_extract", {"urls": ["https://example.com"]}))
    assert "disabled by configuration" in result["error"]


def test_memory_provider_tools_named_in_the_list_are_not_added(home, monkeypatch):
    from agent import memory_manager

    _write_config(home, "tools:\n  disabled_functions: [hindsight_retain]\n")
    monkeypatch.setattr(memory_manager, "memory_provider_tools_exposed", lambda agent: True)
    provider = SimpleNamespace(get_all_tool_schemas=lambda: [
        {"name": "hindsight_retain", "description": "store", "parameters": {"type": "object", "properties": {}}},
        {"name": "hindsight_recall", "description": "search", "parameters": {"type": "object", "properties": {}}},
    ])
    agent = SimpleNamespace(_memory_manager=provider, tools=[], valid_tool_names=set())
    assert memory_manager.inject_memory_provider_tools(agent) == 1
    assert agent.valid_tool_names == {"hindsight_recall"}


def test_the_gateway_rebuilds_cached_agents_when_the_list_changes():
    from gateway.run import GatewayRunner

    before = GatewayRunner._extract_cache_busting_config({"tools": {"disabled_functions": ["web_extract"]}})
    after = GatewayRunner._extract_cache_busting_config({"tools": {"disabled_functions": []}})
    assert before["tools.disabled_functions"] == ["web_extract"]
    assert before != after


def test_an_agent_built_under_the_config_never_offers_the_disabled_tools(home, monkeypatch):
    """Integration: agent-level tools handled before generic dispatch (clarify, delegate_task) and a
    single tool of a needed toolset (browser_vault_fill) are covered, because the names never enter
    the agent's schema or valid names, which every tool call is validated against before any
    dispatch path."""
    from unittest.mock import patch

    def build():
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            return AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                           quiet_mode=True, skip_context_files=True, skip_memory=True)

    disabled = {"clarify", "delegate_task", "browser_vault_fill"}
    before = build()
    assert disabled <= before.valid_tool_names  # control: offered by default

    _write_config(home, "tools:\n  disabled_functions: [clarify, delegate_task, browser_vault_fill]\n")
    after = build()
    assert not disabled & after.valid_tool_names
    assert not disabled & {t["function"]["name"] for t in after.tools}
    assert after.valid_tool_names == before.valid_tool_names - disabled  # nothing else changed
