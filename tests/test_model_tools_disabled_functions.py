"""tools.disabled_functions + --disable-tools: remove single tools below toolset granularity (#31375)."""

import json
import os
import time

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
    monkeypatch.delenv(model_tools.DISABLED_TOOLS_ENV, raising=False)
    monkeypatch.setattr(model_tools, "_DISABLED_FUNCTIONS_CACHE", (None, frozenset()))
    return tmp_path


def test_a_disabled_tool_leaves_its_toolset_but_its_siblings_stay(home):
    before = model_tools._select_tool_names(["web"], None, quiet_mode=True)
    assert {"web_search", "web_extract"} <= before
    _write_config(home, "tools:\n  disabled_functions: [web_extract, todo]\n")
    after = model_tools._select_tool_names(["web", "todo"], None, quiet_mode=True)
    assert "web_extract" not in after and "todo_list" not in after  # legacy alias resolves too
    assert "web_search" in after
    # Direct registry calls (execute_code sandbox, plugins) bypass the agent's name check.
    result = json.loads(model_tools.handle_function_call("web_extract", {"urls": ["https://example.com"]}))
    assert "disabled by configuration" in result["error"]


def test_the_disable_tools_flag_layers_on_config_and_is_never_served_stale(home, monkeypatch):
    """``hermes --disable-tools`` rides on the environment (TUI backend and children inherit it),
    unions with the config list, and a changed value invalidates the fingerprint cache."""
    from hermes_cli._parser import build_top_level_parser, top_level_value_flag_sets

    _write_config(home, "tools:\n  disabled_functions: [web_extract]\n")
    assert model_tools.disabled_function_names() == frozenset({"web_extract"})
    monkeypatch.setenv(model_tools.DISABLED_TOOLS_ENV, "skill_manage, mcp__github__create_issue")
    assert model_tools.disabled_function_names() == frozenset({"web_extract", "skill_manage", "mcp__github__create_issue"})
    monkeypatch.delenv(model_tools.DISABLED_TOOLS_ENV)
    assert model_tools.disabled_function_names() == frozenset({"web_extract"})

    args = build_top_level_parser()[0].parse_args(["--disable-tools", "web_extract,patch", "chat"])
    assert args.disable_tools == "web_extract,patch"
    args = build_top_level_parser()[0].parse_args(["chat", "--disable-tools", "patch"])
    assert args.disable_tools == "patch"
    assert "--disable-tools" in top_level_value_flag_sets()[0]  # argv scanners must not misread the value as a subcommand


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

    _write_config(home, "tools:\n  disabled_functions: [clarify, delegate_task]\n")
    monkeypatch.setenv(model_tools.DISABLED_TOOLS_ENV, "browser_vault_fill")
    after = build()
    assert not disabled & after.valid_tool_names
    assert not disabled & {t["function"]["name"] for t in after.tools}
    assert after.valid_tool_names == before.valid_tool_names - disabled  # nothing else changed
