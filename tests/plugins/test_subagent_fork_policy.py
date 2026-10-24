"""Extension discovery and retention must not implicitly copy parent history."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("mode,options,expected", [
    ("absent", {}, False),
    ("active", {}, False),
    ("active", {"retain": True}, False),
    ("active", {"fork_context": True}, True),
    ("disabled", {"fork_context": True}, False),
    ("unloaded", {"fork_context": True}, False),
    ("denied", {"fork_context": True}, False),
])
def test_registration_and_retention_preserve_fresh_default(tmp_path, monkeypatch, mode, options, expected):
    from agent.subagent_lifecycle import SubagentLifecycleService
    from agent.turn_context import build_turn_context
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.registry import registry

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    cfg = {"compression": {"enabled": False}, "terminal": {"cwd": str(tmp_path)},
           "plugins": {"enabled": ["fork-policy"], "entries": {
               "fork-policy": {"allow_tool_override": mode != "denied"}}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **kw: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda *a, **kw: cfg)
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))
    if mode != "absent":
        plugin = tmp_path / "plugins" / "fork-policy"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: fork-policy\nversion: 0.1.0\n")
        (plugin / "__init__.py").write_text(
            "from agent.subagent_extension import SubagentExtension\n"
            "class Policy(SubagentExtension):\n"
            f"    def enabled(self): return {mode != 'disabled'!r}\n"
            f"    def child_options(self, task): return {options!r}\n"
            "    def format_fork_history(self, messages, *, task): return messages\n"
            "def register(ctx): ctx.register_subagent_extension(Policy())\n")
    discover_plugins(force=True)
    if mode == "unloaded":
        get_plugin_manager().unload("fork-policy")
    observed = {}

    def run(child, user_message, conversation_history=None, task_id=None, **kwargs):
        from agent import conversation_loop as loop
        ctx = build_turn_context(
            child, user_message, None, conversation_history, task_id, None, None,
            restore_or_build_system_prompt=loop._restore_or_build_system_prompt,
            install_safe_stdio=loop._install_safe_stdio, sanitize_surrogates=loop._sanitize_surrogates,
            summarize_user_message_for_log=loop._summarize_user_message_for_log,
            set_session_context=loop.set_session_context, set_current_write_origin=loop.set_current_write_origin,
            ra=loop._ra,
        )
        observed[child.session_id] = json.dumps(ctx.messages)
        child._flush_messages_to_session_db(ctx.messages, conversation_history)
        return {"final_response": "Done", "completed": True, "messages": ctx.messages, "api_calls": 0}

    monkeypatch.setattr(AIAgent, "run_conversation", run)
    monkeypatch.setattr("tools.delegate_tool_dispatch._resolve_async_wake_sid", lambda *args: None)
    with patch("agent.process_bootstrap.OpenAI"), SessionDB(db_path=tmp_path / "state.db") as db:
        parent = AIAgent(api_key="test-key", base_url="http://127.0.0.1:1/v1", model="test/model",
                         provider="custom", enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
                         skip_memory=True, session_db=db, session_id="owner")
        parent._ensure_db_session()
        parent._session_messages = [{"role": "user", "content": "PRIVATE_PARENT_CONTEXT"}]
        try:
            result = json.loads(registry.get_entry("delegate_task").handler({"tasks": [
                {"goal": "Default context"},
                {"goal": "Explicit fresh context", "fork_context": False},
            ]}, parent_agent=parent))
            assert all(row["status"] == "completed" for row in result["results"]), result
            assert ["PRIVATE_PARENT_CONTEXT" in messages for messages in observed.values()].count(True) == expected
            for messages in observed.values():
                if "Default context" in messages:
                    assert ("PRIVATE_PARENT_CONTEXT" in messages) is expected
                elif "Explicit fresh context" in messages:
                    assert "PRIVATE_PARENT_CONTEXT" not in messages
                else:
                    assert "PRIVATE_PARENT_CONTEXT" in messages
            explicit = json.loads(registry.get_entry("delegate_task").handler(
                {"tasks": [{"goal": "Explicit inherited context", "fork_context": True}]}, parent_agent=parent))
            if mode == "active":
                assert explicit["results"][0]["status"] == "completed", explicit
            else:
                assert "fork" in explicit.get("error", ""), explicit
            # The retained API must likewise defer to explicit policy, not infer
            # a fork from the fact that the conversation will be persisted.
            retained = SubagentLifecycleService(lambda: parent).launch_session(goal="Retained default context")
            assert retained["results"][0]["status"] == "completed", retained
            sid = retained["results"][0]["session_id"]
            assert ("PRIVATE_PARENT_CONTEXT" in observed[sid]) is expected
            assert ("PRIVATE_PARENT_CONTEXT" in json.dumps(db.get_messages_as_conversation(sid))) is expected
        finally:
            parent.close()
