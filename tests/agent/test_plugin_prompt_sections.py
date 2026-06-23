"""End-to-end cache-safety tests for plugin system-prompt sections."""

from unittest.mock import patch

from agent.conversation_loop import _restore_or_build_system_prompt
from hermes_cli import plugins as plugins_module
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_state import SessionDB
from run_agent import AIAgent


def _make_agent(session_id: str, db: SessionDB) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            session_id=session_id,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._session_db = db
    agent._session_db_created = True
    return agent


def test_plugin_prompt_is_persisted_and_byte_identical_after_resume(
    tmp_path, monkeypatch
):
    state = {"value": "first", "section_calls": 0, "hint_calls": 0}

    def section(session_info):
        state["section_calls"] += 1
        return f"Fixture state: {state['value']} ({session_info['session_id']})"

    def environment_hints(**_kwargs):
        state["hint_calls"] += 1
        return {"hints": [("Fixture environment", state["value"])]}

    def make_manager():
        manager = PluginManager()
        context = PluginContext(PluginManifest(name="fixture-plugin"), manager)
        context.register_system_prompt_section(
            "fixture.rules",
            section,
            position="after_memory",
            max_chars=200,
        )
        context.register_hook("build_environment_hints", environment_hints)
        return manager

    manager = make_manager()
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-1", "cli")
        first_agent = _make_agent("session-1", db)
        _restore_or_build_system_prompt(first_agent, None, [])
        first_prompt = first_agent._cached_system_prompt

        stored = db.get_session("session-1")
        assert stored["system_prompt"] == first_prompt
        assert stored["plugin_prompt_state"]
        assert state["section_calls"] == 1
        assert state["hint_calls"] == 1

        # Simulate a process restart with changed plugin state and a new agent.
        state["value"] = "second"
        monkeypatch.setattr(plugins_module, "_plugin_manager", make_manager())
        resumed_agent = _make_agent("session-1", db)
        _restore_or_build_system_prompt(
            resumed_agent,
            None,
            [{"role": "user", "content": "continue"}],
        )

        assert resumed_agent._cached_system_prompt.encode(
            "utf-8"
        ) == first_prompt.encode("utf-8")
        assert "Fixture state: first" in resumed_agent._cached_system_prompt
        assert "Fixture environment: first" in resumed_agent._cached_system_prompt
        assert state["section_calls"] == 1
        assert state["hint_calls"] == 1

        # Compression rebuilds use the separately persisted frozen outputs.
        resumed_agent._invalidate_system_prompt()
        rebuilt = resumed_agent._build_system_prompt()
        assert rebuilt == first_prompt
        assert state["section_calls"] == 1
        assert state["hint_calls"] == 1

        # A different session evaluates the plugin again and sees new state.
        db.create_session("session-2", "cli")
        next_agent = _make_agent("session-2", db)
        _restore_or_build_system_prompt(next_agent, None, [])
        assert "Fixture state: second" in next_agent._cached_system_prompt
        assert "Fixture environment: second" in next_agent._cached_system_prompt
        assert state["section_calls"] == 2
        assert state["hint_calls"] == 2
    finally:
        db.close()
