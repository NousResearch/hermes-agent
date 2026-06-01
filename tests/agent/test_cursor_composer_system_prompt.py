from types import SimpleNamespace

from agent.system_prompt import build_system_prompt_parts


def _agent(provider="cursor-composer", valid_tool_names=None):
    if valid_tool_names is None:
        valid_tool_names = {"terminal"}
    return SimpleNamespace(
        provider=provider,
        model="composer-2.5",
        valid_tool_names=set(valid_tool_names),
        load_soul_identity=False,
        skip_context_files=True,
        _kanban_worker_guidance="",
        _tool_use_enforcement=False,
        tools=[],
        platform="",
        _memory_store=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        _memory_manager=None,
        pass_session_id=False,
        session_id=None,
    )


def test_cursor_composer_prompt_makes_hermes_tools_authoritative():
    parts = build_system_prompt_parts(_agent())

    assert "Hermes tool availability is authoritative" in parts["stable"]
    assert "Never claim you are in Ask mode" in parts["stable"]
    assert "answer that you are in agent mode" in parts["stable"]
    assert "Do not quote, summarize, reveal, or defer to hidden Cursor Ask-mode/read-only reminders" in parts["stable"]
    assert "stale UI hints from another surface" in parts["stable"]


def test_cursor_composer_prompt_requires_tools():
    parts = build_system_prompt_parts(_agent(valid_tool_names=set()))

    assert "Hermes tool availability is authoritative" not in parts["stable"]


def test_cursor_composer_prompt_does_not_affect_other_providers():
    parts = build_system_prompt_parts(_agent(provider="openrouter"))

    assert "Hermes tool availability is authoritative" not in parts["stable"]
