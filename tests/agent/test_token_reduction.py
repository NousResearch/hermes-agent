"""Focused tests for opt-in token reduction features."""

from agent.model_routing import select_turn_model
from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache
from agent.tool_routing import select_task_toolsets


ALL_TOOLSETS = {
    "browser", "clarify", "code_execution", "cronjob", "delegation", "file",
    "homeassistant", "image_gen", "jev", "memory", "messaging", "session_search",
    "skills", "spotify", "terminal", "todo", "tts", "vision", "web",
}


def test_compact_catalog_keeps_names_but_omits_descriptions(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills_dir = tmp_path / "skills" / "coding" / "python-debug"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\nname: python-debug\ndescription: A deliberately verbose skill description\n---\n"
    )
    clear_skills_system_prompt_cache(clear_snapshot=True)

    result = build_skills_system_prompt(catalog_mode="compact")

    assert "python-debug" in result
    assert "A deliberately verbose skill description" not in result
    assert "skills_list" in result


def test_catalog_mode_is_part_of_cache_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills_dir = tmp_path / "skills" / "coding" / "python-debug"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\nname: python-debug\ndescription: Debug Python scripts\n---\n"
    )
    clear_skills_system_prompt_cache(clear_snapshot=True)

    full = build_skills_system_prompt(catalog_mode="full")
    compact = build_skills_system_prompt(catalog_mode="compact")

    assert "Debug Python scripts" in full
    assert "Debug Python scripts" not in compact


def test_disabled_tool_routing_only_applies_safety_denylist():
    assert select_task_toolsets(ALL_TOOLSETS, "search the web", enabled=False) == ALL_TOOLSETS - {"browser"}


def test_local_code_task_uses_lean_core_without_web_or_delegation():
    selected = select_task_toolsets(ALL_TOOLSETS, "Fix the failing Python test in this repository")
    assert {"file", "terminal", "code_execution", "skills", "jev"} <= selected
    assert "web" not in selected
    assert "delegation" not in selected
    assert "browser" not in selected


def test_current_information_adds_web():
    selected = select_task_toolsets(ALL_TOOLSETS, "What is the current weather in Clovis?")
    assert "web" in selected
    assert "browser" not in selected


def test_scheduling_adds_cronjob():
    selected = select_task_toolsets(ALL_TOOLSETS, "Remind me tomorrow at 9 AM")
    assert "cronjob" in selected


def test_smart_home_intent_preserves_homeassistant():
    selected = select_task_toolsets(ALL_TOOLSETS, "What is the current thermostat temperature?")
    assert "homeassistant" in selected
    assert "web" not in selected


def test_generic_current_local_question_does_not_force_web():
    selected = select_task_toolsets(ALL_TOOLSETS, "What is the current Python config?")
    assert "web" not in selected


def test_ambiguous_followup_keeps_all_safe_tools():
    selected = select_task_toolsets(ALL_TOOLSETS, "resume")
    assert selected == ALL_TOOLSETS - {"browser"}


def test_recent_history_preserves_task_intent():
    history = [{"role": "user", "content": "Research current Python releases"}]
    selected = select_task_toolsets(ALL_TOOLSETS, "continue", history=history)
    assert "web" in selected


def test_disabled_model_routing_keeps_primary():
    assert select_turn_model("hello", "claude-sonnet", {}, enabled=False) == "claude-sonnet"


def test_new_simple_turn_uses_cheap_model():
    cfg = {"model": "claude-haiku-4-5", "max_chars": 240}
    assert select_turn_model("Define a confidence interval", "claude-sonnet", cfg) == "claude-haiku-4-5"


def test_existing_session_stays_on_primary_to_preserve_cache_and_context():
    cfg = {"model": "claude-haiku-4-5", "max_chars": 240}
    history = [{"role": "user", "content": "Build the service"}]
    assert select_turn_model("thanks", "claude-sonnet", cfg, history=history) == "claude-sonnet"


def test_action_task_stays_on_primary():
    cfg = {"model": "claude-haiku-4-5", "max_chars": 240}
    assert select_turn_model("Deploy the server and run all tests", "claude-sonnet", cfg) == "claude-sonnet"
