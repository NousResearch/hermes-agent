"""The asking agent's own platform must reach skill-index filtering (#140537).

Delegate children are built with ``platform="subagent"``, but the skills-index
builder resolved its platform hint from env vars / gateway session context —
both of which only ever reflect the PARENT's channel (or nothing on a bare
CLI thread). ``skills.platform_disabled.subagent`` therefore never applied to
a child's discovery index, and frontmatter ``session_platforms:`` conditions
were evaluated against the parent's channel instead of the child's.
"""

import re

import pytest


def _skills_body(prompt: str) -> str:
    m = re.search(r"<available_skills>(.*?)</available_skills>", prompt, re.DOTALL)
    return (m.group(1).strip() if m else "")


@pytest.fixture()
def skills_home(tmp_path, monkeypatch):
    """A home with two skills and a config disabling one of them for subagents only."""
    for name in ("child-only-skill", "parent-skill"):
        skill_dir = tmp_path / "skills" / "general" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} description\n---\nbody\n",
            encoding="utf-8",
        )
    (tmp_path / "config.yaml").write_text(
        "skills:\n"
        "  platform_disabled:\n"
        "    subagent:\n"
        "      - child-only-skill\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    from agent import prompt_builder, skill_utils

    skill_utils._raw_config_cache_clear()
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=False)
    # Keep the test about platform scoping, not the plugin registry.
    monkeypatch.setattr(prompt_builder, "_plugin_skill_prompt_rows", lambda *a, **k: [])
    return tmp_path


def test_platform_hint_scopes_disabled_list(skills_home):
    """An explicit subagent hint applies skills.platform_disabled.subagent;
    without one the parent's (unfiltered) view is unchanged."""
    from agent import prompt_builder

    child = _skills_body(prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_home / "skills", platform_hint="subagent"))
    assert "child-only-skill" not in child
    assert "parent-skill" in child

    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=False)
    parent = _skills_body(prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_home / "skills"))
    assert "child-only-skill" in parent
    assert "parent-skill" in parent


def test_platform_hint_filters_plugin_registered_skills(tmp_path, monkeypatch):
    """Plugin-registered skills (ctx.register_skill — the motivating case in #140537)
    pass through the same hint-scoped gates as on-disk ones: the qualified name is
    matched against the platform's disabled list, and frontmatter session_platforms
    conditions evaluate against the asking agent's platform."""
    (tmp_path / "config.yaml").write_text(
        "skills:\n"
        "  platform_disabled:\n"
        "    subagent:\n"
        "      - plugin-x:child-only-skill\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)

    class _FakeManager:
        @staticmethod
        def list_plugin_skill_metadata():
            return [
                {"name": "plugin-x:child-only-skill", "description": "d", "frontmatter": {}},
                {"name": "plugin-x:other-skill", "description": "d", "frontmatter": {}},
                {"name": "plugin-x:telegram-skill", "description": "d",
                 "frontmatter": {"metadata": {"hermes": {"session_platforms": ["telegram"]}}}},
            ]

    import hermes_cli.plugins

    monkeypatch.setattr(hermes_cli.plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(hermes_cli.plugins, "get_plugin_manager", lambda: _FakeManager())
    (tmp_path / "skills").mkdir()
    from agent import prompt_builder, skill_utils

    skill_utils._raw_config_cache_clear()
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=False)

    child = _skills_body(prompt_builder.build_skills_system_prompt(
        skills_dir_override=tmp_path / "skills", platform_hint="subagent"))
    assert "plugin-x:child-only-skill" not in child
    assert "plugin-x:other-skill" in child
    assert "plugin-x:telegram-skill" not in child  # subagent is not in [telegram]

    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=False)
    parent = _skills_body(prompt_builder.build_skills_system_prompt(
        skills_dir_override=tmp_path / "skills"))
    assert "plugin-x:child-only-skill" in parent
    assert "plugin-x:other-skill" in parent
    assert "plugin-x:telegram-skill" in parent  # unknown platform fails open


def test_platform_hint_scopes_session_platforms_condition(skills_home, monkeypatch):
    """Frontmatter session_platforms: conditions evaluate against the hint's platform:
    a telegram-only skill is hidden from a subagent child and fails open without a hint."""
    cond_dir = skills_home / "skills" / "general" / "cond-skill"
    cond_dir.mkdir(parents=True)
    (cond_dir / "SKILL.md").write_text(
        "---\nname: cond-skill\ndescription: telegram only\n"
        "metadata:\n  hermes:\n    session_platforms:\n      - telegram\n---\nbody\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    from agent import prompt_builder

    child = _skills_body(prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_home / "skills", platform_hint="subagent"))
    assert "cond-skill" not in child

    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=False)
    parent = _skills_body(prompt_builder.build_skills_system_prompt(
        skills_dir_override=skills_home / "skills"))
    assert "cond-skill" in parent


def test_skills_prompt_forwards_agent_platform(skills_home):
    """_skills_prompt forwards the agent's own platform, so a delegate child
    (platform="subagent") gets the subagent exclusion list — not whatever
    channel the parent happened to run on."""
    from agent import system_prompt

    class _DB:
        db_path = skills_home / "state.db"

    class _Agent:
        valid_tool_names = {"skills_list", "skill_view"}
        _session_db = _DB()

        def __init__(self, platform):
            self.platform = platform

    child = _skills_body(system_prompt._skills_prompt(_Agent("subagent")))
    assert "child-only-skill" not in child
    assert "parent-skill" in child

    from agent import prompt_builder

    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=False)
    parent_channel = _skills_body(system_prompt._skills_prompt(_Agent("telegram")))
    assert "child-only-skill" in parent_channel
