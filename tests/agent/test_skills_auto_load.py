"""Tests for skills.auto_load config — persistent skill pre-loading at session start.

Tests cover:
- resolve_auto_load_skills() — config reading
- build_auto_load_prompt() — prompt generation
- CLI merge with --skills
- Missing auto_load skill → warning (non-fatal)
- AIAgent._build_system_prompt injects auto_load
"""

from unittest.mock import MagicMock, patch

import pytest


# ── resolve_auto_load_skills ──

def test_resolve_auto_load_skills_reads_from_config():
    """resolve_auto_load_skills reads the auto_load list from user config."""
    from agent.skill_commands import resolve_auto_load_skills

    config = {
        "skills": {
            "auto_load": ["skill-a", "skill-b", "skill-c"],
        }
    }
    result = resolve_auto_load_skills(config)
    assert result == ["skill-a", "skill-b", "skill-c"]


def test_resolve_auto_load_skills_empty_list():
    """Returns empty list when auto_load is empty."""
    from agent.skill_commands import resolve_auto_load_skills

    config = {"skills": {"auto_load": []}}
    result = resolve_auto_load_skills(config)
    assert result == []


def test_resolve_auto_load_skills_missing_key():
    """Returns empty list when auto_load key is missing."""
    from agent.skill_commands import resolve_auto_load_skills

    config = {"skills": {}}
    result = resolve_auto_load_skills(config)
    assert result == []


def test_resolve_auto_load_skills_no_config():
    """Returns empty list when config is None."""
    from agent.skill_commands import resolve_auto_load_skills

    result = resolve_auto_load_skills(None)
    assert result == []


def test_resolve_auto_load_skills_deduplicates():
    """Duplicate entries are deduplicated (first occurrence wins)."""
    from agent.skill_commands import resolve_auto_load_skills

    config = {"skills": {"auto_load": ["skill-a", "skill-b", "skill-a"]}}
    result = resolve_auto_load_skills(config)
    assert result == ["skill-a", "skill-b"]


def test_resolve_auto_load_skills_filters_non_strings():
    """Non-string entries are filtered out."""
    from agent.skill_commands import resolve_auto_load_skills

    config = {"skills": {"auto_load": ["skill-a", 123, None, "", "skill-b"]}}
    result = resolve_auto_load_skills(config)
    assert result == ["skill-a", "skill-b"]


def test_resolve_auto_load_skills_strips_whitespace():
    """Whitespace is stripped from skill names."""
    from agent.skill_commands import resolve_auto_load_skills

    config = {"skills": {"auto_load": ["  skill-a  ", "skill-b"]}}
    result = resolve_auto_load_skills(config)
    assert result == ["skill-a", "skill-b"]


def test_resolve_auto_load_skills_not_a_list():
    """Returns empty list when auto_load is not a list."""
    from agent.skill_commands import resolve_auto_load_skills

    config = {"skills": {"auto_load": "not-a-list"}}
    result = resolve_auto_load_skills(config)
    assert result == []


# ── build_auto_load_prompt ──

def test_build_auto_load_prompt_loads_skills(tmp_path):
    """build_auto_load_prompt loads skills from config and builds prompt."""
    from agent.skill_commands import build_auto_load_prompt

    # Create a real skill
    skill_dir = tmp_path / "test-auto"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-auto\ndescription: Auto-load test.\n---\n\n# Test Auto\n\nContent.\n"
    )

    config = {"skills": {"auto_load": ["test-auto"]}}

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        prompt, loaded, missing = build_auto_load_prompt(user_config=config)

    assert missing == []
    assert loaded == ["test-auto"]
    assert "test-auto" in prompt
    assert "auto-loaded from skills.auto_load config" in prompt


def test_build_auto_load_prompt_activation_note_not_cli_specific(tmp_path):
    """The activation note is origin-agnostic, not CLI-specific."""
    from agent.skill_commands import build_auto_load_prompt

    skill_dir = tmp_path / "test-auto"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-auto\ndescription: Test.\n---\n\n# Test\n\nContent.\n"
    )

    config = {"skills": {"auto_load": ["test-auto"]}}

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        prompt, _, _ = build_auto_load_prompt(user_config=config)

    # Must NOT contain the CLI-specific phrase
    assert "launched this CLI session" not in prompt
    # Must contain the auto_load-specific phrase
    assert "auto-loaded from skills.auto_load config" in prompt


def test_build_auto_load_prompt_reports_missing_non_fatal(tmp_path):
    """Missing auto_load skills are reported in missing, not raised."""
    from agent.skill_commands import build_auto_load_prompt

    config = {"skills": {"auto_load": ["missing-skill"]}}

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        prompt, loaded, missing = build_auto_load_prompt(user_config=config)

    assert missing == ["missing-skill"]
    assert loaded == []
    assert prompt == ""


def test_build_auto_load_prompt_empty_config():
    """Returns empty when no auto_load skills configured."""
    from agent.skill_commands import build_auto_load_prompt

    config = {"skills": {"auto_load": []}}
    prompt, loaded, missing = build_auto_load_prompt(user_config=config)

    assert prompt == ""
    assert loaded == []
    assert missing == []


# ── CLI merge with --skills ──

def test_cli_merges_auto_load_with_cli_skills(monkeypatch):
    """CLI main() displays auto_load skills alongside --skills in 'Activated skills'.

    Functional injection of auto_load happens in AIAgent (new-session gated);
    the CLI display should still reflect both sources.
    """
    import cli as cli_mod

    created = {}

    class _DummyCLI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = "sess-001"
            self.system_prompt = "base"
            self.preloaded_skills = []
            created["cli"] = self

        def show_banner(self): pass
        def show_tools(self): pass
        def show_toolsets(self): pass
        def run(self): pass

    auto_load = ["auto-skill"]

    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kw: _DummyCLI(**kw))
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    import agent.skill_commands as sc_mod
    monkeypatch.setattr(sc_mod, "resolve_auto_load_skills", lambda config: list(auto_load))
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: (
            "prompt", sorted(skills), [],
        ),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="cli-skill", list_tools=True)

    cli_obj = created["cli"]
    assert "auto-skill" in cli_obj.preloaded_skills
    assert "cli-skill" in cli_obj.preloaded_skills


def test_cli_deduplicates_overlapping_skills(monkeypatch):
    """When auto_load and --skills share a skill, it loads only once."""
    import cli as cli_mod

    created = {}

    class _DummyCLI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = "sess-002"
            self.system_prompt = "base"
            self.preloaded_skills = []
            created["cli"] = self

        def show_banner(self): pass
        def show_tools(self): pass
        def show_toolsets(self): pass
        def run(self): pass

    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kw: _DummyCLI(**kw))
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    import agent.skill_commands as sc_mod
    monkeypatch.setattr(sc_mod, "resolve_auto_load_skills", lambda config: ["shared-skill"])
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: (
            "prompt", skills, [],
        ),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="shared-skill", list_tools=True)

    cli_obj = created["cli"]
    # shared-skill should appear only once
    assert cli_obj.preloaded_skills.count("shared-skill") == 1


def test_cli_does_not_error_on_missing_auto_load_skills(monkeypatch):
    """CLI should not raise for missing auto_load skills.

    Auto_load skills are injected later by AIAgent; missing ones are reported
    as a warning by build_auto_load_prompt (covered by
    test_build_auto_load_prompt_reports_missing_non_fatal). The CLI must only
    validate skills it actually injects itself (--skills).
    """
    import cli as cli_mod

    created = {}

    class _DummyCLI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = "sess-003"
            self.system_prompt = "base"
            self.preloaded_skills = []
            created["cli"] = self

        def show_banner(self): pass
        def show_tools(self): pass
        def show_toolsets(self): pass
        def run(self): pass

    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kw: _DummyCLI(**kw))
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    import agent.skill_commands as sc_mod
    monkeypatch.setattr(sc_mod, "resolve_auto_load_skills", lambda config: ["missing-auto"])
    # CLI should only build prompts for --skills entries, not auto_load.
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("prompt", list(skills), []),
    )

    # Should NOT raise ValueError — missing auto_load is handled in AIAgent layer
    with pytest.raises(SystemExit):
        cli_mod.main(skills="valid-cli-skill", list_tools=True)

    cli_obj = created["cli"]
    # Both still appear in display
    assert "missing-auto" in cli_obj.preloaded_skills
    assert "valid-cli-skill" in cli_obj.preloaded_skills


def test_cli_still_errors_for_missing_cli_skills(monkeypatch):
    """Missing --skills still produce a ValueError (not auto_load)."""
    import cli as cli_mod

    class _DummyCLI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = "sess-004"
            self.system_prompt = "base"
            self.preloaded_skills = []

        def show_banner(self): pass
        def show_tools(self): pass
        def show_toolsets(self): pass
        def run(self): pass

    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kw: _DummyCLI(**kw))
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    import agent.skill_commands as sc_mod
    monkeypatch.setattr(sc_mod, "resolve_auto_load_skills", lambda config: [])
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("", [], ["missing-cli"]),
    )

    with pytest.raises(ValueError, match=r"Unknown skill\(s\): missing-cli"):
        cli_mod.main(skills="missing-cli", list_tools=True)


# ── AIAgent._build_system_prompt ──

def test_aiagent_build_system_prompt_injects_auto_load(tmp_path):
    """AIAgent._build_system_prompt() includes auto_load skills."""
    from run_agent import AIAgent

    # Create a real skill for auto_load
    skill_dir = tmp_path / "buildsys-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: buildsys-skill\ndescription: Test.\n---\n\n# Buildsys\n\nContent.\n"
    )

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_commands.resolve_auto_load_skills", return_value=["buildsys-skill"]), \
         patch("run_agent.AIAgent._ensure_db_session"), \
         patch("run_agent._install_safe_stdio"):

        agent = AIAgent.__new__(AIAgent)
        agent.valid_tool_names = {"skills_list", "skill_view", "skill_manage"}
        agent.model = "test-model"
        agent.provider = "test"
        agent.pass_session_id = False
        agent.skip_context_files = True  # avoid upstream context_parts init-order bug
        agent.load_soul_identity = False
        agent._memory_enabled = False
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._memory_store = None
        agent.session_id = "test-session"
        agent.platform = "cli"
        agent._tool_use_enforcement = False
        agent.ephemeral_system_prompt = None
        agent._cached_system_prompt = None

        prompt = agent._build_system_prompt()

    assert "auto-loaded from skills.auto_load config" in prompt
    assert "buildsys-skill" in prompt


def test_aiagent_build_system_prompt_no_auto_load_when_empty(tmp_path):
    """When auto_load is empty, no injection happens."""
    from run_agent import AIAgent

    with patch("agent.skill_commands.resolve_auto_load_skills", return_value=[]), \
         patch("run_agent.AIAgent._ensure_db_session"), \
         patch("run_agent._install_safe_stdio"):

        agent = AIAgent.__new__(AIAgent)
        agent.valid_tool_names = {"skills_list", "skill_view", "skill_manage"}
        agent.model = "test-model"
        agent.provider = "test"
        agent.pass_session_id = False
        agent.skip_context_files = True  # avoid upstream context_parts init-order bug
        agent.load_soul_identity = False
        agent._memory_enabled = False
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._memory_store = None
        agent.session_id = "test-session"
        agent.platform = "cli"
        agent._tool_use_enforcement = False
        agent.ephemeral_system_prompt = None
        agent._cached_system_prompt = None

        prompt = agent._build_system_prompt()

    assert "auto-loaded from skills.auto_load config" not in prompt


def test_aiagent_build_system_prompt_survives_config_errors(tmp_path):
    """Config read errors in auto_load are non-fatal."""
    from run_agent import AIAgent

    with patch(
        "agent.skill_commands.resolve_auto_load_skills",
        side_effect=RuntimeError("config read failed"),
    ), \
         patch("run_agent.AIAgent._ensure_db_session"), \
         patch("run_agent._install_safe_stdio"):

        agent = AIAgent.__new__(AIAgent)
        agent.valid_tool_names = {"skills_list", "skill_view", "skill_manage"}
        agent.model = "test-model"
        agent.provider = "test"
        agent.pass_session_id = False
        agent.skip_context_files = True  # avoid upstream context_parts init-order bug
        agent.load_soul_identity = False
        agent._memory_enabled = False
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._memory_store = None
        agent.session_id = "test-session"
        agent.platform = "cli"
        agent._tool_use_enforcement = False
        agent.ephemeral_system_prompt = None
        agent._cached_system_prompt = None

        # Should NOT raise
        prompt = agent._build_system_prompt()

    assert "auto-loaded" not in prompt
