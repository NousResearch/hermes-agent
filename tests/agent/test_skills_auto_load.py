"""Tests for skills.auto_load config — persistent skill pre-loading at session start.

Tests cover:
- resolve_auto_load_skills() — config reading
- build_auto_load_prompt() — prompt generation
- CLI merge with --skills
- Missing auto_load skill → warning (non-fatal)
- AIAgent._build_system_prompt injects auto_load
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── resolve_auto_load_skills ──

def test_resolved_config_includes_auto_load_default(tmp_path):
    """The canonical DEFAULT_CONFIG path exposes skills.auto_load."""
    script = textwrap.dedent(
        """
        import json
        from hermes_cli.config import load_config

        print(json.dumps(load_config()["skills"]["auto_load"]))
        """
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "profile-home")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


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
    """Loads the active config and returns empty when auto_load is absent."""
    from agent.skill_commands import resolve_auto_load_skills

    with patch("hermes_cli.config.load_config", return_value={}):
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
    assert "auto-loaded via config (skills.auto_load)" in prompt


def test_build_auto_load_prompt_substitutes_session_id(tmp_path):
    from agent.skill_commands import build_auto_load_prompt

    skill_dir = tmp_path / "session-aware"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: session-aware\ndescription: Test.\n---\n\n"
        "Session: ${HERMES_SESSION_ID}\n",
        encoding="utf-8",
    )
    config = {"skills": {"auto_load": ["session-aware"]}}

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        prompt, loaded, missing = build_auto_load_prompt(
            task_id="sess-real-123",
            user_config=config,
        )

    assert loaded == ["session-aware"]
    assert missing == []
    assert "sess-real-123" in prompt
    assert "${HERMES_SESSION_ID}" not in prompt


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
    assert "auto-loaded via config (skills.auto_load)" in prompt


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


def test_preloaded_overlap_counts_as_successful_resolution(tmp_path):
    """An already auto-loaded canonical skill is resolved but not injected twice."""
    from agent.skill_commands import build_preloaded_skills_prompt

    loaded_payload = ({"content": "skill body"}, tmp_path, "shared-skill")
    with (
        patch(
            "agent.skill_commands._load_skill_payload",
            side_effect=[loaded_payload, None],
        ),
        patch("agent.skill_utils.get_disabled_skill_names", return_value=set()),
    ):
        prompt, loaded, missing = build_preloaded_skills_prompt(
            ["shared-alias", "typo"],
            excluded_loaded_names={"shared-skill"},
        )

    assert prompt == ""
    assert loaded == ["shared-skill"]
    assert missing == ["typo"]


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
    auto_calls = []

    def fake_auto_load(**kwargs):
        auto_calls.append(kwargs)
        return "auto prompt", list(auto_load), []

    _real_cli_cls = cli_mod.HermesCLI
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kw: _DummyCLI(**kw))
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    import agent.skill_commands as sc_mod
    monkeypatch.setattr(
        sc_mod,
        "build_auto_load_prompt",
        fake_auto_load,
    )
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None, excluded_loaded_names=None: (
            "prompt", sorted(skills), [],
        ),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="cli-skill", list_tools=True)

    cli_obj = created["cli"]
    # main() backgrounds the --skills load; the display merge happens when
    # the preload is finalized (normally at agent init). Run the real
    # finalize logic against the dummy to join the thread and fold in names.
    _real_cli_cls.finalize_preloaded_skills(cli_obj)
    assert "auto-skill" in cli_obj.preloaded_skills
    assert "cli-skill" in cli_obj.preloaded_skills
    assert auto_calls == [{"task_id": "sess-001", "user_config": {}}]


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

    _real_cli_cls = cli_mod.HermesCLI
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kw: _DummyCLI(**kw))
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    import agent.skill_commands as sc_mod
    monkeypatch.setattr(
        sc_mod,
        "build_auto_load_prompt",
        lambda **kwargs: ("auto prompt", ["shared-skill"], []),
    )
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None, excluded_loaded_names=None: (
            "", ["shared-skill"], ["typo"],
        ),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="shared-skill,typo", list_tools=True)

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

    _real_cli_cls = cli_mod.HermesCLI
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kw: _DummyCLI(**kw))
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    import agent.skill_commands as sc_mod
    monkeypatch.setattr(
        sc_mod,
        "build_auto_load_prompt",
        lambda **kwargs: ("", [], ["valid-cli-skill"]),
    )
    # CLI should only build prompts for --skills entries, not auto_load.
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None, excluded_loaded_names=None: (
            "prompt", list(skills), []
        ),
    )

    # Should NOT raise ValueError — missing auto_load is handled in AIAgent layer
    with pytest.raises(SystemExit):
        cli_mod.main(skills="valid-cli-skill", list_tools=True)

    cli_obj = created["cli"]
    # Join the backgrounded --skills load; must not raise for auto_load misses.
    _real_cli_cls.finalize_preloaded_skills(cli_obj)
    # Only successfully loaded canonical names appear in the display.
    assert "valid-cli-skill" in cli_obj.preloaded_skills


def test_cli_still_errors_for_missing_cli_skills(monkeypatch):
    """Missing --skills still produce a ValueError (not auto_load)."""
    import cli as cli_mod

    created = {}

    class _DummyCLI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = "sess-004"
            self.system_prompt = "base"
            self.preloaded_skills = []
            created["cli"] = self

        def show_banner(self): pass
        def show_tools(self): pass
        def show_toolsets(self): pass
        def run(self): pass

    _real_cli_cls = cli_mod.HermesCLI
    monkeypatch.setattr(cli_mod, "HermesCLI", lambda **kw: _DummyCLI(**kw))
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    import agent.skill_commands as sc_mod
    monkeypatch.setattr(
        sc_mod, "build_auto_load_prompt", lambda **kwargs: ("", [], [])
    )
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None, excluded_loaded_names=None: (
            "", [], ["missing-cli"]
        ),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="missing-cli", list_tools=True)

    # The backgrounded load surfaces the hard failure at finalize time
    # (normally agent init), matching the synchronous path's old contract.
    with pytest.raises(ValueError, match=r"Unknown skill\(s\): missing-cli"):
        _real_cli_cls.finalize_preloaded_skills(created["cli"])


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
         patch("agent.process_bootstrap._install_safe_stdio"):

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

    assert "auto-loaded via config (skills.auto_load)" in prompt
    assert "buildsys-skill" in prompt


def test_aiagent_build_system_prompt_no_auto_load_when_empty(tmp_path):
    """When auto_load is empty, no injection happens."""
    from run_agent import AIAgent

    with patch("agent.skill_commands.resolve_auto_load_skills", return_value=[]), \
         patch("run_agent.AIAgent._ensure_db_session"), \
         patch("agent.process_bootstrap._install_safe_stdio"):

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

    assert "auto-loaded via config (skills.auto_load)" not in prompt


def test_aiagent_build_system_prompt_survives_config_errors(tmp_path):
    """Config read errors in auto_load are non-fatal."""
    from run_agent import AIAgent

    with patch(
        "agent.skill_commands.resolve_auto_load_skills",
        side_effect=RuntimeError("config read failed"),
    ), \
         patch("run_agent.AIAgent._ensure_db_session"), \
         patch("agent.process_bootstrap._install_safe_stdio"):

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


def test_auto_load_disabled_skill_is_reported_not_loaded(tmp_path, monkeypatch):
    """Disabled auto-load entries use the same gate as explicit preloads."""
    from agent.skill_commands import build_auto_load_prompt
    import agent.skill_utils as skill_utils

    skill_dir = tmp_path / "disabled-auto"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: disabled-auto\ndescription: Disabled.\n---\n\nSECRET BODY\n"
    )
    monkeypatch.setattr(
        skill_utils, "get_disabled_skill_names", lambda platform=None: {"disabled-auto"}
    )

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        prompt, loaded, missing = build_auto_load_prompt(
            user_config={"skills": {"auto_load": ["disabled-auto"]}}
        )

    assert (prompt, loaded, missing) == ("", [], ["disabled-auto"])


def test_auto_load_deduplicates_successful_canonical_names(monkeypatch):
    """Different lookup forms resolving to one skill produce one prompt block."""
    from pathlib import Path
    import agent.skill_commands as skill_commands
    import agent.skill_utils as skill_utils

    payload = {"success": True, "name": "canonical", "content": "BODY"}
    monkeypatch.setattr(
        skill_commands,
        "_load_skill_payload",
        lambda identifier, task_id=None: (payload, Path("/tmp/canonical"), "canonical"),
    )
    monkeypatch.setattr(
        skill_utils, "get_disabled_skill_names", lambda platform=None: set()
    )

    prompt, loaded, missing = skill_commands.build_auto_load_prompt(
        user_config={"skills": {"auto_load": ["alias", "/absolute/alias"]}}
    )

    assert loaded == ["canonical"]
    assert missing == []
    assert prompt.count('The "canonical" skill is auto-loaded') == 1


def test_real_profile_config_and_shared_prompt_are_lifecycle_stable(tmp_path):
    """Exercise canonical config + skill I/O through the gateway/TUI prompt path.

    A subprocess gives imports a genuinely isolated HERMES_HOME, avoiding module
    constant/config-cache monkeypatches that previously hid the broken loader.
    """
    script = textwrap.dedent(
        r"""
        import json
        import os
        from pathlib import Path
        from run_agent import AIAgent

        home = Path(os.environ["HERMES_HOME"])
        skills = home / "skills"
        skills.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("skills:\n  auto_load:\n    - stable-skill\n")
        skill_dir = skills / "stable-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\nname: stable-skill\ndescription: Stable.\n---\n\nORIGINAL SKILL BYTES\n"
        )

        def make_agent(platform):
            agent = AIAgent.__new__(AIAgent)
            agent.valid_tool_names = {"skills_list", "skill_view", "skill_manage"}
            agent.model = "before-model"
            agent.provider = "test"
            agent.pass_session_id = False
            agent.skip_context_files = True
            agent.load_soul_identity = False
            agent._memory_enabled = False
            agent._user_profile_enabled = False
            agent._memory_manager = None
            agent._memory_store = None
            agent.session_id = "shared-" + platform
            agent.platform = platform
            agent._tool_use_enforcement = False
            agent.ephemeral_system_prompt = None
            agent._cached_system_prompt = None
            agent._auto_load_skills_resolved = False
            agent._auto_load_skills_result = ("", [], [])
            return agent

        outputs = {}
        for platform in ("telegram", "tui"):
            agent = make_agent(platform)
            first = agent._build_system_prompt()
            outputs[platform] = {
                "loaded": agent._auto_load_skills_result[1],
                "has_original": "ORIGINAL SKILL BYTES" in first,
            }
            if platform == "telegram":
                # Simulate both file/config mutation and model-switch cache
                # invalidation. Rebuild must retain the first resolved bytes.
                cached_auto_prompt = agent._auto_load_skills_result[0]
                skill_file.write_text(
                    "---\nname: stable-skill\ndescription: Stable.\n---\n\nMUTATED BYTES\n"
                )
                (home / "config.yaml").write_text("skills:\n  auto_load: []\n")
                agent.model = "after-model"
                agent._cached_system_prompt = None
                rebuilt = agent._build_system_prompt()
                outputs[platform].update({
                    "stable": (
                        agent._auto_load_skills_result[0] == cached_auto_prompt
                        and "ORIGINAL SKILL BYTES" in rebuilt
                        and "MUTATED BYTES" not in rebuilt
                    ),
                    "resolved": agent._auto_load_skills_resolved,
                })
                # Restore disk for the separate TUI agent, proving each agent
                # resolves independently through the same shared path.
                (home / "config.yaml").write_text(
                    "skills:\n  auto_load:\n    - stable-skill\n"
                )
                skill_file.write_text(
                    "---\nname: stable-skill\ndescription: Stable.\n---\n\nORIGINAL SKILL BYTES\n"
                )

        print(json.dumps(outputs))
        """
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "profile-home")
    env.pop("HERMES_IGNORE_RULES", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip().splitlines()[-1])

    assert data["telegram"] == {
        "loaded": ["stable-skill"],
        "has_original": True,
        "stable": True,
        "resolved": True,
    }
    assert data["tui"] == {
        "loaded": ["stable-skill"],
        "has_original": True,
    }


def test_ignore_rules_is_stable_for_agent_lifetime(monkeypatch):
    """Ignoring rules resolves to an empty immutable lifecycle result."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._auto_load_skills_resolved = False
    agent._auto_load_skills_result = ("", [], [])
    agent.valid_tool_names = set()
    agent.model = "test"
    agent.provider = "test"
    agent.pass_session_id = False
    agent.skip_context_files = True
    agent.load_soul_identity = False
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_manager = None
    agent._memory_store = None
    agent.session_id = "ignore-rules"
    agent.platform = "gateway"
    agent._tool_use_enforcement = False
    agent.ephemeral_system_prompt = None
    agent._cached_system_prompt = None
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")

    first = agent._build_system_prompt()
    monkeypatch.delenv("HERMES_IGNORE_RULES")
    agent._cached_system_prompt = None
    rebuilt = agent._build_system_prompt()

    assert agent._auto_load_skills_resolved is True
    assert agent._auto_load_skills_result == ("", [], [])
    assert rebuilt == first
