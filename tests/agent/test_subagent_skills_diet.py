"""A delegated subagent's ``<available_skills>`` is demoted to names-only for non-coding categories.

Measured on this host: a ``platform="subagent"`` leaf prompt was ~80.5K chars, 65.2K of them
``<available_skills>``.  A leaf worker never reads that index as prose, but it still needs every
name (``skill_view(name)`` must keep resolving), so the fix is the coding posture's demote-to-
``[names only]`` rendering applied unconditionally to subagents — never pruning.
``HERMES_SUBAGENT_FULL_SKILLS=1`` restores the full entries for anyone debugging an index.
"""

import pytest

from agent import coding_context as cc
from agent import prompt_builder
from agent import system_prompt

SKILL_TOOLS = {"skills_list", "skill_view", "skill_manage", "read_file", "terminal"}

SKILLS = (
    ("social-media", "thread-writer", "Write threads"),
    ("investment", "fiscal-terminal", "Run the fiscal terminal"),
    ("finance-imports", "fs-dcf-model", "Build a DCF in Excel"),
    ("github", "github-pr-workflow", "Ship a pull request"),
)


class _Agent:
    """The shape ``_skills_prompt`` reads: platform, tool names, and (absent) session db."""

    platform = "subagent"
    valid_tool_names = SKILL_TOOLS
    _session_db = None


@pytest.fixture(autouse=True)
def _clean_skills_cache():
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    yield
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)


@pytest.fixture()
def skills_home(tmp_path, monkeypatch):
    """A skills tree with two non-coding categories and one coding-adjacent one."""
    home = tmp_path / "home"
    for category, name, desc in SKILLS:
        skill_dir = home / "skills" / category / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n\nbody\n", encoding="utf-8"
        )
    # A code workspace, so the posture (and therefore the focus demotion) can apply here.
    (home / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_SUBAGENT_FULL_SKILLS", raising=False)
    # The production default for a delegated worker: `coding_context: auto`.  Pin it so the
    # user's own config.yaml cannot turn focus mode on/off under the test.
    monkeypatch.setattr(cc, "_coding_mode", lambda *args, **kwargs: "auto")
    return home


def _skills_block(platform: str) -> str:
    agent = type("_A", (_Agent,), {"platform": platform})
    return system_prompt._skills_prompt(agent)


def test_subagent_demotes_non_coding_categories_under_auto(skills_home):
    text = _skills_block("subagent")

    for category in ("social-media", "investment", "finance-imports"):
        assert f"{category} [names only]" in text, category

    # Demoted, never hidden: every name stays visible so skill_view(name) keeps resolving,
    # while the descriptions (the whole 65K) are gone.
    assert "thread-writer" in text
    assert "fiscal-terminal" in text
    assert "fs-dcf-model" in text
    assert "Write threads" not in text
    assert "Run the fiscal terminal" not in text
    assert "Build a DCF in Excel" not in text

    # Coding-adjacent categories keep their full entries (deny-list semantics).
    assert "github-pr-workflow: Ship a pull request" in text


def test_subagent_full_skills_opt_out(skills_home, monkeypatch):
    monkeypatch.setenv("HERMES_SUBAGENT_FULL_SKILLS", "1")

    text = _skills_block("subagent")

    assert "[names only]" not in text
    assert "Write threads" in text


def test_interactive_session_under_auto_keeps_full_entries(skills_home):
    """The demotion is a subagent diet, not an `auto` policy change: a CLI session is untouched."""
    text = _skills_block("cli")

    assert "[names only]" not in text
    assert "Write threads" in text


def test_focus_mode_still_compacts_interactive_sessions(skills_home, monkeypatch):
    """The focus gate in ``RuntimeMode.compact_skill_categories`` is what the fallback bypasses."""
    monkeypatch.setattr(cc, "_coding_mode", lambda *args, **kwargs: "focus")

    assert "social-media" in cc.coding_compact_skill_categories(platform="cli", cwd=skills_home)
    assert cc.resolve_runtime_mode(platform="cli", cwd=skills_home, config={}).config_mode == "focus"


def test_deny_list_covers_the_large_pack_categories():
    assert {"finance-imports", "investment"} <= set(cc._NON_CODING_SKILL_CATEGORIES)
    # Coding-adjacent families must stay full entries.
    for kept in ("software-development", "github", "devops", "testing"):
        assert kept not in cc._NON_CODING_SKILL_CATEGORIES


def test_subagent_is_an_interactive_coding_platform(tmp_path):
    assert "subagent" in cc.INTERACTIVE_CODING_PLATFORMS
    # …and still only adopts the posture in an actual code workspace.
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    plain = tmp_path / "plain"
    plain.mkdir()
    cfg = {"agent": {"coding_context": "auto"}}
    assert cc.is_coding_context(platform="subagent", cwd=project, config=cfg) is True
    assert cc.is_coding_context(platform="subagent", cwd=plain, config=cfg) is False
