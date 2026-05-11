"""Tests for agent.skill_trigger_loader."""
from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(tmp_path: Path, name: str, triggers: list[str], body: str = "skill body") -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = "---\nname: {}\ndescription: test skill\ntriggers:\n".format(name)
    for t in triggers:
        frontmatter += f"  - '{t}'\n"
    frontmatter += "---\n"
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(frontmatter + body)
    return skill_file


def _make_skill_no_triggers(tmp_path: Path, name: str) -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = "---\nname: {}\ndescription: no triggers here\n---\nbody\n".format(name)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(content)
    return skill_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the module-level cache before each test."""
    from agent import skill_trigger_loader
    skill_trigger_loader._trigger_cache.clear()
    yield
    skill_trigger_loader._trigger_cache.clear()


# ---------------------------------------------------------------------------
# Tests: get_triggered_skills
# ---------------------------------------------------------------------------

def test_returns_empty_for_blank_input(tmp_path):
    from agent import skill_trigger_loader
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
        assert skill_trigger_loader.get_triggered_skills("") == []
        assert skill_trigger_loader.get_triggered_skills("   ") == []


def test_matches_simple_string_pattern(tmp_path):
    _make_skill(tmp_path, "my-skill", ["PHP Parse error"], "## fix it")
    from agent import skill_trigger_loader
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
        result = skill_trigger_loader.get_triggered_skills("PHP Parse error: syntax error")
    assert len(result) == 1
    assert "fix it" in result[0]


def test_match_is_case_insensitive(tmp_path):
    _make_skill(tmp_path, "s", ["php parse error"])
    from agent import skill_trigger_loader
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
        result = skill_trigger_loader.get_triggered_skills("PHP PARSE ERROR in controller")
    assert len(result) == 1


def test_no_match_returns_empty(tmp_path):
    _make_skill(tmp_path, "s", ["PHP Parse error"])
    from agent import skill_trigger_loader
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
        result = skill_trigger_loader.get_triggered_skills("everything is fine")
    assert result == []


def test_skill_without_triggers_is_skipped(tmp_path):
    _make_skill_no_triggers(tmp_path, "no-trigger-skill")
    from agent import skill_trigger_loader
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
        result = skill_trigger_loader.get_triggered_skills("any text at all")
    assert result == []


def test_multiple_skills_can_match(tmp_path):
    _make_skill(tmp_path, "skill-a", ["parse error"], "body A")
    _make_skill(tmp_path, "skill-b", ["syntax error"], "body B")
    from agent import skill_trigger_loader
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
        result = skill_trigger_loader.get_triggered_skills("parse error and syntax error both")
    assert len(result) == 2


def test_max_skills_cap(tmp_path):
    from agent import skill_trigger_loader
    original_max = skill_trigger_loader.MAX_AUTO_LOADED_SKILLS
    skill_trigger_loader.MAX_AUTO_LOADED_SKILLS = 2
    try:
        for i in range(5):
            _make_skill(tmp_path, f"skill-{i}", ["match me"], f"body {i}")
        with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
            result = skill_trigger_loader.get_triggered_skills("match me please")
        assert len(result) <= 2
    finally:
        skill_trigger_loader.MAX_AUTO_LOADED_SKILLS = original_max


def test_each_skill_loaded_only_once_per_match(tmp_path):
    """A skill with multiple matching patterns should only be included once."""
    _make_skill(tmp_path, "s", ["error one", "error two"], "skill body")
    from agent import skill_trigger_loader
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
        result = skill_trigger_loader.get_triggered_skills("error one and error two")
    assert len(result) == 1


def test_invalid_regex_pattern_is_skipped_gracefully(tmp_path):
    """An invalid regex in frontmatter must not crash — just skip that pattern."""
    _make_skill(tmp_path, "bad-regex", ["[invalid(regex"], "body")
    from agent import skill_trigger_loader
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
        # Should not raise
        result = skill_trigger_loader.get_triggered_skills("[invalid(regex — matching text")
    assert isinstance(result, list)


def test_nonexistent_skills_home_returns_empty(tmp_path):
    from agent import skill_trigger_loader
    missing = tmp_path / "does_not_exist"
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=missing):
        result = skill_trigger_loader.get_triggered_skills("anything")
    assert result == []


def test_mtime_cache_is_used(tmp_path):
    """Second call with same mtime should not re-read the file."""
    skill_file = _make_skill(tmp_path, "cached", ["cached pattern"], "cached body")
    from agent import skill_trigger_loader
    with mock.patch.object(skill_trigger_loader, "_get_skills_home", return_value=tmp_path):
        r1 = skill_trigger_loader.get_triggered_skills("cached pattern")
        # Overwrite content without changing mtime (force same mtime)
        mtime = skill_file.stat().st_mtime
        skill_file.write_text("---\nname: cached\ntriggers:\n  - 'cached pattern'\n---\nnew body\n")
        import os; os.utime(skill_file, (mtime, mtime))
        r2 = skill_trigger_loader.get_triggered_skills("cached pattern")
    # Both results should have the original content (cache hit)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: format_triggered_skills_block
# ---------------------------------------------------------------------------

def test_format_empty_list():
    from agent.skill_trigger_loader import format_triggered_skills_block
    assert format_triggered_skills_block([]) == ""


def test_format_single_skill():
    from agent.skill_trigger_loader import format_triggered_skills_block
    result = format_triggered_skills_block(["# My Skill\n\nDo this."])
    assert "Auto-loaded skill 1" in result
    assert "My Skill" in result
    assert "Do this." in result


def test_format_multiple_skills():
    from agent.skill_trigger_loader import format_triggered_skills_block
    result = format_triggered_skills_block(["skill A content", "skill B content"])
    assert "Auto-loaded skill 1" in result
    assert "Auto-loaded skill 2" in result
    assert "skill A content" in result
    assert "skill B content" in result
