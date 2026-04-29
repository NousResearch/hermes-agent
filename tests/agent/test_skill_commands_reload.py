"""Tests for ``agent.skill_commands.reload_skills`` and the ``skills_reload`` tool.

Covers the helper that powers ``/reload-skills`` (CLI + gateway slash command)
and the ``skills_reload`` agent tool — both clear in-process skill caches and
return a diff of newly-visible / removed skill names.
"""

import json
import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest


def _write_skill(skills_dir: Path, name: str, description: str = "") -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: {description or f'{name} skill'}
            ---
            body
            """
        )
    )
    return skill_dir


@pytest.fixture
def hermes_home(monkeypatch):
    """Isolate HERMES_HOME for ``reload_skills`` tests.

    Rather than popping cache-bearing modules from ``sys.modules`` (which
    races against pytest-xdist's parallel workers), we monkeypatch the
    module-level ``HERMES_HOME`` / ``SKILLS_DIR`` constants in place so the
    isolation is local to this fixture's scope.
    """
    td = tempfile.mkdtemp(prefix="hermes-reload-skills-")
    monkeypatch.setenv("HERMES_HOME", td)
    home = Path(td)
    (home / "skills").mkdir(parents=True, exist_ok=True)

    # Import lazily (inside fixture) so the modules are already resident,
    # then redirect their captured paths at the new temp dir.
    import tools.skills_tool as _st
    import agent.skill_commands as _sc

    monkeypatch.setattr(_st, "HERMES_HOME", home, raising=False)
    monkeypatch.setattr(_st, "SKILLS_DIR", home / "skills", raising=False)
    # Reset the in-process slash-command cache so each test starts from zero.
    monkeypatch.setattr(_sc, "_skill_commands", {}, raising=False)

    yield home

    shutil.rmtree(td, ignore_errors=True)


class TestReloadSkillsHelper:
    """``agent.skill_commands.reload_skills``."""

    def test_returns_expected_keys(self, hermes_home):
        from agent.skill_commands import reload_skills

        result = reload_skills()
        assert set(result) == {"added", "removed", "unchanged", "total", "commands"}
        assert result["total"] == 0
        assert result["added"] == []
        assert result["removed"] == []

    def test_detects_newly_added_skill(self, hermes_home):
        from agent.skill_commands import reload_skills, get_skill_commands

        # Prime the cache so subsequent diff is meaningful
        get_skill_commands()

        _write_skill(hermes_home / "skills", "demo")
        result = reload_skills()

        assert result["added"] == ["demo"]
        assert result["removed"] == []
        assert result["total"] == 1
        assert result["commands"] == 1

    def test_detects_removed_skill(self, hermes_home):
        from agent.skill_commands import reload_skills

        skill_dir = _write_skill(hermes_home / "skills", "demo")
        # First reload: demo present
        first = reload_skills()
        assert first["total"] == 1

        # Remove and reload
        shutil.rmtree(skill_dir)
        second = reload_skills()

        assert second["removed"] == ["demo"]
        assert second["added"] == []
        assert second["total"] == 0

    def test_clears_prompt_cache_snapshot(self, hermes_home):
        """The disk snapshot at ``.skills_prompt_snapshot.json`` must be removed."""
        from agent.prompt_builder import _skills_prompt_snapshot_path
        from agent.skill_commands import reload_skills

        snapshot = _skills_prompt_snapshot_path()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text("{}")
        assert snapshot.exists()

        reload_skills()

        assert not snapshot.exists(), "prompt cache snapshot should be removed"

    def test_unchanged_skills_appear_in_unchanged_list(self, hermes_home):
        from agent.skill_commands import reload_skills, get_skill_commands

        _write_skill(hermes_home / "skills", "alpha")
        # Prime cache
        get_skill_commands()

        # Call reload again with no FS changes
        result = reload_skills()
        assert "alpha" in result["unchanged"]
        assert result["added"] == []
        assert result["removed"] == []


class TestSkillsReloadTool:
    """``tools.skills_tool.skills_reload`` — the agent-facing tool."""

    def test_tool_returns_json(self, hermes_home):
        from tools.skills_tool import skills_reload

        out = skills_reload()
        result = json.loads(out)
        assert result["success"] is True
        assert set(result) == {
            "success",
            "added",
            "removed",
            "unchanged_count",
            "total",
            "commands",
        }

    def test_tool_reports_added_skill(self, hermes_home):
        from agent.skill_commands import get_skill_commands
        from tools.skills_tool import skills_reload

        get_skill_commands()  # prime cache
        _write_skill(hermes_home / "skills", "freshly-added", "fresh skill")

        result = json.loads(skills_reload())
        assert result["success"] is True
        assert result["added"] == ["freshly-added"]
        assert result["total"] == 1

    def test_tool_is_registered_in_skills_toolset(self, hermes_home):
        # Importing the module triggers registry.register
        import tools.skills_tool  # noqa: F401
        from tools.registry import registry

        assert "skills_reload" in registry.get_tool_names_for_toolset("skills")
        assert registry.get_toolset_for_tool("skills_reload") == "skills"

    def test_tool_schema_has_no_required_args(self, hermes_home):
        import tools.skills_tool  # noqa: F401
        from tools.registry import registry

        schema = registry.get_schema("skills_reload")
        assert schema["name"] == "skills_reload"
        # Caller invokes with no arguments; tool returns the diff verbatim.
        assert schema["parameters"].get("required", []) == []
