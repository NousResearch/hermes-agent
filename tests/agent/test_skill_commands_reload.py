"""Tests for ``agent.skill_commands.reload_skills``.

Covers the helper that powers ``/reload-skills`` (CLI + gateway slash command).
The helper rescans the skills directory and returns a diff of what changed.
It does NOT invalidate the skills system-prompt cache — skills are invoked
at runtime via ``/skill-name``, ``skills_list``, or ``skill_view`` and don't
need to live in the system prompt.

``added`` and ``removed`` are lists of ``{"name": str, "description": str}``
dicts. Descriptions are truncated to 60 chars.
"""

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

    Rather than popping cache-bearing modules from ``sys.modules``,
    we monkeypatch the module-level ``HERMES_HOME`` / ``SKILLS_DIR``
    constants in place so the isolation is local to this fixture's scope.
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



    def test_detects_removed_skill_carries_description(self, hermes_home):
        from agent.skill_commands import reload_skills

        skill_dir = _write_skill(hermes_home / "skills", "demo", "soon to be gone")
        # First reload: demo present
        first = reload_skills()
        assert first["total"] == 1
        assert first["added"] == [{"name": "demo", "description": "soon to be gone"}]

        # Remove and reload — the description must survive the removal diff
        # (we cached it from the pre-rescan snapshot).
        shutil.rmtree(skill_dir)
        second = reload_skills()

        assert second["removed"] == [{"name": "demo", "description": "soon to be gone"}]
        assert second["added"] == []
        assert second["total"] == 0



    def test_does_not_invalidate_prompt_cache_snapshot(self, hermes_home):
        """reload_skills must NOT delete the skills prompt-cache snapshot.

        Skills are called at runtime — the system prompt doesn't need to
        mention them for the model to use them — so reloading them should
        preserve prefix caching.
        """
        from agent.prompt_builder import _skills_prompt_snapshot_path
        from agent.skill_commands import reload_skills

        snapshot = _skills_prompt_snapshot_path()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text("{}")
        assert snapshot.exists()

        reload_skills()

        assert snapshot.exists(), (
            "prompt cache snapshot should be preserved — skills don't live "
            "in the system prompt so there's no reason to invalidate it"
        )
