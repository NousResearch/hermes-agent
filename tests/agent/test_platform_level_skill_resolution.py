"""Tests for ADR-0001: platform-level skill resolution in profile mode.

These tests pin down the contract that ``get_all_skills_dirs()`` (and the
``SKILLS_DIR + platform-dir + get_external_skills_dirs()`` composition in
``tools.skills_tool`` / ``agent.skill_commands``) auto-includes the
platform-level skills dir when running in profile mode.

Profile mode is when ``HERMES_HOME = ~/.hermes/profiles/<name>`` (or any
custom deployment where the default-hermes-root walk-up is non-trivial).
In default mode ``HERMES_HOME = ~/.hermes`` and the platform dir IS the
local dir, so it is correctly NOT added twice.

Without ADR-0001, dispatcher-spawned workers (whose HERMES_HOME is pinned
to the assignee profile) cannot see skills installed at the platform level
under ``~/.hermes/skills/`` — see kanban t_36a73fcc (totum-orchestrator
crash on missing `totum-platform`).
"""

from unittest.mock import patch

import pytest


# ── get_all_skills_dirs() ──────────────────────────────────────────────


class TestGetAllSkillsDirsPlatformWalkUp:
    """get_all_skills_dirs() must include the platform skills dir in profile mode."""

    def test_default_mode_does_not_duplicate_platform_dir(self, tmp_path, monkeypatch):
        """In default mode (HERMES_HOME = ~/.hermes), the platform dir IS the
        local dir and must not be added twice."""
        from agent.skill_utils import get_all_skills_dirs

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "skills").mkdir()
        # Default mode: HERMES_HOME is the platform root itself.
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr("agent.skill_utils.get_skills_dir", lambda: hermes_home / "skills")

        dirs = list(get_all_skills_dirs())
        resolved = [d.resolve() for d in dirs]
        # The platform dir should appear once, not twice.
        assert resolved.count((hermes_home / "skills").resolve()) == 1

    def test_profile_mode_includes_platform_skills_dir(self, tmp_path, monkeypatch):
        """In profile mode (HERMES_HOME = ~/.hermes/profiles/<name>), the
        platform-level ~/.hermes/skills/ must be auto-included in addition
        to the profile's own skills/."""
        from agent.skill_utils import get_all_skills_dirs

        platform_root = tmp_path / ".hermes"
        profile_dir = platform_root / "profiles" / "demo"
        profile_skills = profile_dir / "skills"
        platform_skills = platform_root / "skills"
        profile_skills.mkdir(parents=True)
        platform_skills.mkdir(parents=True)

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setattr("agent.skill_utils.get_skills_dir", lambda: profile_skills)

        dirs = list(get_all_skills_dirs())
        resolved = [d.resolve() for d in dirs]
        assert profile_skills.resolve() in resolved
        assert platform_skills.resolve() in resolved

    def test_profile_mode_skips_platform_skills_if_missing(self, tmp_path, monkeypatch):
        """When the platform skills dir does not exist on disk (brand-new
        install), get_all_skills_dirs() must not error and must not include
        a non-existent path that callers would then have to filter."""
        from agent.skill_utils import get_all_skills_dirs

        platform_root = tmp_path / ".hermes"
        profile_dir = platform_root / "profiles" / "demo"
        profile_skills = profile_dir / "skills"
        profile_skills.mkdir(parents=True)
        # NOTE: platform_root/skills is intentionally NOT created.

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setattr("agent.skill_utils.get_skills_dir", lambda: profile_skills)

        dirs = list(get_all_skills_dirs())
        assert profile_skills in dirs
        # The non-existent platform dir must not be appended.
        assert (platform_root / "skills") not in dirs

    def test_profile_mode_dedups_when_platform_equals_local(self, tmp_path, monkeypatch):
        """Defensive dedup: if for some reason the local and platform resolve
        to the same path (e.g. user-mode with HERMES_HOME pointing at the
        default profile layout), the platform dir must not be appended
        twice."""
        from agent.skill_utils import get_all_skills_dirs

        hermes_home = tmp_path / ".hermes"
        skills = hermes_home / "skills"
        skills.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr("agent.skill_utils.get_skills_dir", lambda: skills)

        dirs = list(get_all_skills_dirs())
        resolved = [d.resolve() for d in dirs]
        assert resolved.count(skills.resolve()) == 1


# ── tools.skills_tool — skill_view and _find_all_skills ───────────────


class TestSkillViewPlatformWalkUp:
    """skill_view() must see platform-level skill stubs in profile mode.

    This is the dispatcher code path: kanban spawns ``hermes -p <assignee>
    --skills X chat -q "..."`` and the worker subprocess resolves skill
    ``X`` via ``skill_view`` → ``build_preloaded_skills_prompt``. Without
    the platform-dir walk-up, profile-mode workers raise
    ``ValueError("Unknown skill(s): X")`` for skills installed at the
    platform level only.
    """

    def test_skill_view_resolves_platform_skill_in_profile_mode(
        self, tmp_path, monkeypatch
    ):
        from tools.skills_tool import skill_view

        platform_root = tmp_path / ".hermes"
        profile_dir = platform_root / "profiles" / "demo"
        profile_skills = profile_dir / "skills"
        platform_skills = platform_root / "skills"
        profile_skills.mkdir(parents=True)
        platform_skills.mkdir(parents=True)

        platform_skill = platform_skills / "totum-platform-audit"
        platform_skill.mkdir()
        (platform_skill / "SKILL.md").write_text(
            "---\nname: totum-platform-audit\n"
            "description: platform-level stub\n"
            "---\n# stub\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR", profile_skills
        )
        monkeypatch.setattr(
            "agent.skill_utils.get_skills_dir", lambda: profile_skills
        )

        import json as _json

        result = _json.loads(skill_view("totum-platform-audit"))
        assert result["success"] is True
        assert "platform-level stub" in result["content"]

    def test_skill_view_respects_monkeypatched_skills_dir(
        self, tmp_path, monkeypatch
    ):
        """Existing tests monkeypatch tools.skills_tool.SKILLS_DIR to point
        at a temp dir. The platform-dir walk-up must not override that
        patch — the local override wins. ADR-0001 preserves this contract
        so the test base keeps working.
        """
        from tools.skills_tool import skill_view

        # The local override is set up as a real skills dir with a real skill
        local_skills = tmp_path / "local-skills"
        local_skills.mkdir()
        local_skill = local_skills / "fake-skill"
        local_skill.mkdir()
        (local_skill / "SKILL.md").write_text(
            "---\nname: fake-skill\ndescription: from local override\n---\n# local\n",
            encoding="utf-8",
        )

        # The "platform" dir does not contain the skill; the local dir wins.
        platform_skills = tmp_path / "platform-skills"
        platform_skills.mkdir()
        platform_skill = platform_skills / "fake-skill"
        platform_skill.mkdir()
        (platform_skill / "SKILL.md").write_text(
            "---\nname: fake-skill\ndescription: from platform\n---\n# platform\n",
            encoding="utf-8",
        )

        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", local_skills)

        import json as _json

        result = _json.loads(skill_view("fake-skill"))
        assert result["success"] is True
        # Local override wins — content is "from local override", not "from platform".
        assert "from local override" in result["content"]


# ── agent.skill_commands.build_preloaded_skills_prompt — stub log ─────


class TestBuildPreloadedSkillsPromptStubLogging:
    """When build_preloaded_skills_prompt loads a skill whose SKILL.md has
    metadata.hermes.stub: true, it must emit a WARN log so operators see
    when a worker is loading a stub instead of the canonical content.
    """

    def test_no_warn_for_real_skill(self, tmp_path, monkeypatch, caplog):
        import logging
        from agent.skill_commands import build_preloaded_skills_prompt

        local_skills = tmp_path / "local-skills"
        local_skills.mkdir()
        real_skill = local_skills / "real-skill"
        real_skill.mkdir()
        # Real skill — large body, no stub marker.
        body = "# real\n\n" + ("lots of content\n" * 200)
        (real_skill / "SKILL.md").write_text(
            "---\nname: real-skill\ndescription: a real skill\n---\n" + body,
            encoding="utf-8",
        )

        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", local_skills)
        monkeypatch.setattr(
            "agent.skill_utils.get_skills_dir", lambda: local_skills
        )

        with caplog.at_level(logging.WARNING, logger="agent.skill_commands"):
            prompt, loaded, missing = build_preloaded_skills_prompt(["real-skill"])
        assert loaded == ["real-skill"]
        assert missing == []
        # No [Skill stub] warning should fire for the real skill.
        assert not any("[Skill stub" in rec.message for rec in caplog.records)

    def test_warn_for_platform_stub_with_marker(self, tmp_path, monkeypatch, caplog):
        """A platform-level stub with metadata.hermes.stub: true must
        trigger exactly one [Skill stub] WARN log per loaded skill."""
        import logging
        from agent.skill_commands import build_preloaded_skills_prompt

        platform_root = tmp_path / ".hermes"
        profile_dir = platform_root / "profiles" / "demo"
        profile_skills = profile_dir / "skills"
        platform_skills = platform_root / "skills"
        profile_skills.mkdir(parents=True)
        platform_skills.mkdir(parents=True)

        stub = platform_skills / "totum-platform-audit"
        stub.mkdir()
        (stub / "SKILL.md").write_text(
            "---\n"
            "name: totum-platform-audit\n"
            "description: platform stub\n"
            "metadata:\n"
            "  hermes:\n"
            "    stub: true\n"
            "    canonical_source: ~/.hermes/profiles/spec-writer/skills/totum-platform-audit/SKILL.md\n"
            "---\n"
            "# platform stub\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setattr(
            "tools.skills_tool.SKILLS_DIR", profile_skills
        )
        monkeypatch.setattr(
            "agent.skill_utils.get_skills_dir", lambda: profile_skills
        )

        with caplog.at_level(logging.WARNING, logger="agent.skill_commands"):
            prompt, loaded, missing = build_preloaded_skills_prompt(
                ["totum-platform-audit"]
            )

        assert loaded == ["totum-platform-audit"]
        assert missing == []
        stub_warnings = [
            rec for rec in caplog.records if "[Skill stub]" in rec.message
        ]
        assert len(stub_warnings) == 1
        msg = stub_warnings[0].message
        assert "totum-platform-audit" in msg
        assert "Canonical content at" in msg
        # The canonical_source should appear in the warning so operators can act.
        assert "spec-writer" in msg

    def test_missing_skill_still_returned_in_missing_list(self, tmp_path, monkeypatch):
        """ADR-0001 must not silently swallow genuinely-missing skills.
        If neither local nor platform contains the skill, it remains in
        ``missing`` (and the dispatcher raises Unknown skill(s) as before).
        """
        from agent.skill_commands import build_preloaded_skills_prompt

        local_skills = tmp_path / "local-skills"
        local_skills.mkdir()
        monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", local_skills)

        prompt, loaded, missing = build_preloaded_skills_prompt(["nonexistent-skill"])
        assert loaded == []
        assert missing == ["nonexistent-skill"]
