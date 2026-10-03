"""Regression tests for the ``filter_skill_visible`` plugin hook (#125257).

A plugin that knows which skills the current runtime surface can execute must be able to
hide an incompatible skill BEFORE the model ever sees it offered — in the system-prompt
skills index AND in ``skills_list``. ``None``/``True`` (no opinion) and hook errors keep
the skill visible; stock gates still apply first, so a plugin can only ADD hiding.
"""

import json
import sys

import pytest

from unittest.mock import patch

import hermes_cli.plugins as plugins_mod
from agent import prompt_builder as pb
from agent.skill_utils import plugin_filter_hides_skill
from tools.skills_tool import _find_all_skills, _skills_scan_signature


def _make_skill(skills_dir, name, frontmatter_extra="", body="Step 1."):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n{frontmatter_extra}---\n\n# {name}\n\n{body}\n"
    )
    return skill_dir


class _ManagerStub:
    """Minimal PluginManager double: hooks only, no discovery."""

    def __init__(self):
        self._hooks = {}

    def has_hook(self, hook_name):
        return bool(self._hooks.get(hook_name))

    def invoke_hook(self, hook_name, **kwargs):
        results = []
        for cb in self._hooks.get(hook_name, []):
            ret = cb(**kwargs)
            if ret is not None:
                results.append(ret)
        return results


@pytest.fixture()
def surface_gate(monkeypatch, tmp_path):
    """A plugin manager whose filter_skill_visible hides skills tagged for another surface."""
    manager = _ManagerStub()
    seen = []

    def gate(skill_name="", frontmatter=None, session_info=None, **_):
        seen.append({"skill_name": skill_name, "frontmatter": frontmatter or {},
                     "session_info": session_info or {}})
        surfaces = (frontmatter or {}).get("skill_surface") or []
        if isinstance(surfaces, str):
            surfaces = [surfaces]
        return False if "desktop" in surfaces else None

    manager._hooks["filter_skill_visible"] = [gate]
    # Patch BOTH slots: get_plugin_manager() adopts a monkeypatched single-slot manager into
    # the keyed cache, which would otherwise leak the stub into later tests in this process.
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    return manager, seen


def _no_hook(monkeypatch):
    monkeypatch.setattr(plugins_mod, "_plugin_manager", _ManagerStub())
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})


# ---------------------------------------------------------------------------
# plugin_filter_hides_skill — the shared helper
# ---------------------------------------------------------------------------


class TestPluginFilterHidesSkill:
    def test_false_hides_and_none_keeps(self, surface_gate):
        assert plugin_filter_hides_skill("desktop-only", {"skill_surface": ["desktop"]}) is True
        assert plugin_filter_hides_skill("anywhere", {}) is False

    def test_callback_receives_name_frontmatter_session_info(self, surface_gate):
        _, seen = surface_gate
        plugin_filter_hides_skill("s1", {"name": "s1", "skill_surface": ["desktop"]})
        assert len(seen) == 1
        assert seen[0]["skill_name"] == "s1"
        assert seen[0]["frontmatter"].get("skill_surface") == ["desktop"]
        assert {"platform", "profile_name", "cwd"} <= set(seen[0]["session_info"])

    def test_raising_callback_fails_open(self, monkeypatch):
        manager = _ManagerStub()
        manager._hooks["filter_skill_visible"] = lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))
        monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
        assert plugin_filter_hides_skill("s1", {}) is False

    def test_no_callback_registered_is_visible(self, monkeypatch):
        _no_hook(monkeypatch)
        assert plugin_filter_hides_skill("s1", {}) is False


# ---------------------------------------------------------------------------
# skills_list listing path
# ---------------------------------------------------------------------------


class TestFindAllSkills:
    def test_hook_hides_tagged_skill_from_listing(self, surface_gate, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.skills_tool._SKILLS_CACHE", {})
        _make_skill(tmp_path, "plain-skill")
        _make_skill(tmp_path, "desktop-only", frontmatter_extra="skill_surface:\n  - desktop\n")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            names = {s["name"] for s in _find_all_skills()}
        assert names == {"plain-skill"}

    @pytest.mark.platforms("any")
    def test_stock_gates_still_apply_without_hook(self, monkeypatch, tmp_path):
        _no_hook(monkeypatch)
        monkeypatch.setattr("tools.skills_tool._SKILLS_CACHE", {})
        _make_skill(tmp_path, "visible")
        # The stock gate hides a skill for a different OS on every real host.
        other_platform = "linux" if sys.platform == "win32" else "windows"
        _make_skill(tmp_path, "other-platform", frontmatter_extra=f"platforms: [{other_platform}]\n")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            names = {s["name"] for s in _find_all_skills()}
        assert "visible" in names
        assert "other-platform" not in names

    def test_signature_tracks_hook_registration(self, surface_gate, monkeypatch, tmp_path):
        dirs = [tmp_path]
        with_hook = _skills_scan_signature(dirs, set())
        _no_hook(monkeypatch)
        without_hook = _skills_scan_signature(dirs, set())
        assert with_hook[-1] is True
        assert without_hook[-1] is False


# ---------------------------------------------------------------------------
# system-prompt skills index path
# ---------------------------------------------------------------------------


class TestSkillsSystemPromptIndex:
    def _render(self, monkeypatch, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(pb, "get_skills_dir", lambda: skills, raising=True)
        monkeypatch.setattr(pb, "get_all_skills_dirs", lambda: [skills], raising=True)
        monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *a, **k: set())
        monkeypatch.setattr(pb, "_skills_prompt_snapshot_path", lambda: tmp_path / "snap.json")
        pb.clear_skills_system_prompt_cache()
        return skills

    def test_hook_hides_tagged_skill_from_index(self, surface_gate, monkeypatch, tmp_path):
        skills = self._render(monkeypatch, tmp_path)
        _make_skill(skills, "plain-skill")
        _make_skill(skills, "desktop-only", frontmatter_extra="skill_surface:\n  - desktop\n")
        out = pb.build_skills_system_prompt()
        assert "plain-skill" in out
        assert "desktop-only" not in out

    def test_snapshot_roundtrips_frontmatter_for_the_hook(self, surface_gate, monkeypatch, tmp_path):
        skills = self._render(monkeypatch, tmp_path)
        _make_skill(skills, "plain-skill")
        _make_skill(skills, "snap-hidden", frontmatter_extra="skill_surface:\n  - desktop\n")
        first = pb.build_skills_system_prompt()
        pb.clear_skills_system_prompt_cache()
        snapshot = json.loads((tmp_path / "snap.json").read_text(encoding="utf-8"))
        entry = next(e for e in snapshot["skills"] if e["frontmatter_name"] == "snap-hidden")
        assert entry["frontmatter"].get("skill_surface") == ["desktop"]
        second = pb.build_skills_system_prompt()  # served FROM the snapshot now
        assert "snap-hidden" not in second
        assert "plain-skill" in second
        assert first == second

    def test_no_hook_leaves_index_unchanged(self, monkeypatch, tmp_path):
        _no_hook(monkeypatch)
        skills = self._render(monkeypatch, tmp_path)
        _make_skill(skills, "plain-skill")
        _make_skill(skills, "desktop-only", frontmatter_extra="skill_surface:\n  - desktop\n")
        out = pb.build_skills_system_prompt()
        assert "plain-skill" in out
        assert "desktop-only" in out
