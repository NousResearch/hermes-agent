"""Tests for curator iteration cap and external-symlink exclusion.

Regression tests for #44771: curator entered a multi-hour loop making 811
API calls because max_iterations was 9999 and 26 external symlinked skills
inflated the candidate list.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

_symlinks_available = True
try:
    os.symlink
except AttributeError:
    _symlinks_available = False
else:
    if sys.platform == "win32":
        _tmp = Path(os.environ.get("TEMP", ".")) / "_symlink_test"
        try:
            _tmp.symlink_to(".")
            _tmp.unlink()
        except OSError:
            _symlinks_available = False

skip_no_symlinks = pytest.mark.skipif(
    not _symlinks_available,
    reason="symlinks not available (Windows without developer mode)",
)


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + freshly reloaded curator + skill_usage modules."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.skill_usage as usage
    importlib.reload(usage)
    import agent.curator as curator
    importlib.reload(curator)

    monkeypatch.setattr(curator, "_run_llm_review", lambda prompt: "llm-stub")
    monkeypatch.setattr(curator, "_load_config", lambda: {})
    monkeypatch.setattr(usage, "_prune_builtins_enabled", lambda: False)

    return {"home": home, "curator": curator, "usage": usage}


def _write_skill(skills_dir: Path, name: str):
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: x\n---\n", encoding="utf-8",
    )
    return d


def _mark_agent_created(usage_mod, name: str):
    data = usage_mod.load_usage()
    data[name] = usage_mod._empty_record()
    data[name]["created_by"] = "agent"
    usage_mod.save_usage(data)


class TestMaxIterationsCap:
    """curator.get_max_iterations() must return a sane default and honor config."""

    def test_default_is_200(self, curator_env):
        c = curator_env["curator"]
        assert c.get_max_iterations() == 200

    def test_config_override(self, curator_env, monkeypatch):
        c = curator_env["curator"]
        monkeypatch.setattr(c, "_load_config", lambda: {"max_iterations": 500})
        assert c.get_max_iterations() == 500

    def test_invalid_config_falls_back(self, curator_env, monkeypatch):
        c = curator_env["curator"]
        monkeypatch.setattr(c, "_load_config", lambda: {"max_iterations": "not-a-number"})
        assert c.get_max_iterations() == 200


@skip_no_symlinks
class TestExternalSymlinksExcluded:
    """Symlinked skills pointing outside ~/.hermes/skills/ must be skipped."""

    def test_external_symlink_excluded(self, curator_env):
        u = curator_env["usage"]
        skills_dir = curator_env["home"] / "skills"

        external_dir = curator_env["home"].parent / "agents" / "skills" / "lark-im"
        external_dir.mkdir(parents=True)
        (external_dir / "SKILL.md").write_text(
            "---\nname: lark-im\ndescription: x\n---\n", encoding="utf-8",
        )

        link = skills_dir / "lark-im"
        link.symlink_to(external_dir)

        _mark_agent_created(u, "lark-im")

        names = u.list_agent_created_skill_names()
        assert "lark-im" not in names

    def test_internal_symlink_included(self, curator_env):
        u = curator_env["usage"]
        skills_dir = curator_env["home"] / "skills"

        real_dir = skills_dir / ".internal" / "my-skill"
        real_dir.mkdir(parents=True)
        (real_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: x\n---\n", encoding="utf-8",
        )

        link = skills_dir / "my-skill"
        link.symlink_to(real_dir)

        _mark_agent_created(u, "my-skill")

        names = u.list_agent_created_skill_names()
        assert "my-skill" in names

    def test_non_symlink_still_included(self, curator_env):
        u = curator_env["usage"]
        skills_dir = curator_env["home"] / "skills"

        _write_skill(skills_dir, "normal-skill")
        _mark_agent_created(u, "normal-skill")

        names = u.list_agent_created_skill_names()
        assert "normal-skill" in names
