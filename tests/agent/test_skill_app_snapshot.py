"""Application requirements survive disk and in-process skill-index reuse."""

import sys

import pytest

from hermes_platform import declaration


@pytest.fixture(autouse=True)
def _legacy_flat_skills_index(monkeypatch):
    """This module renders the LEGACY flat skills index.

    The index is RETIRED by default (card t_cc3c6951, operator ruling 2026-09-30); the curated
    hierarchical surface is the routing surface now. This test exercises the renderer that still
    exists behind the explicit escape hatch, so it is pinned ON here. The DARK default is guarded by
    tests/agent/test_skills_index_deprecated.py.
    """
    monkeypatch.setenv("HERMES_SKILLS_INDEX", "flat")


def test_application_gate_rechecks_snapshot_without_losing_description(tmp_path, monkeypatch):
    from agent import prompt_builder as pb

    monkeypatch.setattr(declaration, "_REGISTRY", {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    skills = tmp_path / "skills"
    skill = skills / "app-guide" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: app-guide\ndescription: Application instructions.\n"
        "requires_apps: [thing]\n---\nHelp with the application.\n", encoding="utf-8",
    )
    app = tmp_path / "application.exe"
    declaration.register("thing", declaration.parse_declaration(
        "thing", {sys.platform: {"presence": "executable", "location": str(app)}},
        {"app": True}, where="test-plugin/plugin.yaml",
    ))
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    try:
        def build():
            return pb._build_skills_system_prompt_inner(skills, [], None, None, None)

        assert "app-guide" not in build()
        snapshot = pb._load_skills_snapshot(skills)
        assert snapshot is not None
        assert snapshot["skills"][0]["requires_apps"] == ["thing"]
        app.write_text("presence fixture", encoding="utf-8")
        assert "app-guide: Application instructions." in build()
        app.unlink()
        assert "app-guide" not in build()
        pb.clear_skills_system_prompt_cache()
        assert "app-guide" not in build()
    finally:
        pb.clear_skills_system_prompt_cache(clear_snapshot=True)
