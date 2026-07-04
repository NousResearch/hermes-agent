"""Regression tests for gateway channel auto-loading of skill bundles."""
from types import SimpleNamespace

import yaml


def _write_skill(base, name, body="Follow this skill."):
    skill_dir = base / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Test skill {name}
---
# {name}

{body}
""",
        encoding="utf-8",
    )


def test_auto_loader_accepts_skill_bundle_names(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    hermes_home = tmp_path / "hermes"
    bundles_dir = hermes_home / "skill-bundles"
    bundles_dir.mkdir(parents=True)
    _write_skill(hermes_home, "skill-a")
    _write_skill(hermes_home, "skill-b")
    (bundles_dir / "ops-core.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "ops-core",
                "description": "Ops baseline",
                "skills": ["skill-a", "skill-b"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    # tools.skills_tool keeps SKILLS_DIR as a module-level compatibility
    # constant, so point it at the temporary test home after import too.
    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", hermes_home / "skills")

    runner = object.__new__(GatewayRunner)
    event = SimpleNamespace(text="hello", auto_skill=["/ops-core"])
    session_entry = SimpleNamespace(created_at=1, updated_at=1, was_auto_reset=False)

    assert runner._inject_auto_skills_for_new_session(event, session_entry, "task-1", "session-1")

    assert "ops-core" in event.text
    assert "skill-a" in event.text
    assert "skill-b" in event.text
    assert event.text.endswith("hello")
