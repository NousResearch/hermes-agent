"""Regression tests for curator skill activity timestamps."""

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _write_skill(skills_dir: Path, name: str) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n\n# {name}\n",
        encoding="utf-8",
    )


@pytest.fixture
def curator_modules(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import tools.skill_usage as skill_usage
    import agent.curator as curator

    importlib.reload(skill_usage)
    importlib.reload(curator)
    return home, skill_usage, curator


def test_recent_view_activity_prevents_false_stale_transition(curator_modules, monkeypatch):
    home, skill_usage, curator = curator_modules
    skills_dir = home / "skills"
    _write_skill(skills_dir, "recently-viewed")

    now = datetime(2026, 4, 30, tzinfo=timezone.utc)
    created_at = (now - timedelta(days=60)).isoformat()
    last_viewed_at = (now - timedelta(days=1)).isoformat()
    skill_usage.save_usage({
        "recently-viewed": {
            "created_at": created_at,
            "last_viewed_at": last_viewed_at,
            "view_count": 1,
            "state": "active",
        }
    })
    monkeypatch.setattr(curator, "get_stale_after_days", lambda: 30)
    monkeypatch.setattr(curator, "get_archive_after_days", lambda: 90)

    counts = curator.apply_automatic_transitions(now=now)

    assert counts["marked_stale"] == 0
    assert skill_usage.get_record("recently-viewed")["state"] == "active"


def test_fresh_create_reusing_dead_skill_name_is_not_archived(curator_modules, monkeypatch):
    """#65992: skill_manage(create) reports skills/<category>/<name>/SKILL.md
    but the file ends up in skills/.archive/<name>/.

    A usage record left behind by a previous skill with the same name (removed
    without skill_manage(delete), so ``forget()`` never ran) carries an
    expired inactivity clock. The next automatic-transition pass reads that
    clock, decides the seconds-old skill is long-inactive, and
    ``archive_skill()`` relocates it — flattening the category, which is
    exactly the reported filesystem shape. A successful create must start a
    new life: the stale record is discarded, so the pass leaves the new
    skill alone.
    """
    home, skill_usage, curator = curator_modules
    skills_dir = home / "skills"

    now = datetime.now(timezone.utc)
    stale = (now - timedelta(days=200)).isoformat()
    skill_usage.save_usage({
        "fresh-skill": {
            "created_by": "agent",
            "use_count": 3,
            "last_used_at": stale,
            "created_at": stale,
            "state": "active",
        }
    })

    from tools.skill_manager_tool import skill_manage
    import json

    raw = skill_manage(
        action="create",
        name="fresh-skill",
        category="devops",
        content="---\nname: fresh-skill\ndescription: test skill\n---\n\n# fresh-skill\n",
    )
    result = json.loads(raw)
    assert result["success"] is True
    assert (skills_dir / "devops" / "fresh-skill" / "SKILL.md").exists()

    monkeypatch.setattr(curator, "get_stale_after_days", lambda: 30)
    monkeypatch.setattr(curator, "get_archive_after_days", lambda: 90)

    counts = curator.apply_automatic_transitions(now=now)

    assert counts["archived"] == 0
    assert (skills_dir / "devops" / "fresh-skill" / "SKILL.md").exists(), (
        "freshly created skill was relocated away from its reported path"
    )
    assert not (skills_dir / ".archive" / "fresh-skill").exists(), (
        "freshly created skill was archived by the automatic-transition pass"
    )
