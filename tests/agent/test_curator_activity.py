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
