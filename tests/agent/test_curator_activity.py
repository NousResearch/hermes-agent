"""Regression tests for curator skill activity timestamps."""

import importlib
from datetime import datetime, timedelta, timezone, UTC
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

    from tools import skill_usage
    from agent import curator

    importlib.reload(skill_usage)
    importlib.reload(curator)
    return home, skill_usage, curator


def test_recent_view_activity_prevents_false_stale_transition(curator_modules, monkeypatch):
    home, skill_usage, curator = curator_modules
    skills_dir = home / "skills"
    _write_skill(skills_dir, "recently-viewed")

    now = datetime(2026, 4, 30, tzinfo=UTC)
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


def test_restored_archived_skill_stays_active_without_faking_usage(curator_modules, monkeypatch):
    """Restoring an archived >30d-idle managed skill stamps ``restored_at`` so the next
    curator pass does not immediately re-archive it, while preserving genuine usage telemetry."""
    home, skill_usage, curator = curator_modules
    skills_dir = home / "skills"
    _write_skill(skills_dir, "old-managed")

    restore_time = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    old_used_at = (restore_time - timedelta(days=45)).isoformat()
    old_created_at = (restore_time - timedelta(days=60)).isoformat()
    skill_usage.save_usage({
        "old-managed": {
            "created_by": "agent",
            "created_at": old_created_at,
            "last_used_at": old_used_at,
            "use_count": 3,
            "state": skill_usage.STATE_ACTIVE,
        }
    })
    monkeypatch.setattr(curator, "get_stale_after_days", lambda: 14)
    monkeypatch.setattr(curator, "get_archive_after_days", lambda: 30)

    # First pass archives the >30d-idle skill.
    archived_counts = curator.apply_automatic_transitions(now=restore_time - timedelta(days=1))
    assert archived_counts["archived"] == 1
    assert not (skills_dir / "old-managed").exists()
    assert (skills_dir / ".archive" / "old-managed").exists()
    assert skill_usage.get_record("old-managed")["state"] == skill_usage.STATE_ARCHIVED

    # Restore at restore_time.
    monkeypatch.setattr(skill_usage, "_now_iso", lambda: restore_time.isoformat())
    ok, msg = skill_usage.restore_skill("old-managed")
    assert ok, msg
    assert (skills_dir / "old-managed").exists()

    rec = skill_usage.get_record("old-managed")
    assert rec["state"] == skill_usage.STATE_ACTIVE
    assert rec["archived_at"] is None
    assert rec["restored_at"] == restore_time.isoformat()
    # Genuine usage telemetry must not be fabricated.
    assert rec["use_count"] == 3
    assert rec["last_used_at"] == old_used_at
    assert skill_usage.activity_count(rec) == 3
    assert skill_usage.latest_activity_at(rec) == restore_time.isoformat()

    # Immediate next curator pass must keep the restored skill active (not re-archive it).
    post_restore = curator.apply_automatic_transitions(now=restore_time + timedelta(days=1))
    assert post_restore["archived"] == 0
    assert post_restore["marked_stale"] == 0
    assert (skills_dir / "old-managed").exists()
    assert skill_usage.get_record("old-managed")["state"] == skill_usage.STATE_ACTIVE

    # Once restored_at ages past stale_after_days (14d), it transitions to stale normally.
    stale_pass = curator.apply_automatic_transitions(now=restore_time + timedelta(days=15))
    assert stale_pass["marked_stale"] == 1
    assert stale_pass["archived"] == 0
    assert skill_usage.get_record("old-managed")["state"] == skill_usage.STATE_STALE

    # Once restored_at ages past archive_after_days (30d), it archives normally again.
    archive_pass = curator.apply_automatic_transitions(now=restore_time + timedelta(days=31))
    assert archive_pass["archived"] == 1
    assert not (skills_dir / "old-managed").exists()
    assert skill_usage.get_record("old-managed")["state"] == skill_usage.STATE_ARCHIVED


def test_restored_never_used_skill_gets_fresh_window_and_ages_normally(curator_modules, monkeypatch):
    """A restored skill with use_count == 0 stays active during the fresh window and ages out normally."""
    home, skill_usage, curator = curator_modules
    skills_dir = home / "skills"
    _write_skill(skills_dir, "never-used")

    restore_time = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    old_created_at = (restore_time - timedelta(days=60)).isoformat()
    skill_usage.save_usage({
        "never-used": {
            "created_by": "agent",
            "created_at": old_created_at,
            "use_count": 0,
            "state": skill_usage.STATE_ACTIVE,
        }
    })
    monkeypatch.setattr(curator, "get_stale_after_days", lambda: 14)
    monkeypatch.setattr(curator, "get_archive_after_days", lambda: 30)

    assert curator.apply_automatic_transitions(now=restore_time - timedelta(days=1))["archived"] == 1

    monkeypatch.setattr(skill_usage, "_now_iso", lambda: restore_time.isoformat())
    ok, msg = skill_usage.restore_skill("never-used")
    assert ok, msg

    rec = skill_usage.get_record("never-used")
    assert rec["use_count"] == 0
    assert rec["last_used_at"] is None
    assert rec["restored_at"] == restore_time.isoformat()

    assert curator.apply_automatic_transitions(now=restore_time + timedelta(days=1))["archived"] == 0
    assert skill_usage.get_record("never-used")["state"] == skill_usage.STATE_ACTIVE

    assert curator.apply_automatic_transitions(now=restore_time + timedelta(days=15))["marked_stale"] == 1
    assert skill_usage.get_record("never-used")["state"] == skill_usage.STATE_STALE

    assert curator.apply_automatic_transitions(now=restore_time + timedelta(days=31))["archived"] == 1
    assert skill_usage.get_record("never-used")["state"] == skill_usage.STATE_ARCHIVED


def test_adopt_still_does_not_reset_inactivity_clock(curator_modules, monkeypatch):
    """Adopting a long-idle unmanaged skill must NOT stamp restored_at or defer archival."""
    home, skill_usage, curator = curator_modules
    skills_dir = home / "skills"
    _write_skill(skills_dir, "legacy-idle")

    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    old_used_at = (now - timedelta(days=45)).isoformat()
    skill_usage.save_usage({
        "legacy-idle": {
            "created_by": None,
            "created_at": (now - timedelta(days=60)).isoformat(),
            "last_used_at": old_used_at,
            "use_count": 2,
            "state": skill_usage.STATE_ACTIVE,
        }
    })
    monkeypatch.setattr(curator, "get_stale_after_days", lambda: 14)
    monkeypatch.setattr(curator, "get_archive_after_days", lambda: 30)

    ok, msg = skill_usage.adopt_skill("legacy-idle")
    assert ok, msg
    rec = skill_usage.get_record("legacy-idle")
    assert rec["restored_at"] is None
    assert skill_usage.latest_activity_at(rec) == old_used_at

    counts = curator.apply_automatic_transitions(now=now)
    assert counts["archived"] == 1
    assert skill_usage.get_record("legacy-idle")["state"] == skill_usage.STATE_ARCHIVED
