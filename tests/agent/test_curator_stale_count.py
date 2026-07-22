"""Tests for agent.curator.stale_skill_count — managed staleness surfacing.

The count backs the `hermes status` staleness line, the /api/curator
``stale_skill_count`` field, and the dashboard curator card. It uses the same
managed candidate and cron-exclusion semantics as automatic transitions and
must stay silent — returning 0 — when the curator is disabled.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)
OLD = (NOW - timedelta(days=45)).isoformat()
FRESH = (NOW - timedelta(days=2)).isoformat()


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

    # Default: no config file → curator defaults. Tests can override.
    monkeypatch.setattr(curator, "_load_config", lambda: {})

    yield {"home": home, "curator": curator, "usage": usage}


def _record(usage, **overrides):
    rec = usage._empty_record()
    rec.update(overrides)
    return rec


def _write_skill(home, name):
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n",
        encoding="utf-8",
    )


def _save_managed_records(curator_env, records):
    u = curator_env["usage"]
    managed = {}
    for name, record in records.items():
        _write_skill(curator_env["home"], name)
        managed[name] = {**record, "created_by": "agent"}
    u.save_usage(managed)


def test_counts_records_idle_past_stale_window(curator_env):
    u, c = curator_env["usage"], curator_env["curator"]
    _save_managed_records(curator_env, {
        "dusty": _record(u, created_by="agent", last_used_at=OLD, created_at=OLD),
        "busy": _record(u, created_by="agent", last_used_at=FRESH, created_at=OLD),
    })
    assert c.stale_skill_count(now=NOW) == 1


def test_view_and_patch_count_as_activity(curator_env):
    """latest_activity_at semantics: a recently viewed/patched skill is not stale."""
    u, c = curator_env["usage"], curator_env["curator"]
    _save_managed_records(curator_env, {
        "viewed": _record(u, last_used_at=OLD, last_viewed_at=FRESH, created_at=OLD),
        "patched": _record(u, last_used_at=OLD, last_patched_at=FRESH, created_at=OLD),
    })
    assert c.stale_skill_count(now=NOW) == 0


def test_never_active_falls_back_to_created_at(curator_env):
    u, c = curator_env["usage"], curator_env["curator"]
    _save_managed_records(curator_env, {
        "old-never-used": _record(u, created_at=OLD),
        "new-never-used": _record(u, created_at=FRESH),
    })
    assert c.stale_skill_count(now=NOW) == 1


def test_pinned_and_archived_records_are_skipped(curator_env):
    u, c = curator_env["usage"], curator_env["curator"]
    _save_managed_records(curator_env, {
        "pinned": _record(u, last_used_at=OLD, created_at=OLD, pinned=True),
        "archived": _record(u, last_used_at=OLD, created_at=OLD, state=u.STATE_ARCHIVED),
        "plain": _record(u, last_used_at=OLD, created_at=OLD),
    })
    assert c.stale_skill_count(now=NOW) == 1


def test_respects_configured_stale_after_days(curator_env, monkeypatch):
    u, c = curator_env["usage"], curator_env["curator"]
    _save_managed_records(
        curator_env,
        {"dusty": _record(u, last_used_at=OLD, created_at=OLD)},
    )
    # OLD is 45 days back — stale at the 30d default, fresh at a 60d window.
    assert c.stale_skill_count(now=NOW) == 1
    monkeypatch.setattr(c, "_load_config", lambda: {"stale_after_days": 60})
    assert c.stale_skill_count(now=NOW) == 0


def test_disabled_curator_returns_zero_without_reading_usage(curator_env, monkeypatch):
    u, c = curator_env["usage"], curator_env["curator"]
    _save_managed_records(
        curator_env,
        {"dusty": _record(u, last_used_at=OLD, created_at=OLD)},
    )
    monkeypatch.setattr(c, "_load_config", lambda: {"enabled": False})

    def _boom():
        raise AssertionError("load_usage must not run when the curator is disabled")

    monkeypatch.setattr(u, "load_usage", _boom)
    assert c.stale_skill_count(now=NOW) == 0


def test_missing_usage_file_counts_zero(curator_env):
    assert curator_env["curator"].stale_skill_count(now=NOW) == 0


def test_unparseable_timestamps_are_skipped(curator_env):
    u, c = curator_env["usage"], curator_env["curator"]
    _save_managed_records(curator_env, {
        "garbled": _record(u, last_used_at="not-a-date", created_at="also-not-a-date"),
    })
    assert c.stale_skill_count(now=NOW) == 0


def test_non_managed_and_cron_referenced_skills_are_skipped(curator_env, monkeypatch):
    u, c = curator_env["usage"], curator_env["curator"]
    records = {
        "managed": _record(u, last_used_at=OLD, created_at=OLD, created_by="agent"),
        "cron-dep": _record(u, last_used_at=OLD, created_at=OLD, created_by="agent"),
        "manual": _record(u, last_used_at=OLD, created_at=OLD),
        "hub-skill": _record(u, last_used_at=OLD, created_at=OLD, created_by="agent"),
        "plan": _record(u, last_used_at=OLD, created_at=OLD, created_by="agent"),
    }
    for name in records:
        _write_skill(curator_env["home"], name)
    hub_dir = curator_env["home"] / "skills" / ".hub"
    hub_dir.mkdir()
    (hub_dir / "lock.json").write_text(
        '{"version": 1, "installed": {"hub-skill": {"source": "test"}}}',
        encoding="utf-8",
    )
    u.save_usage(records)
    monkeypatch.setattr(c, "_cron_referenced_skills", lambda: {"cron-dep"})

    assert c.stale_skill_count(now=NOW) == 1
