"""Tests for agent.curator.stale_skill_count — cheap staleness surfacing.

The count backs the `hermes status` staleness line, the /api/curator
``stale_skill_count`` field, and the dashboard curator card. It reads only
``.usage.json`` (no skill-dir walk, no state writes) and must stay silent —
returning 0 — when the curator is disabled.
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


def test_counts_records_idle_past_stale_window(curator_env):
    u, c = curator_env["usage"], curator_env["curator"]
    u.save_usage({
        "dusty": _record(u, created_by="agent", last_used_at=OLD, created_at=OLD),
        "busy": _record(u, created_by="agent", last_used_at=FRESH, created_at=OLD),
    })
    assert c.stale_skill_count(now=NOW) == 1


def test_view_and_patch_count_as_activity(curator_env):
    """latest_activity_at semantics: a recently viewed/patched skill is not stale."""
    u, c = curator_env["usage"], curator_env["curator"]
    u.save_usage({
        "viewed": _record(u, last_used_at=OLD, last_viewed_at=FRESH, created_at=OLD),
        "patched": _record(u, last_used_at=OLD, last_patched_at=FRESH, created_at=OLD),
    })
    assert c.stale_skill_count(now=NOW) == 0


def test_never_active_falls_back_to_created_at(curator_env):
    u, c = curator_env["usage"], curator_env["curator"]
    u.save_usage({
        "old-never-used": _record(u, created_at=OLD),
        "new-never-used": _record(u, created_at=FRESH),
    })
    assert c.stale_skill_count(now=NOW) == 1


def test_pinned_and_archived_records_are_skipped(curator_env):
    u, c = curator_env["usage"], curator_env["curator"]
    u.save_usage({
        "pinned": _record(u, last_used_at=OLD, created_at=OLD, pinned=True),
        "archived": _record(u, last_used_at=OLD, created_at=OLD, state=u.STATE_ARCHIVED),
        "plain": _record(u, last_used_at=OLD, created_at=OLD),
    })
    assert c.stale_skill_count(now=NOW) == 1


def test_respects_configured_stale_after_days(curator_env, monkeypatch):
    u, c = curator_env["usage"], curator_env["curator"]
    u.save_usage({"dusty": _record(u, last_used_at=OLD, created_at=OLD)})
    # OLD is 45 days back — stale at the 30d default, fresh at a 60d window.
    assert c.stale_skill_count(now=NOW) == 1
    monkeypatch.setattr(c, "_load_config", lambda: {"stale_after_days": 60})
    assert c.stale_skill_count(now=NOW) == 0


def test_disabled_curator_returns_zero_without_reading_usage(curator_env, monkeypatch):
    u, c = curator_env["usage"], curator_env["curator"]
    u.save_usage({"dusty": _record(u, last_used_at=OLD, created_at=OLD)})
    monkeypatch.setattr(c, "_load_config", lambda: {"enabled": False})

    def _boom():
        raise AssertionError("load_usage must not run when the curator is disabled")

    monkeypatch.setattr(u, "load_usage", _boom)
    assert c.stale_skill_count(now=NOW) == 0


def test_missing_usage_file_counts_zero(curator_env):
    assert curator_env["curator"].stale_skill_count(now=NOW) == 0


def test_unparseable_timestamps_are_skipped(curator_env):
    u, c = curator_env["usage"], curator_env["curator"]
    u.save_usage({
        "garbled": _record(u, last_used_at="not-a-date", created_at="also-not-a-date"),
    })
    assert c.stale_skill_count(now=NOW) == 0
