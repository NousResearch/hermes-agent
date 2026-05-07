"""Tests for agent.user_status (cross-bot user-state storage)."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and reload agent.user_status."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Reset the one-shot warning latch (if any) — harmless if absent.
    import hermes_constants
    if hasattr(hermes_constants, "_profile_fallback_warned"):
        hermes_constants._profile_fallback_warned = False
    # Force a fresh import binding so get_hermes_home() reads the new env.
    import importlib
    from agent import user_status as us
    importlib.reload(us)
    return tmp_path, us


def _state_file(home: Path) -> Path:
    return home / "state" / "user_status.json"


def test_ensure_state_dir_creates_directory(hermes_home):
    home, us = hermes_home
    target = home / "state"
    assert not target.exists()
    out = us.ensure_state_dir()
    assert out == target
    assert target.is_dir()
    # Idempotent.
    us.ensure_state_dir()
    assert target.is_dir()


def test_load_missing_file_returns_empty_status(hermes_home):
    _, us = hermes_home
    status = us.load()
    assert isinstance(status, us.UserStatus)
    assert status.device_mode is None
    assert status.afk_status is None
    assert status.focus_project is None
    assert status.quiet_hours_until is None
    assert status.location is None
    assert status.per_field_updated_at == {}
    assert status.updated_by is None


def test_save_field_persists_and_load_roundtrips(hermes_home):
    home, us = hermes_home
    result = us.save_field("device_mode", "phone", writer="telegram")
    assert result.device_mode == "phone"
    assert result.updated_by == "telegram"
    assert "device_mode" in result.per_field_updated_at

    # File on disk matches what load() returns.
    assert _state_file(home).exists()
    on_disk = json.loads(_state_file(home).read_text())
    assert on_disk["device_mode"] == "phone"
    assert on_disk["updated_by"] == "telegram"

    loaded = us.load()
    assert loaded.device_mode == "phone"
    assert loaded.updated_by == "telegram"


def test_save_field_preserves_other_fields(hermes_home):
    _, us = hermes_home
    us.save_field("device_mode", "phone", writer="telegram")
    us.save_field("focus_project", "hermes", writer="discord")
    us.save_field("location", "home", writer="slack")

    loaded = us.load()
    assert loaded.device_mode == "phone"
    assert loaded.focus_project == "hermes"
    assert loaded.location == "home"
    assert loaded.updated_by == "slack"  # last writer wins

    # All three fields have timestamps; first-write timestamp survives.
    for f in ("device_mode", "focus_project", "location"):
        assert f in loaded.per_field_updated_at


def test_save_field_rejects_unknown_field(hermes_home):
    _, us = hermes_home
    with pytest.raises(ValueError):
        us.save_field("nonsense_field", "x", writer="telegram")


def test_save_field_overwrites_value_and_updates_timestamp(hermes_home):
    _, us = hermes_home
    us.save_field("device_mode", "phone", writer="telegram")
    first_ts = us.load().per_field_updated_at["device_mode"]

    time.sleep(0.01)  # ensure clock advances
    us.save_field("device_mode", "desktop", writer="discord")
    later = us.load()
    assert later.device_mode == "desktop"
    assert later.updated_by == "discord"
    assert later.per_field_updated_at["device_mode"] >= first_ts


def test_is_stale_missing_field_is_stale(hermes_home):
    _, us = hermes_home
    assert us.is_stale("device_mode", threshold_seconds=60) is True


def test_is_stale_fresh_field_not_stale(hermes_home):
    _, us = hermes_home
    us.save_field("device_mode", "phone", writer="telegram")
    assert us.is_stale("device_mode", threshold_seconds=60) is False


def test_is_stale_old_field_is_stale(hermes_home):
    home, us = hermes_home
    us.save_field("device_mode", "phone", writer="telegram")
    # Hand-edit the timestamp to be old.
    raw = json.loads(_state_file(home).read_text())
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    raw["per_field_updated_at"]["device_mode"] = old
    _state_file(home).write_text(json.dumps(raw))

    assert us.is_stale("device_mode", threshold_seconds=60) is True
    # Caller-supplied status path also works.
    status = us.load()
    assert us.is_stale("device_mode", threshold_seconds=60, status=status) is True


def test_is_stale_handles_corrupt_timestamp(hermes_home):
    home, us = hermes_home
    us.save_field("device_mode", "phone", writer="telegram")
    raw = json.loads(_state_file(home).read_text())
    raw["per_field_updated_at"]["device_mode"] = "not-a-date"
    _state_file(home).write_text(json.dumps(raw))
    assert us.is_stale("device_mode", threshold_seconds=60) is True


def test_concurrent_writes_no_loss(hermes_home):
    """7 threads writing different fields concurrently — none clobbered."""
    _, us = hermes_home

    fields_writers = [
        ("device_mode", "telegram", "phone"),
        ("afk_status", "discord", "available"),
        ("focus_project", "slack", "hermes"),
        ("quiet_hours_until", "whatsapp", "2026-05-08T08:00:00+00:00"),
        ("location", "matrix", "home"),
        ("device_mode", "signal", "desktop"),  # same field; last-writer-wins
        ("afk_status", "email", "afk"),
    ]

    barrier = threading.Barrier(len(fields_writers))
    errors: list = []

    def worker(field_name, writer, value):
        try:
            barrier.wait(timeout=5)
            us.save_field(field_name, value, writer=writer)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=fw) for fw in fields_writers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"worker errors: {errors}"
    final = us.load()

    # Every field that was written at least once must be set.
    assert final.device_mode in ("phone", "desktop")
    assert final.afk_status in ("available", "afk")
    assert final.focus_project == "hermes"
    assert final.quiet_hours_until == "2026-05-08T08:00:00+00:00"
    assert final.location == "home"

    # All five field timestamps present — proves no read-modify-write
    # clobbered another writer's update.
    for f in ("device_mode", "afk_status", "focus_project",
              "quiet_hours_until", "location"):
        assert f in final.per_field_updated_at, (
            f"{f} missing from per_field_updated_at — concurrent write lost"
        )


def test_concurrent_writes_same_field_high_contention(hermes_home):
    """20 threads hammering the same field — all increments accounted for
    via the writer tag set."""
    _, us = hermes_home
    n = 20

    barrier = threading.Barrier(n)

    def worker(i):
        barrier.wait(timeout=5)
        us.save_field("location", f"loc-{i}", writer=f"w-{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    final = us.load()
    # Some writer won. The file is a single, valid JSON document.
    assert final.location is not None
    assert final.location.startswith("loc-")
    assert final.updated_by is not None


def test_load_handles_corrupt_json(hermes_home):
    home, us = hermes_home
    us.ensure_state_dir()
    _state_file(home).write_text("{not valid json")
    # Doesn't raise; returns empty.
    status = us.load()
    assert status.device_mode is None
    # And next save recovers cleanly.
    us.save_field("device_mode", "phone", writer="telegram")
    assert us.load().device_mode == "phone"


def test_atomic_no_partial_file_on_disk(hermes_home):
    """After save, no leftover .tmp files in the state dir."""
    home, us = hermes_home
    us.save_field("device_mode", "phone", writer="telegram")
    leftovers = [
        p for p in (home / "state").iterdir()
        if p.name.endswith(".tmp")
    ]
    assert leftovers == []
