"""Unit tests for ``_prune_stale_platform_writers`` (gateway.status).

The bug: a platform entry stamped by a now-dead gateway still reports ``state: connected``,
so a readiness check that trusts ``gateway_state.json`` is lied to (e.g. the ``homeassistant``
record carrying writer_pid 22962 long after that gateway exited). The fix is dead-only:
prune entries whose writer PID is provably gone; flag, never silently remove, entries whose
writer is a DIFFERENT live process.
"""

from __future__ import annotations

from gateway import status


def test_dead_writer_is_pruned(monkeypatch):
    monkeypatch.setattr(
        status, "_get_process_start_time", lambda pid: None if pid == 22962 else 111
    )
    platforms = {
        "telegram": {"state": "connected", "writer_pid": 2308},
        "homeassistant": {"state": "connected", "writer_pid": 22962},
    }
    status._prune_stale_platform_writers(platforms, current_pid=2308)
    assert "homeassistant" not in platforms
    assert "telegram" in platforms


def test_live_foreign_writer_is_flagged_not_pruned(monkeypatch):
    monkeypatch.setattr(
        status, "_get_process_start_time", lambda pid: None if pid == 22962 else 111
    )
    platforms = {"homeassistant": {"state": "connected", "writer_pid": 777}}
    status._prune_stale_platform_writers(platforms, current_pid=2308)
    assert "homeassistant" in platforms
    assert platforms["homeassistant"]["needs_attention"] is True


def test_current_writer_is_left_untouched(monkeypatch):
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 111)
    platforms = {"telegram": {"state": "connected", "writer_pid": 2308}}
    status._prune_stale_platform_writers(platforms, current_pid=2308)
    assert "telegram" in platforms
    assert "needs_attention" not in platforms["telegram"]


def test_unstamped_entry_is_left_alone(monkeypatch):
    # A legacy entry with no writer_pid must not be pruned or flagged (fail safe).
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    platforms = {"legacy": {"state": "connected"}}
    status._prune_stale_platform_writers(platforms, current_pid=2308)
    assert "legacy" in platforms
    assert "needs_attention" not in platforms["legacy"]


def test_non_dict_entry_is_skipped(monkeypatch):
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    platforms = {"odd": "not-a-dict"}
    status._prune_stale_platform_writers(platforms, current_pid=2308)
    assert platforms["odd"] == "not-a-dict"
