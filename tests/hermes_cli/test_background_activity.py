from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.background_activity import (
    ActivityLease,
    format_elapsed,
    list_active_work,
    observe_lifecycle,
    sanitize_title,
)


def test_sanitize_title_removes_sensitive_shape_and_bounds_length():
    raw = "  Deploy /Users/alice/project with sk-secret-token\nsecond line  "
    clean = sanitize_title(raw, max_length=32)
    assert "/Users/" not in clean
    assert "sk-secret" not in clean
    assert "\n" not in clean
    assert len(clean) <= 32


def test_lease_aggregates_concurrent_workers_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = ActivityLease.start(key="one", title="First card", worker="kanban", profile="coder", model="m1")
    second = ActivityLease.start(key="two", title="Second card", worker="delegate", profile="default", model="m2")
    try:
        snapshot = list_active_work(tmp_path)
        assert [item["key"] for item in snapshot] == ["one", "two"]
        assert all(item["elapsed_seconds"] >= 0 for item in snapshot)
    finally:
        first.close(state="completed")
        second.close(state="cancelled")
    assert list_active_work(tmp_path) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits are not enforced on Windows")
def test_lease_metadata_is_owner_only(tmp_path):
    lease = ActivityLease.start(key="private", title="Private card", worker="kanban", home=tmp_path)
    try:
        assert stat.S_IMODE(lease.path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(lease.path.stat().st_mode) == 0o600
    finally:
        lease.close()


def test_reader_removes_stale_process_lease(tmp_path):
    root = tmp_path / "cache" / "background-work"
    root.mkdir(parents=True)
    (root / "stale.json").write_text(json.dumps({
        "key": "stale", "title": "Old", "worker": "kanban", "profile": "coder",
        "model": "m", "pid": 99999999, "process_started_at": 1, "started_at": time.time() - 5,
        "state": "running",
    }))
    assert list_active_work(tmp_path) == []
    assert not (root / "stale.json").exists()


def test_elapsed_uses_compact_units():
    assert format_elapsed(0) == "0s"
    assert format_elapsed(65) == "1m 05s"
    assert format_elapsed(3661) == "1h 01m"


def test_lifecycle_tracks_kanban_and_delegate_without_prompt_leak(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_123")
    monkeypatch.setenv("HERMES_BACKGROUND_WORK_TITLE", "Safe card title")
    monkeypatch.setenv("HERMES_PROFILE", "coder")

    observe_lifecycle("on_session_start", session_id="worker-session", model="gpt")
    observe_lifecycle(
        "subagent_start", child_session_id="child-session", child_goal="secret user prompt",
        child_role="leaf",
    )
    active = list_active_work(tmp_path)
    assert {item["title"] for item in active} == {"Safe card title", "Delegated worker"}

    observe_lifecycle("subagent_stop", child_session_id="child-session", child_status="failed")
    observe_lifecycle("on_session_end", session_id="worker-session", completed=False, interrupted=True)
    assert list_active_work(tmp_path) == []
