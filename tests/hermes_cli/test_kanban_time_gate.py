"""Behavior contracts for per-task Kanban dispatch time gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_time_gate as ktg
from tools import kanban_tools as kt


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def test_dispatch_window_uses_timezone_wall_clock_across_dst() -> None:
    window = ktg.normalize_dispatch_window("01:00-03:00 America/Chicago")

    # The repeated fall-back hour is inside the window in both UTC folds.
    assert ktg.dispatch_gate_open(None, window, now=_epoch("2024-11-03T06:30:00"))
    assert ktg.dispatch_gate_open(None, window, now=_epoch("2024-11-03T07:30:00"))
    # Spring-forward skips 02:xx; the gate closes at the first real 03:00.
    assert ktg.dispatch_gate_open(None, window, now=_epoch("2024-03-10T07:30:00"))
    assert not ktg.dispatch_gate_open(None, window, now=_epoch("2024-03-10T08:30:00"))

    overnight = ktg.normalize_dispatch_window("23:00-05:30 America/Chicago")
    assert ktg.dispatch_gate_open(None, overnight, now=_epoch("2026-01-02T05:30:00"))
    assert ktg.dispatch_gate_open(None, overnight, now=_epoch("2026-01-02T11:29:00"))
    assert not ktg.dispatch_gate_open(None, overnight, now=_epoch("2026-01-02T11:30:00"))

    with pytest.raises(ValueError, match="timezone-aware"):
        ktg.normalize_dispatch_after("2026-01-01T12:00:00")


def test_gate_is_enforced_end_to_end_and_resets_failures_on_lapse(
    kanban_home, monkeypatch,
) -> None:
    now = _epoch("2026-01-01T00:00:00")
    future = "2026-01-01T01:00:00Z"
    monkeypatch.setattr(ktg.time, "time", lambda: now)

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(
            conn,
            title="windowed child",
            assignee="review",
            parents=(parent,),
            dispatch_after=future,
        )
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
        conn.execute(
            "UPDATE tasks SET consecutive_failures=2, last_failure_error='old failure' "
            "WHERE id=?",
            (child,),
        )
        conn.commit()

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "todo"
        assert kb.claim_task(conn, child) is None
        assert all(row["id"] != child for row in kbd._lane_rows(conn, "ready"))

        monkeypatch.setattr(ktg.time, "time", lambda: now + 3600)
        assert kb.recompute_ready(conn) == 1
        opened = kb.get_task(conn, child)
        assert opened.status == "ready"
        assert opened.consecutive_failures == 0
        assert opened.last_failure_error is None
        assert kb.claim_task(conn, child, claimer="gate-test") is not None

    cli_created = json.loads(
        kc.run_slash(
            "create 'cli gated' --assignee review "
            "--dispatch-window '23:00-05:30 America/Chicago' --json"
        )
    )
    assert cli_created["dispatch_window"] == "23:00-05:30 America/Chicago"
    assert "Cleared dispatch gate" in kc.run_slash(f"gate {cli_created['id']} --clear")

    tool_created = json.loads(
        kt._handle_create(
            {
                "title": "tool gated",
                "assignee": "review",
                "dispatch_after": future,
            }
        )
    )
    assert tool_created["dispatch_after"] == _epoch("2026-01-01T01:00:00")
