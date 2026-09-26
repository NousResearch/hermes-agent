from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest

from hermes_cli import kanban_diagnostics as kd

CAP_2 = {"kanban": {"max_in_progress_per_profile": 2}}
NOW = 1_000_000


def _ready(priority=225, **overrides):
    task = {"id": "t_starved", "title": "release", "assignee": "builder", "status": "ready",
            "priority": priority, "claim_lock": None, "created_at": NOW - 25 * 60,
            "consecutive_failures": 0, "last_failure_error": None}
    task.update(overrides)
    return task


def _lanes(holder_priorities=(255, 150), idle=("reviewer",)):
    return {
        "holders": {"builder": [{"id": f"t_hold{p}", "title": "h", "priority": p,
                                 "last_heartbeat_at": NOW - 30}
                                for p in sorted(holder_priorities)]},
        "idle_profiles": list(idle),
    }


def _inversions(task, lanes, config=CAP_2, now=NOW):
    promoted = [{"kind": "promoted", "created_at": NOW - 25 * 60, "payload": None}]
    diags = kd.compute_task_diagnostics(task, promoted, [], now=now, config=config, lanes=lanes)
    return [d for d in diags if d.kind == "lane_priority_inversion"]


def test_live_shape_is_flagged_naming_lower_holder_and_idle_lane():
    [diag] = _inversions(_ready(), _lanes())
    assert diag.severity == "error"
    assert diag.data["outranked_holders"] == ["t_hold150"]
    assert diag.data["idle_profiles"] == ["reviewer"]
    assert diag.data["running"] == 2 and diag.data["cap"] == 2
    assert diag.data["runs"] == 0
    assert "reviewer" in diag.detail
    assert any(a.payload.get("command") == "hermes kanban reassign t_starved reviewer"
               for a in diag.actions)


def test_not_flagged_when_every_holder_outranks_the_card():
    assert _inversions(_ready(priority=100), _lanes()) == []


def test_equal_priority_holder_is_not_an_inversion():
    assert _inversions(_ready(priority=150), _lanes()) == []


def test_not_flagged_when_lane_has_a_free_slot():
    assert _inversions(_ready(), _lanes(holder_priorities=(150,))) == []


def test_not_flagged_without_a_per_profile_cap():
    assert _inversions(_ready(), _lanes(), config={"kanban": {}}) == []


def test_not_flagged_before_threshold():
    task = _ready(created_at=NOW - 60)
    diags = kd.compute_task_diagnostics(task, [{"kind": "promoted", "created_at": NOW - 60}], [],
                                        now=NOW, config=CAP_2, lanes=_lanes())
    assert [d for d in diags if d.kind == "lane_priority_inversion"] == []


def test_claimed_card_is_not_flagged():
    assert _inversions(_ready(claim_lock="host:1"), _lanes()) == []


def test_no_idle_lane_still_flags_but_suggests_nothing():
    [diag] = _inversions(_ready(), _lanes(idle=()))
    assert diag.data["idle_profiles"] == []
    assert "No other lane is fully idle" in diag.detail
    assert not any(a.kind == "cli_hint" and "reassign" in a.payload.get("command", "")
                   for a in diag.actions)


def test_own_lane_is_never_offered_as_reroute_target():
    [diag] = _inversions(_ready(), _lanes(idle=("builder", "reviewer")))
    assert diag.data["idle_profiles"] == ["reviewer"]


def test_without_lane_snapshot_rule_stays_silent():
    assert _inversions(_ready(), None) == []


@pytest.fixture()
def board_home(monkeypatch):
    home = tempfile.mkdtemp(prefix="kanban_lane_inversion_")
    for prof in ("builder", "reviewer", "default"):
        os.makedirs(os.path.join(home, "profiles", prof), exist_ok=True)
        with open(os.path.join(home, "profiles", prof, "config.yaml"), "w") as fh:
            fh.write("{}\n")
    monkeypatch.setenv("HERMES_HOME", home)
    for mod in list(sys.modules):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def test_high_priority_card_behind_two_lower_holders_is_flagged_and_holders_survive(board_home):
    kb = board_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_diagnostics as kdiag

    def spawn_alive(*_args, **_kwargs):
        return os.getpid()

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        high_holder = kb.create_task(conn, title="holder", assignee="builder", priority=255)
        low_holder = kb.create_task(conn, title="holder", assignee="builder", priority=150)
    with kbc.connect_closing() as conn:
        first = kbd.dispatch_once(conn, spawn_fn=spawn_alive, max_in_progress_per_profile=2)
    assert sorted(s[0] for s in first.spawned) == sorted([high_holder, low_holder])

    with kbc.connect_closing() as conn:
        starved = kb.create_task(conn, title="release", assignee="builder", priority=225)
        before = {r["id"]: (r["status"], r["claim_lock"]) for r in conn.execute(
            "SELECT id, status, claim_lock FROM tasks WHERE id IN (?, ?)", (high_holder, low_holder))}
    with kbc.connect_closing() as conn:
        second = kbd.dispatch_once(conn, spawn_fn=spawn_alive, max_in_progress_per_profile=2)
        after = {r["id"]: (r["status"], r["claim_lock"]) for r in conn.execute(
            "SELECT id, status, claim_lock FROM tasks WHERE id IN (?, ?)", (high_holder, low_holder))}
        kb.create_task(conn, title="queued elsewhere", assignee="default")
        starved_row = conn.execute("SELECT * FROM tasks WHERE id = ?", (starved,)).fetchone()
        events = kb.list_events(conn, starved)
        runs = kb.list_runs(conn, starved)
        lanes = kbd.lane_occupancy(conn)

    assert second.spawned == []
    assert [c[0] for c in second.skipped_per_profile_capped] == [starved]
    assert second.crashed == [] and second.reclaimed == 0
    assert after == before
    assert all(status == "running" for status, _ in after.values())
    assert starved_row["status"] == "ready" and runs == []

    diags = kdiag.compute_task_diagnostics(
        starved_row, events, runs, now=int(time.time()) + 600, config=CAP_2, lanes=lanes)
    [diag] = [d for d in diags if d.kind == "lane_priority_inversion"]
    assert diag.data["outranked_holders"] == [low_holder]
    assert diag.data["idle_profiles"] == ["reviewer"]
