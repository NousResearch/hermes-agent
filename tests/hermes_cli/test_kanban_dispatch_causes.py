"""Tests for BUILD-263: dispatcher spawn-refusal cause tracking + aggregation.

Covers three things that used to be silently invisible to an operator
reading logs after a "0 workers spawned" dispatcher tick:

1. Exceptions raised while resolving a task's workspace or invoking
   ``spawn_fn`` are now logged (not just recorded on the task row via
   ``_record_spawn_failure``) and surfaced on ``DispatchResult.spawn_errors``.
2. Lost atomic-claim races (``claim_task`` / ``claim_review_task`` returning
   ``None``) are tracked on ``DispatchResult.claim_race`` instead of being a
   silent ``continue``.
3. ``summarize_dispatch_causes()`` aggregates all of the above (plus the
   pre-existing respawn-guard / quota / concurrency-cap / collision buckets)
   into the compact breakdown string the "dispatcher stuck" warning and
   Telegram escalation both use.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_task_status(conn: sqlite3.Connection, task_id: str, status: str) -> None:
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


# ---------------------------------------------------------------------------
# DispatchResult new fields
# ---------------------------------------------------------------------------


def test_dispatch_result_new_fields_default_empty():
    res = kb.DispatchResult()
    assert res.claim_race == []
    assert res.spawn_errors == []
    assert res.max_in_progress_deferred == 0


# ---------------------------------------------------------------------------
# Spawn / workspace-resolution exceptions are logged, not swallowed
# ---------------------------------------------------------------------------


def test_workspace_resolution_exception_is_logged_and_recorded(
    kanban_home, all_assignees_spawnable, monkeypatch, caplog,
):
    def boom_resolve(task, board=None):
        raise RuntimeError("no venv at profile path")

    monkeypatch.setattr(kb, "resolve_workspace", boom_resolve)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="alice")
        with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
            res = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 123)

    assert len(res.spawn_errors) == 1
    tid, err = res.spawn_errors[0]
    assert tid == t
    assert "no venv at profile path" in err
    assert any(
        "workspace resolution failed" in rec.message and t in rec.message
        for rec in caplog.records
    ), f"expected a WARNING log naming {t}; got: {[r.message for r in caplog.records]}"


def test_spawn_fn_exception_is_logged_and_recorded(
    kanban_home, all_assignees_spawnable, caplog,
):
    def boom(task, workspace):
        raise RuntimeError("profile venv missing python3")

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="alice")
        with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
            res = kb.dispatch_once(conn, spawn_fn=boom)

    assert len(res.spawn_errors) == 1
    tid, err = res.spawn_errors[0]
    assert tid == t
    assert "profile venv missing python3" in err
    assert any(
        "spawn_fn raised" in rec.message and t in rec.message
        for rec in caplog.records
    ), f"expected a WARNING log naming {t}; got: {[r.message for r in caplog.records]}"
    # Existing behavior preserved: task returns to ready for the next tick.
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"


def test_review_spawn_fn_exception_is_logged_and_recorded(
    kanban_home, all_assignees_spawnable, caplog,
):
    def boom(task, workspace, board=None):
        raise RuntimeError("review worker crashed on start")

    with kb.connect() as conn:
        t = kb.create_task(conn, title="review me", assignee="alice")
        _set_task_status(conn, t, "review")
        with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
            res = kb.dispatch_once(conn, spawn_fn=boom)

    assert len(res.spawn_errors) == 1
    tid, err = res.spawn_errors[0]
    assert tid == t
    assert "review worker crashed on start" in err
    assert any(
        "review spawn_fn raised" in rec.message and t in rec.message
        for rec in caplog.records
    ), f"expected a WARNING log naming {t}; got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Claim races
# ---------------------------------------------------------------------------


def test_ready_claim_race_is_tracked_not_swallowed(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """When claim_task loses the race (returns None), the task id must show
    up in DispatchResult.claim_race instead of vanishing with a bare
    `continue` — the pre-BUILD-263 behavior had zero visibility here."""
    monkeypatch.setattr(kb, "claim_task", lambda *a, **kw: None)

    spawn_calls = []

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="alice")
        res = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: spawn_calls.append(1))

    assert t in res.claim_race
    assert res.spawned == []
    assert spawn_calls == []


def test_review_claim_race_is_tracked_not_swallowed(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kb, "claim_review_task", lambda *a, **kw: None)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="review me", assignee="alice")
        _set_task_status(conn, t, "review")
        res = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: None)

    assert t in res.claim_race
    assert res.spawned == []


# ---------------------------------------------------------------------------
# max_in_progress_deferred
# ---------------------------------------------------------------------------


def test_max_in_progress_deferred_counts_ready_rows_at_cap(
    kanban_home, all_assignees_spawnable,
):
    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="a", assignee="alice")
        t2 = kb.create_task(conn, title="b", assignee="bob")
        kb.claim_task(conn, t1)
        kb.claim_task(conn, t2)
        # Two more ready — cap is already met, so both are deferred.
        kb.create_task(conn, title="c", assignee="bob")
        kb.create_task(conn, title="d", assignee="alice")
        res = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: None, max_in_progress=2)

    assert res.max_in_progress_deferred == 2
    assert res.spawned == []


def test_max_in_progress_deferred_zero_when_headroom_exists(
    kanban_home, all_assignees_spawnable,
):
    with kb.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        res = kb.dispatch_once(
            conn, spawn_fn=lambda *a, **k: 1, max_in_progress=5,
        )

    assert res.max_in_progress_deferred == 0
    assert len(res.spawned) == 1


# ---------------------------------------------------------------------------
# summarize_dispatch_causes
# ---------------------------------------------------------------------------


def test_summarize_dispatch_causes_empty_when_nothing_to_report():
    assert kb.summarize_dispatch_causes([kb.DispatchResult()]) == ""
    assert kb.summarize_dispatch_causes([]) == ""


def test_summarize_dispatch_causes_matches_spec_example_format():
    """The exact example format from the BUILD-263 spec:
    "causes: respawn_guarded(active_pr)=3, quota=1" """
    res = kb.DispatchResult(
        respawn_guarded=[
            ("t1", "active_pr"),
            ("t2", "active_pr"),
            ("t3", "active_pr"),
            ("t4", "blocker_auth"),
        ],
    )
    assert (
        kb.summarize_dispatch_causes([res])
        == "respawn_guarded(active_pr)=3, quota=1"
    )


def test_summarize_dispatch_causes_quota_folds_rate_limit_cooldown_and_rate_limited():
    res = kb.DispatchResult(
        respawn_guarded=[("t1", "rate_limit_cooldown")],
        rate_limited=["t2", "t3"],
    )
    assert kb.summarize_dispatch_causes([res]) == "quota=3"


def test_summarize_dispatch_causes_buckets_each_new_cause():
    res = kb.DispatchResult(
        claim_race=["t1"],
        spawn_errors=[("t2", "boom"), ("t3", "boom2")],
        max_in_progress_deferred=4,
        skipped_per_profile_capped=[("t4", "alice", 2)],
        workspace_collisions=[("t5", "t6")],
        skipped_unassigned=["t7"],
        skipped_nonspawnable=["t8"],
        skipped_locked=True,
    )
    causes = kb.summarize_dispatch_causes([res])
    parts = {p.split("=")[0]: int(p.split("=")[1]) for p in causes.split(", ")}
    assert parts == {
        "concurrency_cap": 4,
        "spawn_exception": 2,
        "claim_race": 1,
        "concurrency_cap(per_profile)": 1,
        "workspace_collision": 1,
        "unassigned": 1,
        "nonspawnable": 1,
        "dispatch_lock_contended": 1,
    }


def test_summarize_dispatch_causes_aggregates_across_multiple_results():
    """Across a bad-tick streak (or multiple boards in one tick), counts sum."""
    r1 = kb.DispatchResult(respawn_guarded=[("t1", "active_pr")])
    r2 = kb.DispatchResult(respawn_guarded=[("t2", "active_pr"), ("t3", "recent_success")])
    assert (
        kb.summarize_dispatch_causes([r1, r2])
        == "respawn_guarded(active_pr)=2, respawn_guarded(recent_success)=1"
    )


def test_summarize_dispatch_causes_skips_none_entries():
    """Defensive: a board that raised before producing a DispatchResult
    contributes a None to the accumulated list; must not crash."""
    res = kb.DispatchResult(respawn_guarded=[("t1", "active_pr")])
    assert (
        kb.summarize_dispatch_causes([None, res, None])
        == "respawn_guarded(active_pr)=1"
    )
