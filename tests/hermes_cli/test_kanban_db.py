"""Tests for the Kanban DB layer (hermes_cli.kanban_db)."""

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------

def test_init_db_is_idempotent(kanban_home):
    # Second call should not error or drop data.
    with kb.connect() as conn:
        kb.create_task(conn, title="persisted")
    kb.init_db()
    with kb.connect() as conn:
        tasks = kb.list_tasks(conn)
    assert len(tasks) == 1
    assert tasks[0].title == "persisted"


def test_init_creates_expected_tables(kanban_home):
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    names = {r["name"] for r in rows}
    assert {"tasks", "task_links", "task_comments", "task_events"} <= names


# ---------------------------------------------------------------------------
# Task creation + status inference
# ---------------------------------------------------------------------------

def test_create_task_no_parents_is_ready(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="alice")
        t = kb.get_task(conn, tid)
    assert t is not None
    assert t.status == "ready"
    assert t.assignee == "alice"
    assert t.workspace_kind == "scratch"


def test_create_task_with_parent_is_todo_until_parent_done(kanban_home):
    with kb.connect() as conn:
        p = kb.create_task(conn, title="parent")
        c = kb.create_task(conn, title="child", parents=[p])
        assert kb.get_task(conn, c).status == "todo"
        kb.complete_task(conn, p, result="ok")
        assert kb.get_task(conn, c).status == "ready"


def test_create_task_unknown_parent_errors(kanban_home):
    with kb.connect() as conn, pytest.raises(ValueError, match="unknown parent"):
        kb.create_task(conn, title="orphan", parents=["t_ghost"])


def test_workspace_kind_validation(kanban_home):
    with kb.connect() as conn, pytest.raises(ValueError, match="workspace_kind"):
        kb.create_task(conn, title="bad ws", workspace_kind="cloud")


# ---------------------------------------------------------------------------
# Links + dependency resolution
# ---------------------------------------------------------------------------

def test_link_demotes_ready_child_to_todo_when_parent_not_done(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        assert kb.get_task(conn, b).status == "ready"
        kb.link_tasks(conn, a, b)
        assert kb.get_task(conn, b).status == "todo"


def test_link_keeps_ready_child_when_parent_already_done(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        kb.complete_task(conn, a)
        b = kb.create_task(conn, title="b")
        assert kb.get_task(conn, b).status == "ready"
        kb.link_tasks(conn, a, b)
        assert kb.get_task(conn, b).status == "ready"


def test_link_rejects_self_loop(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        with pytest.raises(ValueError, match="itself"):
            kb.link_tasks(conn, a, a)


def test_link_detects_cycle(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b", parents=[a])
        c = kb.create_task(conn, title="c", parents=[b])
        with pytest.raises(ValueError, match="cycle"):
            kb.link_tasks(conn, c, a)
        with pytest.raises(ValueError, match="cycle"):
            kb.link_tasks(conn, b, a)


def test_recompute_ready_cascades_through_chain(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b", parents=[a])
        c = kb.create_task(conn, title="c", parents=[b])
        assert [kb.get_task(conn, x).status for x in (a, b, c)] == \
               ["ready", "todo", "todo"]
        kb.complete_task(conn, a)
        assert kb.get_task(conn, b).status == "ready"
        kb.complete_task(conn, b)
        assert kb.get_task(conn, c).status == "ready"


def test_recompute_ready_fan_in_waits_for_all_parents(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        c = kb.create_task(conn, title="c", parents=[a, b])
        kb.complete_task(conn, a)
        assert kb.get_task(conn, c).status == "todo"
        kb.complete_task(conn, b)
        assert kb.get_task(conn, c).status == "ready"


# ---------------------------------------------------------------------------
# Atomic claim (CAS)
# ---------------------------------------------------------------------------

def test_claim_once_wins_second_loses(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        first = kb.claim_task(conn, t, claimer="host:1")
        assert first is not None and first.status == "running"
        second = kb.claim_task(conn, t, claimer="host:2")
        assert second is None


def test_claim_fails_on_non_ready(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        # Move to todo by introducing an unsatisfied parent.
        p = kb.create_task(conn, title="p")
        kb.link_tasks(conn, p, t)
        assert kb.get_task(conn, t).status == "todo"
        assert kb.claim_task(conn, t) is None


def test_stale_claim_reclaimed(kanban_home, monkeypatch):
    import signal
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        killed: list[int] = []
        state = {"alive": True}

        def _signal(pid, sig):
            killed.append(sig)
            if sig == signal.SIGTERM:
                state["alive"] = False

        kb._set_worker_pid(conn, t, 12345)
        # Rewind claim_expires so it looks stale.
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 3600, t),
        )
        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: state["alive"])
        reclaimed = kb.release_stale_claims(conn, signal_fn=_signal)
        assert reclaimed == 1
        assert kb.get_task(conn, t).status == "ready"
        assert killed == [signal.SIGTERM]


def test_max_runtime_uses_current_run_start_after_retry(kanban_home):
    """A retry should get a fresh max-runtime window.

    ``tasks.started_at`` intentionally records the first time the task ever
    started. Runtime enforcement must therefore use the active
    ``task_runs.started_at`` row; otherwise every retry of an old task is
    immediately timed out again.
    """
    with kb.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        t = kb.create_task(
            conn, title="retry", assignee="a", max_runtime_seconds=10,
        )

        kb.claim_task(conn, t, claimer=f"{host}:first")
        first_run_id = kb.latest_run(conn, t).id
        old_started = int(time.time()) - 20
        conn.execute(
            "UPDATE tasks SET started_at = ?, worker_pid = ? WHERE id = ?",
            (old_started, 999999, t),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ?, worker_pid = ? WHERE id = ?",
            (old_started, 999999, first_run_id),
        )

        timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda _pid, _sig: None)
        assert timed_out == [t]
        assert kb.get_task(conn, t).status == "ready"

        kb.claim_task(conn, t, claimer=f"{host}:retry")
        retry_run = kb.latest_run(conn, t)
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?",
            (999999, t),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
            (999999, retry_run.id),
        )

        timed_out = kb.enforce_max_runtime(conn, signal_fn=lambda _pid, _sig: None)
        assert timed_out == []
        assert kb.get_task(conn, t).status == "running"


def test_heartbeat_extends_claim(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        claimer = "host:hb"
        kb.claim_task(conn, t, claimer=claimer, ttl_seconds=60)
        original = kb.get_task(conn, t).claim_expires
        # Rewind then heartbeat.
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (0, t))
        ok = kb.heartbeat_claim(conn, t, claimer=claimer, ttl_seconds=3600)
        assert ok
        new = kb.get_task(conn, t).claim_expires
        assert new > int(time.time()) + 3000


def test_concurrent_claims_only_one_wins(kanban_home):
    """Fire N threads claiming the same task; exactly one must win."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="race", assignee="a")

    def attempt(i):
        with kb.connect() as c:
            return kb.claim_task(c, t, claimer=f"host:{i}")

    n_workers = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(attempt, range(n_workers)))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert winners[0].status == "running"


# ---------------------------------------------------------------------------
# Complete / block / unblock / archive / assign
# ---------------------------------------------------------------------------

def test_complete_records_result(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        assert kb.complete_task(conn, t, result="done and dusted")
        task = kb.get_task(conn, t)
    assert task.status == "done"
    assert task.result == "done and dusted"
    assert task.completed_at is not None


def test_block_then_unblock(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        assert kb.block_task(conn, t, reason="need input")
        assert kb.get_task(conn, t).status == "blocked"
        assert kb.unblock_task(conn, t)
        assert kb.get_task(conn, t).status == "ready"


# ---------------------------------------------------------------------------
# Parent-completion invariant at the claim gate (RCA t_a6acd07d)
# ---------------------------------------------------------------------------

def test_claim_rejects_when_parents_not_done(kanban_home):
    """claim_task must refuse ready->running if any parent isn't 'done'.

    Simulates the create-then-link race: a task gets status='ready' via a
    racy writer while it still has undone parents. The claim gate must
    detect the violation, demote the child back to 'todo', append a
    'claim_rejected' event, and return None. Covers Fix 1 of the RCA.
    """
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(
            conn, title="child", assignee="a", parents=[parent],
        )
        # Child correctly starts 'todo' because parent is not 'done'.
        assert kb.get_task(conn, child).status == "todo"
        # Simulate the race: a racy writer force-promotes the child to
        # 'ready' while parent is still pending.
        conn.execute(
            "UPDATE tasks SET status='ready' WHERE id=?", (child,),
        )
        conn.commit()
        assert kb.get_task(conn, child).status == "ready"

        result = kb.claim_task(conn, child, claimer="host:1")

    assert result is None
    with kb.connect() as conn:
        assert kb.get_task(conn, child).status == "todo"
        events = conn.execute(
            "SELECT kind, payload FROM task_events "
            "WHERE task_id = ? ORDER BY id",
            (child,),
        ).fetchall()
    kinds = [e["kind"] for e in events]
    assert "claim_rejected" in kinds
    # No 'claimed' event was emitted for the blocked attempt.
    assert "claimed" not in kinds


def test_claim_succeeds_once_parents_done(kanban_home):
    """After parents complete, recompute_ready -> claim_task must succeed."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(
            conn, title="child", assignee="a", parents=[parent],
        )
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="ok")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"
        claimed = kb.claim_task(conn, child, claimer="host:1")
    assert claimed is not None
    assert claimed.status == "running"


def test_create_with_parents_stays_todo_until_parents_done(kanban_home):
    """kanban_create(parents=[...]) must land in 'todo' and only promote on parent done."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(
            conn, title="child", assignee="a", parents=[parent],
        )
        assert kb.get_task(conn, child).status == "todo"
        # Dispatcher tick between create and some later event must NOT
        # produce a winner for this child.
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, child).status == "todo"
        # Complete parent; complete_task internally runs recompute_ready,
        # which promotes the child to 'ready'.
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="ok")
        assert kb.get_task(conn, child).status == "ready"


def test_unblock_with_pending_parents_goes_to_todo(kanban_home):
    """unblock_task must re-gate on parent completion (Fix 3).

    A task blocked while parents are still in progress must return to
    'todo' (not 'ready') on unblock. Otherwise the dispatcher will claim
    it immediately, repeating Bug 2 from the RCA.
    """
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(
            conn, title="child", assignee="a", parents=[parent],
        )
        # Force child into 'blocked' regardless of parent progress
        # (simulates a worker that self-blocked, or an operator block).
        conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=?", (child,),
        )
        conn.commit()
        assert kb.unblock_task(conn, child)
        assert kb.get_task(conn, child).status == "todo"
        # After parent completes + recompute, the child is ready.
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="ok")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_unblock_without_parents_goes_to_ready(kanban_home):
    """Parent-free unblock still produces 'ready' (behavior preserved)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="lone", assignee="a")
        kb.claim_task(conn, t)
        assert kb.block_task(conn, t, reason="need input")
        assert kb.unblock_task(conn, t)
        assert kb.get_task(conn, t).status == "ready"


def test_assign_refuses_while_running(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        with pytest.raises(RuntimeError, match="currently running"):
            kb.assign_task(conn, t, "b")


def test_assign_reassigns_when_not_running(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        assert kb.assign_task(conn, t, "b")
        assert kb.get_task(conn, t).assignee == "b"


def test_assignee_normalized_to_lowercase_on_create_and_assign(kanban_home):
    """Dashboard/CLI may pass title-cased profile labels; DB + spawn use canonical id."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="cased", assignee="Jules")
        assert kb.get_task(conn, tid).assignee == "jules"
        assert kb.assign_task(conn, tid, "Librarian")
        assert kb.get_task(conn, tid).assignee == "librarian"


def test_list_tasks_assignee_filter_case_insensitive(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="q", assignee="jules")
        found = kb.list_tasks(conn, assignee="Jules")
        assert len(found) == 1 and found[0].id == tid


def test_archive_hides_from_default_list(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        kb.complete_task(conn, t)
        assert kb.archive_task(conn, t)
        assert len(kb.list_tasks(conn)) == 0
        assert len(kb.list_tasks(conn, include_archived=True)) == 1


# ---------------------------------------------------------------------------
# Comments / events / worker context
# ---------------------------------------------------------------------------

def test_comments_recorded_in_order(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        kb.add_comment(conn, t, "user", "first")
        kb.add_comment(conn, t, "researcher", "second")
        comments = kb.list_comments(conn, t)
    assert [c.body for c in comments] == ["first", "second"]
    assert [c.author for c in comments] == ["user", "researcher"]


def test_empty_comment_rejected(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        with pytest.raises(ValueError, match="body is required"):
            kb.add_comment(conn, t, "user", "")


def test_events_capture_lifecycle(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        kb.complete_task(conn, t, result="ok")
        events = kb.list_events(conn, t)
    kinds = [e.kind for e in events]
    assert "created" in kinds
    assert "claimed" in kinds
    assert "completed" in kinds


def test_worker_context_includes_parent_results_and_comments(kanban_home):
    with kb.connect() as conn:
        p = kb.create_task(conn, title="p")
        kb.complete_task(conn, p, result="PARENT_RESULT_MARKER")
        c = kb.create_task(conn, title="child", parents=[p])
        kb.add_comment(conn, c, "user", "CLARIFICATION_MARKER")
        ctx = kb.build_worker_context(conn, c)
    assert "PARENT_RESULT_MARKER" in ctx
    assert "CLARIFICATION_MARKER" in ctx
    assert c in ctx
    assert "child" in ctx


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def test_dispatch_dry_run_does_not_claim(kanban_home, all_assignees_spawnable):
    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="a", assignee="alice")
        t2 = kb.create_task(conn, title="b", assignee="bob")
        res = kb.dispatch_once(conn, dry_run=True)
    assert {s[0] for s in res.spawned} == {t1, t2}
    with kb.connect() as conn:
        # Dry run must NOT mutate status.
        assert kb.get_task(conn, t1).status == "ready"
        assert kb.get_task(conn, t2).status == "ready"


def test_dispatch_skips_unassigned(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="floater")
        res = kb.dispatch_once(conn, dry_run=True)
    assert t in res.skipped_unassigned
    assert t not in res.skipped_nonspawnable
    assert not res.spawned


def test_dispatch_skips_nonspawnable_into_separate_bucket(kanban_home, monkeypatch):
    """Tasks whose assignee fails profile_exists() must NOT land in
    ``skipped_unassigned`` (which is operator-actionable) — they go in
    the dedicated ``skipped_nonspawnable`` bucket so health telemetry
    can suppress false-positive "stuck" warnings."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        t = kb.create_task(conn, title="for-terminal", assignee="orion-cc")
        res = kb.dispatch_once(conn, dry_run=True)
    assert t in res.skipped_nonspawnable
    assert t not in res.skipped_unassigned
    assert not res.spawned


def test_has_spawnable_ready_false_when_only_terminal_lanes(kanban_home, monkeypatch):
    """``has_spawnable_ready`` returns False when every ready task is
    assigned to a control-plane lane — used by gateway/CLI dispatchers
    to silence the stuck-warn while terminals still have queued work."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        kb.create_task(conn, title="t1", assignee="orion-cc")
        kb.create_task(conn, title="t2", assignee="orion-research")
        assert kb.has_spawnable_ready(conn) is False


def test_has_spawnable_ready_true_when_real_profile_present(kanban_home, monkeypatch):
    """``has_spawnable_ready`` returns True as soon as ANY ready task
    has an assignee that maps to a real Hermes profile — preserves the
    real "stuck" signal when a daily/agent task is queued."""
    from hermes_cli import profiles
    monkeypatch.setattr(
        profiles, "profile_exists", lambda name: name == "daily"
    )
    with kb.connect() as conn:
        kb.create_task(conn, title="terminal-task", assignee="orion-cc")
        kb.create_task(conn, title="hermes-task", assignee="daily")
        assert kb.has_spawnable_ready(conn) is True


def test_has_spawnable_ready_false_on_empty_queue(kanban_home):
    """Empty queue is the trivial false case — no ready tasks at all."""
    with kb.connect() as conn:
        assert kb.has_spawnable_ready(conn) is False


def test_dispatch_promotes_ready_and_spawns(kanban_home, all_assignees_spawnable):
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append((task.id, task.assignee, workspace))

    with kb.connect() as conn:
        p = kb.create_task(conn, title="p", assignee="alice")
        c = kb.create_task(conn, title="c", assignee="bob", parents=[p])
        # Finish parent outside dispatch; promotion happens inside.
        kb.complete_task(conn, p)
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
    # Spawned c (a was already done when dispatch was called).
    assert len(spawns) == 1
    assert spawns[0][0] == c
    assert spawns[0][1] == "bob"
    # c is now running
    with kb.connect() as conn:
        assert kb.get_task(conn, c).status == "running"


def test_dispatch_spawn_failure_releases_claim(kanban_home, all_assignees_spawnable):
    def boom(task, workspace):
        raise RuntimeError("spawn failed")

    with kb.connect() as conn:
        t = kb.create_task(conn, title="boom", assignee="alice")
        kb.dispatch_once(conn, spawn_fn=boom)
        # Must return to ready so the next tick can retry.
        assert kb.get_task(conn, t).status == "ready"
        assert kb.get_task(conn, t).claim_lock is None


def test_dispatch_reclaims_stale_before_spawning(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="alice")
        kb.claim_task(conn, t)
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 1, t),
        )
        res = kb.dispatch_once(conn, dry_run=True)
    assert res.reclaimed == 1


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def test_scratch_workspace_created_under_hermes_home(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
    assert ws.exists()
    assert ws.is_dir()
    assert "kanban" in str(ws)


def test_dir_workspace_honors_given_path(kanban_home, tmp_path):
    target = tmp_path / "my-vault"
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="biz", workspace_kind="dir", workspace_path=str(target)
        )
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
    assert ws == target
    assert ws.exists()


def test_worktree_workspace_returns_intended_path(kanban_home, tmp_path):
    target = str(tmp_path / ".worktrees" / "my-task")
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="ship", workspace_kind="worktree", workspace_path=target
        )
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
    # We do NOT auto-create worktrees; the worker's skill handles that.
    assert str(ws) == target


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------

def test_tenant_column_filters_listings(kanban_home):
    with kb.connect() as conn:
        kb.create_task(conn, title="a1", tenant="biz-a")
        kb.create_task(conn, title="b1", tenant="biz-b")
        kb.create_task(conn, title="shared")  # no tenant
        biz_a = kb.list_tasks(conn, tenant="biz-a")
        biz_b = kb.list_tasks(conn, tenant="biz-b")
    assert [t.title for t in biz_a] == ["a1"]
    assert [t.title for t in biz_b] == ["b1"]


def test_tenant_propagates_to_events(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="tenant-task", tenant="biz-a")
        events = kb.list_events(conn, t)
    # The "created" event should have tenant in its payload.
    created = [e for e in events if e.kind == "created"]
    assert created and created[0].payload.get("tenant") == "biz-a"


# ---------------------------------------------------------------------------
# Shared-board path resolution (issue #19348)
#
# The kanban board is a cross-profile coordination primitive: a worker
# spawned with `hermes -p <profile>` must read/write the same kanban.db
# as the dispatcher that claimed the task. These tests exercise the
# path-resolution layer directly and would have caught the regression
# where `kanban_db_path()` resolved to the active profile's HERMES_HOME.
# ---------------------------------------------------------------------------

class TestSharedBoardPaths:
    """`kanban_home`/`kanban_db_path`/`workspaces_root`/`worker_log_path`
    must anchor at the **shared root**, not the active profile's HERMES_HOME."""

    def _set_home(self, monkeypatch, tmp_path, hermes_home):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)

    def test_default_install_anchors_at_home_dot_hermes(
        self, tmp_path, monkeypatch
    ):
        # Standard install: HERMES_HOME == ~/.hermes, no profile active.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        self._set_home(monkeypatch, tmp_path, default_home)

        assert kb.kanban_home() == default_home
        assert kb.kanban_db_path() == default_home / "kanban.db"
        assert kb.workspaces_root() == default_home / "kanban" / "workspaces"
        assert (
            kb.worker_log_path("t_demo")
            == default_home / "kanban" / "logs" / "t_demo.log"
        )

    def test_profile_worker_resolves_to_shared_root(
        self, tmp_path, monkeypatch
    ):
        # Reproduces the bug: dispatcher uses ~/.hermes/kanban.db,
        # worker spawned with -p <profile> previously resolved to
        # ~/.hermes/profiles/<profile>/kanban.db. After the fix both
        # converge on ~/.hermes/kanban.db.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, profile_home)

        # All four resolvers must anchor at the shared root, not the
        # profile-local HERMES_HOME.
        assert kb.kanban_home() == default_home
        assert kb.kanban_db_path() == default_home / "kanban.db"
        assert kb.workspaces_root() == default_home / "kanban" / "workspaces"
        assert (
            kb.worker_log_path("t_0d214f19")
            == default_home / "kanban" / "logs" / "t_0d214f19.log"
        )

        # Sanity: the profile-local path that used to be returned is
        # explicitly NOT what we resolve to anymore.
        assert kb.kanban_db_path() != profile_home / "kanban.db"

    def test_dispatcher_and_profile_worker_converge(
        self, tmp_path, monkeypatch
    ):
        # End-to-end convergence: resolve the path under each side's
        # HERMES_HOME and confirm equality. This is the property the
        # dispatcher/worker handoff actually depends on.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "coder"
        profile_home.mkdir(parents=True)

        # Dispatcher's perspective.
        self._set_home(monkeypatch, tmp_path, default_home)
        dispatcher_db = kb.kanban_db_path()
        dispatcher_ws = kb.workspaces_root()
        dispatcher_log = kb.worker_log_path("t_handoff")

        # Worker's perspective (profile activated by `hermes -p coder`).
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        worker_db = kb.kanban_db_path()
        worker_ws = kb.workspaces_root()
        worker_log = kb.worker_log_path("t_handoff")

        assert dispatcher_db == worker_db
        assert dispatcher_ws == worker_ws
        assert dispatcher_log == worker_log

    def test_docker_custom_hermes_home_uses_env_path_directly(
        self, tmp_path, monkeypatch
    ):
        # Docker / custom deployment: HERMES_HOME points outside ~/.hermes.
        # `get_default_hermes_root()` returns env_home directly when it
        # is not a `<root>/profiles/<name>` shape and not under
        # `Path.home() / ".hermes"`.
        custom_root = tmp_path / "opt" / "hermes"
        custom_root.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, custom_root)

        assert kb.kanban_home() == custom_root
        assert kb.kanban_db_path() == custom_root / "kanban.db"

    def test_docker_profile_layout_uses_grandparent(
        self, tmp_path, monkeypatch
    ):
        # Docker profile shape: HERMES_HOME=/opt/hermes/profiles/coder;
        # `get_default_hermes_root()` walks up to /opt/hermes because
        # the immediate parent dir is named "profiles".
        custom_root = tmp_path / "opt" / "hermes"
        profile = custom_root / "profiles" / "coder"
        profile.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, profile)

        assert kb.kanban_home() == custom_root
        assert kb.kanban_db_path() == custom_root / "kanban.db"

    def test_explicit_override_via_hermes_kanban_home(
        self, tmp_path, monkeypatch
    ):
        # Explicit override: HERMES_KANBAN_HOME beats every other
        # resolution rule.
        default_home = tmp_path / ".hermes"
        profile_home = default_home / "profiles" / "any"
        profile_home.mkdir(parents=True)
        override = tmp_path / "shared-board"
        override.mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(override))

        assert kb.kanban_home() == override
        assert kb.kanban_db_path() == override / "kanban.db"
        assert kb.workspaces_root() == override / "kanban" / "workspaces"

    def test_empty_override_falls_through(self, tmp_path, monkeypatch):
        # Empty/whitespace override is treated as unset.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        monkeypatch.setenv("HERMES_KANBAN_HOME", "   ")

        assert kb.kanban_home() == default_home

    def test_dispatcher_and_worker_share_a_real_database(
        self, tmp_path, monkeypatch
    ):
        # Belt-and-suspenders: round-trip a task across the two
        # HERMES_HOME perspectives via a real SQLite file. Without the
        # fix the worker would open a different file and see no rows.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)

        # Dispatcher creates the board and a task.
        self._set_home(monkeypatch, tmp_path, default_home)
        kb.init_db()
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="cross-profile")

        # Worker switches to the profile HERMES_HOME and reads.
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        with kb.connect() as conn:
            task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.title == "cross-profile"

    def test_hermes_kanban_db_pin_beats_kanban_home(
        self, tmp_path, monkeypatch
    ):
        # HERMES_KANBAN_DB pins the file path directly and beats both
        # HERMES_KANBAN_HOME and the `get_default_hermes_root()` path.
        # This is the env the dispatcher injects into workers.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        umbrella = tmp_path / "umbrella"
        umbrella.mkdir()
        pinned_db = tmp_path / "pinned" / "board.db"
        pinned_db.parent.mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(umbrella))
        monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned_db))

        assert kb.kanban_db_path() == pinned_db
        # workspaces_root still follows HERMES_KANBAN_HOME -- the pins
        # are independent.
        assert kb.workspaces_root() == umbrella / "kanban" / "workspaces"

    def test_hermes_kanban_workspaces_root_pin_beats_kanban_home(
        self, tmp_path, monkeypatch
    ):
        # HERMES_KANBAN_WORKSPACES_ROOT pins the workspaces root directly.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        umbrella = tmp_path / "umbrella"
        umbrella.mkdir()
        pinned_ws = tmp_path / "pinned-workspaces"
        pinned_ws.mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(umbrella))
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(pinned_ws))

        assert kb.workspaces_root() == pinned_ws
        # kanban_db_path still follows HERMES_KANBAN_HOME.
        assert kb.kanban_db_path() == umbrella / "kanban.db"

    def test_empty_per_path_overrides_fall_through(
        self, tmp_path, monkeypatch
    ):
        # Empty/whitespace pins are treated as unset, same as
        # HERMES_KANBAN_HOME.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(default_home))
        monkeypatch.setenv("HERMES_KANBAN_DB", "   ")
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", "")

        assert kb.kanban_db_path() == default_home / "kanban.db"
        assert kb.workspaces_root() == default_home / "kanban" / "workspaces"

    def test_dispatcher_spawn_injects_kanban_db_and_workspaces_root(
        self, tmp_path, monkeypatch
    ):
        # The dispatcher's `_default_spawn` must inject HERMES_KANBAN_DB
        # and HERMES_KANBAN_WORKSPACES_ROOT into the worker env so the
        # worker converges on the dispatcher's paths even when the
        # `-p <profile>` flag rewrites HERMES_HOME.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        self._set_home(monkeypatch, tmp_path, default_home)

        captured = {}

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["env"] = kwargs.get("env", {})
                self.pid = 4242

        monkeypatch.setattr("subprocess.Popen", _FakePopen)

        task = kb.Task(
            id="t_dispatch_env",
            title="x",
            body=None,
            assignee="coder",
            status="ready",
            priority=0,
            created_by=None,
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="scratch",
            workspace_path=None,
            claim_lock=None,
            claim_expires=None,
            tenant=None,
        )
        kb._default_spawn(task, str(tmp_path / "ws"))

        env = captured["env"]
        assert env["HERMES_KANBAN_DB"] == str(default_home / "kanban.db")
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(
            default_home / "kanban" / "workspaces"
        )
        assert env["HERMES_KANBAN_TASK"] == "t_dispatch_env"


# ---------------------------------------------------------------------------
# latest_summary / latest_summaries — surface task_runs.summary handoffs
# ---------------------------------------------------------------------------

def test_latest_summary_returns_none_when_no_runs(kanban_home):
    """A freshly-created task has no runs and therefore no summary."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="fresh", assignee="alice")
        assert kb.latest_summary(conn, t) is None


def test_latest_summary_returns_summary_after_complete(kanban_home):
    """``complete_task(summary=...)`` is the canonical kanban-worker
    handoff; ``latest_summary`` must surface it so dashboards/CLI can
    render what the worker actually did."""
    handoff = "shipped 3 files, ran tests, opened PR #42"
    with kb.connect() as conn:
        t = kb.create_task(conn, title="work", assignee="alice")
        kb.complete_task(conn, t, summary=handoff)
        assert kb.latest_summary(conn, t) == handoff


def test_latest_summary_picks_newest_when_multiple_runs(kanban_home):
    """When a task has been re-run (block → unblock → complete), the
    newest run's summary wins. We unblock to take the task back to
    ``ready``, then complete a second time and verify the second
    summary surfaces."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="retry", assignee="alice")
        kb.complete_task(conn, t, summary="first attempt")
        # Move back to ready by direct SQL — block_task / unblock_task
        # paths require an active claim, but we just want a second run
        # row to exist with a later ended_at.
        conn.execute(
            "UPDATE tasks SET status='ready', completed_at=NULL WHERE id=?",
            (t,),
        )
        # Sleep 1s so the second run's ended_at is provably later than
        # the first (complete_task uses int(time.time())).
        time.sleep(1.05)
        kb.complete_task(conn, t, summary="second attempt — final")
        assert kb.latest_summary(conn, t) == "second attempt — final"


def test_latest_summary_skips_empty_string(kanban_home):
    """A run with an empty-string summary should not mask an earlier
    populated one — empty strings carry no information."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="t", assignee="alice")
        kb.complete_task(conn, t, summary="real handoff")
        # Inject a later run with empty summary directly. Workers
        # writing "" instead of None is a real shape we want to ignore.
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at, ended_at, "
            "outcome, summary) VALUES (?, 'done', ?, ?, 'completed', ?)",
            (t, int(time.time()) + 1, int(time.time()) + 2, ""),
        )
        conn.commit()
        assert kb.latest_summary(conn, t) == "real handoff"


def test_latest_summaries_batch_omits_tasks_without_summary(kanban_home):
    """``latest_summaries`` is the dashboard's N+1 escape hatch — it
    must return only entries for tasks that actually have a summary,
    keep the per-task latest, and accept an empty input gracefully."""
    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="a", assignee="alice")
        t2 = kb.create_task(conn, title="b", assignee="bob")
        t3 = kb.create_task(conn, title="c", assignee="carol")
        kb.complete_task(conn, t1, summary="alpha")
        kb.complete_task(conn, t3, summary="charlie")
        out = kb.latest_summaries(conn, [t1, t2, t3])
        assert out == {t1: "alpha", t3: "charlie"}
        # Empty input → empty dict, no SQL syntax error from "IN ()".
        assert kb.latest_summaries(conn, []) == {}



# ---------------------------------------------------------------------------
# NFS / network-filesystem fallback (see hermes_state.apply_wal_with_fallback)
# ---------------------------------------------------------------------------

def test_connect_falls_back_to_delete_on_locking_protocol(kanban_home, caplog):
    """kanban_db.connect() must handle ``locking protocol`` on NFS/SMB.

    Without this fallback, the gateway's kanban dispatcher crashes every
    60s and the kanban migration (``consecutive_failures`` ADD COLUMN) is
    retried forever — which is what the real-world user report shows
    (see hermes-agent issue #22032).
    """
    import sqlite3 as _sqlite3
    from unittest.mock import patch as _patch

    # Clear module cache so a fresh connect() is attempted
    kb._INITIALIZED_PATHS.clear()

    real_connect = _sqlite3.connect

    class _WalBlockingConnection(_sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if "journal_mode=wal" in sql.lower().replace(" ", ""):
                raise _sqlite3.OperationalError("locking protocol")
            return super().execute(sql, *args, **kwargs)

    def wal_blocking_connect(*args, **kwargs):
        return real_connect(
            *args, factory=_WalBlockingConnection, **kwargs
        )

    with _patch("hermes_cli.kanban_db.sqlite3.connect", side_effect=wal_blocking_connect):
        with caplog.at_level("WARNING", logger="hermes_state"):
            conn = kb.connect()

    # One fallback warning, naming kanban.db
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "kanban.db" in r.getMessage()
    ]
    assert len(warnings) >= 1, (
        f"Expected a kanban.db WARNING, got: {[r.getMessage() for r in caplog.records]}"
    )

    # DB still usable end-to-end — create + list a task
    t = kb.create_task(conn, title="post-fallback task")
    tasks = kb.list_tasks(conn)
    assert any(row.id == t for row in tasks)
    conn.close()


def test_unlink_tasks_triggers_recompute_ready(kanban_home):
    """Regression test for issue #22459.

    Removing a dependency via unlink_tasks must immediately promote the child
    to ready when all remaining parents are done — same contract as
    complete_task and unblock_task.

    Before the fix, child stayed 'todo' indefinitely after unlink; only the
    next dispatcher tick or a manual 'hermes kanban recompute' would promote it.
    """
    with kb.connect() as conn:
        # A is done.
        a = kb.create_task(conn, title="parent-done")
        kb.complete_task(conn, a)

        # C is running (not done) — blocks child B.
        c = kb.create_task(conn, title="parent-running")
        kb.claim_task(conn, c, claimer="worker:1")

        # B depends on both A (done) and C (running) → stays todo.
        b = kb.create_task(conn, title="child", parents=[a, c])
        assert kb.get_task(conn, b).status == "todo"

        # Remove the blocking dependency C → B.
        removed = kb.unlink_tasks(conn, c, b)
        assert removed is True

        # B's only remaining parent is A (done) → must be ready immediately.
        assert kb.get_task(conn, b).status == "ready", (
            "child should promote to ready immediately after unlink_tasks "
            "removes its last blocking dependency"
        )
# ---------------------------------------------------------------------------
# _add_column_if_missing / _migrate_add_optional_columns idempotency (#21708)
# ---------------------------------------------------------------------------

def test_add_column_if_missing_is_idempotent_on_race(kanban_home):
    """``_add_column_if_missing`` must swallow 'duplicate column name' errors.

    Regression for #21708: the kanban dispatcher opens the DB twice per tick
    (once via _tick_once_for_board, once via init_db's discard-and-reconnect
    path).  A second concurrent connection runs _migrate_add_optional_columns
    before the first one commits, so ALTER TABLE raises OperationalError with
    'duplicate column name: consecutive_failures'.  Without the idempotency
    guard that crashes the dispatcher on the first tick after every restart.
    """
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT NOT NULL)"
    )

    # First call adds the column — returns True.
    added = kb._add_column_if_missing(conn, "tasks", "extra_col", "extra_col TEXT")
    assert added is True
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "extra_col" in cols

    # Second call on same connection — column already exists — must return
    # False without raising, simulating the race the dispatcher hits.
    added_again = kb._add_column_if_missing(
        conn, "tasks", "extra_col", "extra_col TEXT"
    )
    assert added_again is False

    conn.close()


def test_migrate_add_optional_columns_tolerates_concurrent_migration(kanban_home):
    """Full _migrate_add_optional_columns must not raise when columns already
    exist (issue #21708 race window — two connections migrate concurrently)."""
    import sqlite3

    # Schema already in fully-migrated state (all optional columns present).
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_failure_error TEXT,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            current_run_id INTEGER,
            workflow_template_id TEXT,
            current_step_key TEXT,
            skills TEXT,
            max_retries INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE task_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id    TEXT NOT NULL DEFAULT '',
            run_id     INTEGER,
            kind       TEXT NOT NULL DEFAULT '',
            payload    TEXT,
            created_at INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    # Running migration on an already-migrated schema must not raise.
    kb._migrate_add_optional_columns(conn)
    conn.close()
