"""Regression suite for the t_6a6ac2d3 "superseded-but-alive worker" incident.

Timeline, reconstructed from the live board (ground truth):

* 15:39:58 the worker called ``block_task`` — run 2473 went terminal
  (``outcome='blocked'``), ``tasks.current_run_id`` became NULL and
  ``tasks.worker_pid`` was cleared.
* 15:40:38 the EM ran ``unblock`` — the card went back to ``ready``.
* The worker process for run 2473 was STILL ALIVE and kept working until
  20:25. Every ``kanban_heartbeat`` it made came back with the generic
  ``could not heartbeat t_6a6ac2d3 (unknown id or not running)``, which
  is indistinguishable from "you typo'd the id".
* The dispatcher emitted ``respawn_guarded {"reason":"active_pr"}`` every
  ~60s from 15:41 to 20:33 (~170 events) and never respawned the card,
  because rule 4's bypass list did not contain ``unblocked``.
* Meanwhile the health telemetry logged "stuck: ready queue non-empty ...
  0 workers spawned" — a false alarm about a dispatcher that was
  deliberately (if wrongly) deferring.

Three distinct defects, covered here:

1. the heartbeat has no way to say "your run was superseded, exit";
2. a superseded-but-alive worker is unreapable once ``tasks.worker_pid``
   is cleared;
3. the respawn guard's per-rule bypass lists deadlock the card.

Conventions (fixture shape, PR-seam stubbing) follow
``test_kanban_respawn_guard_changes_requested_deadlock.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest


PR_URL = "https://github.com/NousResearch/hermes-agent/pull/4774"


@pytest.fixture()
def kb(monkeypatch):
    """Fresh HERMES_HOME + kanban DB, with an 'a' profile that can spawn."""
    test_home = tempfile.mkdtemp(prefix="kanban_superseded_test_")
    for prof in ("a", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    # A dispatched worker inherits HERMES_KANBAN_* pins pointing at the REAL
    # board; strip them so this file can never write onto a live board.
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_LOGS_ROOT",
                "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK"):
        monkeypatch.delenv(var, raising=False)
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from tests.hermes_cli._kanban_modules import KanbanModules
    from hermes_cli import profiles
    # profile_exists resolves from HOME, not HERMES_HOME: treat assignees as real.
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    yield KanbanModules()


@pytest.fixture(autouse=True)
def _clear_pr_caches(kb):
    for name in ("_PR_GUARD_CACHE", "_PR_STATE_CACHE"):
        getattr(kb, name, {}).clear()
    yield
    for name in ("_PR_GUARD_CACHE", "_PR_STATE_CACHE"):
        getattr(kb, name, {}).clear()


def _stub_pr_status(kb, monkeypatch, status) -> None:
    """Stub every PR-metadata seam so no test shells out to ``gh``."""
    monkeypatch.setattr(
        kb, "_fetch_pr_status",
        lambda owner, repo, number: dict(status) if status else None,
        raising=False,
    )
    monkeypatch.setattr(
        kb, "_pr_url_is_open",
        lambda _url: bool(status) and status.get("state") == "OPEN",
        raising=False,
    )


def _backdate_comments(kb, conn, task_id, ts: int) -> None:
    """Move every comment on ``task_id`` to ``ts`` (event ordering control)."""
    conn.execute(
        "UPDATE task_comments SET created_at = ? WHERE task_id = ?",
        (ts, task_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 1. The incident, end to end: block -> unblock -> the next tick MUST spawn
# ---------------------------------------------------------------------------


def test_unblocked_card_with_open_pr_respawns(
    kb, monkeypatch, all_assignees_spawnable,
):
    """The exact t_6a6ac2d3 sequence must produce a spawn, not a guard."""
    _stub_pr_status(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [{"name": "test", "conclusion": "SUCCESS"}],
    })
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship the thing", assignee="a")
        kb.add_comment(conn, tid, "worker", f"Opened PR: {PR_URL}")
        conn.commit()

        # The worker claims, spawns, then blocks: run goes terminal, the
        # task-row pid is cleared, current_run_id becomes NULL.
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb.latest_run(conn, tid).id
        kb._set_worker_pid(conn, tid, 424242)
        assert kb.block_task(conn, tid, reason="needs a decision",
                             expected_run_id=run_id)
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.get_task(conn, tid).current_run_id is None

        # Before the unblock, the guard is doing its job: an open PR with
        # no deliberate re-queue behind it.
        assert kb.check_respawn_guard(conn, tid) == "active_pr"

        # The EM unblocks — NEW information (their decision) that only
        # reaches a worker through a fresh spawn's worker_context.
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"

        assert kb.check_respawn_guard(conn, tid) is None, (
            "an operator unblock AFTER the PR comment is a deliberate "
            "re-queue and must release the active_pr guard"
        )

        spawned_pids = []

        def spawn_fn(*a, **kw):
            spawned_pids.append(99001)
            return 99001

        res = kb.dispatch_once(conn, spawn_fn=spawn_fn)
        assert [t[0] for t in res.spawned] == [tid], (
            f"dispatch must spawn the unblocked card; got spawned="
            f"{res.spawned} respawn_guarded={res.respawn_guarded}"
        )
        assert res.respawn_guarded == []


# ---------------------------------------------------------------------------
# 2-4. Heartbeat directives
# ---------------------------------------------------------------------------


def test_heartbeat_from_superseded_run_returns_exit_directive(kb, monkeypatch):
    """The superseded worker must be told to stop, with the real ids."""
    import tools.kanban_tools as kt

    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship the thing", assignee="a")
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.latest_run(conn, tid).id
        assert kb.block_task(conn, tid, reason="decide please",
                             expected_run_id=run_id)
        assert kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid) is not None
        new_run_id = kb.latest_run(conn, tid).id
        assert new_run_id != run_id

        # DB layer: structured, and still falsy for legacy bool callers.
        hb = kb.heartbeat_worker(conn, tid, expected_run_id=run_id)
        assert not hb
        assert hb.superseded and not hb.unknown_task
        assert hb.expected_run_id == run_id
        assert hb.current_run_id == new_run_id
        assert hb.task_status == "running"

    # Tool layer: the message the worker actually reads.
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setattr(kt, "_enforce_worker_task_ownership", lambda _tid: None)
    monkeypatch.setattr(
        kt, "_reject_delegated_child_mutation", lambda _name: None,
    )
    out = kt._handle_heartbeat({"task_id": tid})

    assert "superseded" in out
    assert "EXIT IMMEDIATELY" in out
    assert "fresh dispatch" in out
    # Names the status and BOTH run ids — the generic wording is the defect.
    assert "'running'" in out
    assert str(run_id) in out
    assert str(new_run_id) in out
    assert "unknown id or not running" not in out, (
        "the generic message is exactly what let the t_6a6ac2d3 worker "
        "keep going for 4h45m"
    )


def test_heartbeat_with_unknown_task_id_still_reports_unknown(kb, monkeypatch):
    """A genuinely bogus id must NOT be reported as a supersede."""
    import tools.kanban_tools as kt

    with kb.connect_closing() as conn:
        hb = kb.heartbeat_worker(conn, "t_does_not_exist")
        assert not hb
        assert hb.unknown_task and not hb.superseded

    monkeypatch.setattr(kt, "_enforce_worker_task_ownership", lambda _tid: None)
    monkeypatch.setattr(
        kt, "_reject_delegated_child_mutation", lambda _name: None,
    )
    out = kt._handle_heartbeat({"task_id": "t_does_not_exist"})
    assert "unknown id" in out
    assert "superseded" not in out
    assert "EXIT IMMEDIATELY" not in out


def test_heartbeat_on_current_run_still_succeeds(kb, monkeypatch):
    """No regression for the healthy path (DB layer and tool layer)."""
    import tools.kanban_tools as kt

    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship the thing", assignee="a")
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.latest_run(conn, tid).id

        hb = kb.heartbeat_worker(conn, tid, note="alive",
                                 expected_run_id=run_id)
        assert hb  # truthy — legacy callers keep working
        assert hb.ok and not hb.superseded and not hb.unknown_task
        assert hb.current_run_id == run_id
        assert kb.get_task(conn, tid).last_heartbeat_at is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setattr(kt, "_enforce_worker_task_ownership", lambda _tid: None)
    monkeypatch.setattr(
        kt, "_reject_delegated_child_mutation", lambda _name: None,
    )
    out = kt._handle_heartbeat({"task_id": tid})
    assert '"ok": true' in out.lower().replace("'", '"') or '"ok"' in out
    assert "superseded" not in out


# Superseded-but-alive worker reaping (defect 2) is covered upstream by
# ``reap_terminal_workers`` (#111791), which keys on the closed run's retained
# pid + spawn fingerprint; see tests for that function.


# ---------------------------------------------------------------------------
# 7. The bypass is DERIVED, not a per-rule list
# ---------------------------------------------------------------------------


def test_requeue_after_the_pr_comment_bypasses_but_before_does_not(
    kb, monkeypatch,
):
    """The predicate compares against EACH rule's own evidence timestamp."""
    _stub_pr_status(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [{"name": "test", "conclusion": "SUCCESS"}],
    })
    with kb.connect_closing() as conn:
        # Case A: unblock BEFORE the PR comment -> the guard must HOLD.
        # (Otherwise any historical unblock would permanently disarm the
        # rule — the failure mode a single global timestamp would create.)
        early = kb.create_task(conn, title="early unblock", assignee="a")
        assert kb.block_task(conn, early, reason="x")
        assert kb.unblock_task(conn, early)
        kb.add_comment(conn, early, "worker", f"Opened PR: {PR_URL}")
        # Put the comment strictly after the unblock event.
        _backdate_comments(kb, conn, early, int(time.time()) + 5)
        assert kb.check_respawn_guard(conn, early) == "active_pr", (
            "an unblock that predates the PR comment must not bypass"
        )

        # Case B: unblock AFTER the PR comment -> the guard must RELEASE.
        late = kb.create_task(conn, title="late unblock", assignee="a")
        kb.add_comment(conn, late, "worker", f"Opened PR: {PR_URL}")
        _backdate_comments(kb, conn, late, int(time.time()) - 60)
        assert kb.check_respawn_guard(conn, late) == "active_pr"
        assert kb.block_task(conn, late, reason="x")
        assert kb.unblock_task(conn, late)
        assert kb.check_respawn_guard(conn, late) is None


# Upstream deliberately scopes the per-rule re-queue sets (rule 4 treats only
# handoffs as bypasses — a crash/reclaim must NOT re-spawn the PR's author,
# #111910). The local fix adds exactly the missing operator ``unblocked``.
@pytest.mark.parametrize("kind", ["unblocked"])
def test_every_requeue_kind_bypasses_every_timestamped_rule(
    kb, monkeypatch, kind,
):
    """One predicate, applied uniformly — not a list per rule.

    The same event stream must release BOTH timestamped rules
    (``recent_success`` on a completed run, ``active_pr`` on a PR
    comment). Rule 4 shipped with its own private bypass that lacked
    ``unblocked``; this asserts the shared derivation instead.
    """
    _stub_pr_status(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [{"name": "test", "conclusion": "SUCCESS"}],
    })
    with kb.connect_closing() as conn:
        # active_pr: PR comment, then the re-queue event.
        pr_task = kb.create_task(conn, title="pr", assignee="a")
        kb.add_comment(conn, pr_task, "worker", f"Opened PR: {PR_URL}")
        _backdate_comments(kb, conn, pr_task, int(time.time()) - 60)
        assert kb.check_respawn_guard(conn, pr_task) == "active_pr"

        # recent_success: a completed run, then the re-queue event.
        done_task = kb.create_task(conn, title="done", assignee="a")
        assert kb.claim_task(conn, done_task) is not None
        assert kb.complete_task(conn, done_task, summary="done")
        conn.execute(
            "UPDATE task_runs SET ended_at = ? WHERE task_id = ?",
            (int(time.time()) - 60, done_task),
        )
        conn.commit()
        assert kb.check_respawn_guard(conn, done_task) == "recent_success"

        for tid in (pr_task, done_task):
            with kb.write_txn(conn):
                kb._append_event(conn, tid, kind, {"synthetic": True})
            assert kb.check_respawn_guard(conn, tid) is None, (
                f"a {kind!r} event after the evidence is a deliberate "
                f"re-queue and must release every timestamped rule"
            )


def test_rate_limit_cooldown_is_not_bypassable(kb, monkeypatch):
    """Explicit non-regression: the cooldown is a timer, not stale evidence.

    Re-probing a quota bucket we have already proven empty just hammers
    it, so ``rate_limit_cooldown`` reports no evidence timestamp and the
    shared bypass cannot fire on it.
    """
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="throttled", assignee="a")
        assert kb.claim_task(conn, tid) is not None
        with kb.write_txn(conn):
            kb._end_run(conn, tid, outcome="rate_limited",
                        status="rate_limited", error="429 quota exceeded")
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,),
        )
        conn.commit()
        assert kb.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        assert kb.unblock_task(conn, tid) is False  # not blocked; emit by hand
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "unblocked", None)
        assert kb.check_respawn_guard(conn, tid) == "rate_limit_cooldown", (
            "a re-queue must not re-probe a quota bucket mid-cooldown"
        )


def test_green_pr_awaiting_review_still_guards(kb, monkeypatch):
    """Negative control: the guard must not widen into 'always respawn'."""
    _stub_pr_status(kb, monkeypatch, {
        "state": "OPEN",
        "reviewDecision": "REVIEW_REQUIRED",
        "statusCheckRollup": [
            {"name": "test", "conclusion": "SUCCESS"},
            {"name": "build", "conclusion": "SUCCESS"},
        ],
    })
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="ship the thing", assignee="a")
        kb.add_comment(conn, tid, "worker", f"Opened PR: {PR_URL}")
        conn.commit()
        assert kb.check_respawn_guard(conn, tid) == "active_pr"
