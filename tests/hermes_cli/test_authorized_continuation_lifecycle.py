
from hermes_cli import kanban_db_connect, kanban_db_dispatch, kanban_db_notify
"""Real lifecycle regressions for review t_fd19cae6 (no live board/network)."""
import argparse
import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd

PR = "https://github.com/example/project/pull/42"


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    with kanban_db_connect.connect(db) as c:
        assert Path(c.execute("PRAGMA database_list").fetchone()[2]) == db
        yield c


def card(conn):
    tid = kb.create_task(conn, title="Continue same Draft PR", assignee="integrator")
    kb.add_comment(conn, tid, "integrator", f"Draft PR {PR}")
    return tid


def authorize(conn, tid):
    assert kb.promote_task(conn, tid, actor="operator", reason="Continue same PR") == (True, None)


def test_automatic_stale_recovery_cannot_renew_authorization(conn, monkeypatch):
    tid = card(conn)
    now = time.time()
    monkeypatch.setattr(kb.time, "time", lambda: now + 2)
    # Existing supported operator unblock, then actual claims and TTL recovery.
    assert kb.block_task(conn, tid, reason="operator pause")
    assert kb.unblock_task(conn, tid)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None
    for _ in range(3):
        assert kb.claim_task(conn, tid, claimer="remote-host:123")
        assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"
        with kanban_db_connect.write_txn(conn):
            conn.execute("UPDATE tasks SET claim_expires = 1 WHERE id = ?", (tid,))
        assert kb.release_stale_claims(conn) == 1
        assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"
        # Force the next claim only to adversarially repeat the recovery API;
        # the real dispatcher must never reach it without new authorization.


def test_manual_reclaim_authorizes_once(conn):
    tid = card(conn)
    assert kb.claim_task(conn, tid, claimer="remote-host:123")
    assert kb.reclaim_task(conn, tid, reason="operator requested retry")
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None
    assert kb.claim_task(conn, tid, claimer="remote-host:123")
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"


@pytest.mark.parametrize("comment_first", [True, False])
def test_same_second_order_is_unambiguous(conn, monkeypatch, comment_first):
    monkeypatch.setattr(kb.time, "time", lambda: 1800000000)
    tid = kb.create_task(conn, title="Ordering", assignee="integrator")
    assert kb.block_task(conn, tid, reason="pause")
    if comment_first:
        kb.add_comment(conn, tid, "integrator", PR)
    assert kb.unblock_task(conn, tid)
    if not comment_first:
        kb.add_comment(conn, tid, "integrator", PR)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == (None if comment_first else "active_pr")


def test_review_changes_authorize_same_card_exactly_once(conn):
    tid = card(conn)
    implementation = kb.claim_task(conn, tid, claimer="remote-host:123")
    assert kb.request_review(conn, tid, reviewer="pr-reviewer", summary="Candidate A", expected_run_id=implementation.current_run_id)
    review = kb.claim_review_task(conn, tid, claimer="remote-host:124")
    assert kb.request_changes(conn, tid, reason="Fix finding", expected_run_id=review.current_run_id) == (True, "integrator")
    assert kb.get_task(conn, tid).assignee == "integrator"
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None
    rework = kb.claim_task(conn, tid, claimer="remote-host:123")
    assert rework
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"
    assert kb.request_review(conn, tid, summary="Candidate B", expected_run_id=rework.current_run_id)
    assert kb.get_task(conn, tid).assignee == "pr-reviewer"
    assert kanban_db_dispatch.check_respawn_guard(conn, tid, lane="review") is None
    assert len(kb.list_tasks(conn)) == 1
    final_review = kb.claim_review_task(conn, tid, claimer="remote-host:124")
    assert final_review
    # Fixture gate evidence, not a claim about a real GitHub PR or hosted CI.
    evidence = {"head": "fixture-candidate-b", "review_head": "fixture-candidate-b",
                "review": "PASS", "ci_head": "fixture-candidate-b",
                "ci": "success", "draft": False}
    assert kb.complete_task(conn, tid, summary="Fixture exact-head gates passed",
                            metadata=evidence, expected_run_id=final_review.current_run_id)
    assert kb.latest_run(conn, tid).metadata == evidence
    assert kb.get_task(conn, tid).status == "done"


def test_automatic_promotion_is_not_new_authorization(conn):
    tid = card(conn)
    assert kb.claim_task(conn, tid, claimer="remote-host:123")
    dependency = kb.create_task(conn, title="Real prerequisite", assignee="planner")
    kb.link_tasks(conn, dependency, tid)
    assert kb.block_task(conn, tid, reason="waiting", kind="dependency")
    assert kb.complete_task(conn, dependency, summary="Prerequisite completed")
    kb.recompute_ready(conn)
    assert kb.get_task(conn, tid).status == "ready"
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"


def test_authorization_survives_real_parent_gate_but_is_not_replenished(conn):
    parent = kb.create_task(conn, title="Prerequisite", assignee="planner")
    tid = card(conn)
    assert kb.block_task(conn, tid, reason="operator pause")
    kb.link_tasks(conn, parent, tid)
    assert kb.unblock_task(conn, tid)
    assert kb.get_task(conn, tid).status == "todo"
    assert kb.claim_task(conn, tid, claimer="remote-host:123") is None
    assert kb.complete_task(conn, parent, summary="Prerequisite finished")
    kb.recompute_ready(conn)
    assert kb.get_task(conn, tid).status == "ready"
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None
    assert kb.claim_task(conn, tid, claimer="remote-host:123")
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"


def diagnostics(conn, tid):
    return kd.compute_task_diagnostics(kb.get_task(conn, tid), kb.list_events(conn, tid), kb.list_runs(conn, tid), config=dict(kd.DEFAULT_CONFIG))


def test_ready_diagnostic_action_really_authorizes_one_continuation(conn, capsys):
    tid = card(conn)
    kanban_db_dispatch._stamp_respawn_guard_event(conn, tid, "active_pr")
    diag = next(d for d in diagnostics(conn, tid) if d.kind == "respawn_guard_hold")
    command = diag.actions[0].payload["command"]
    assert command == f"hermes kanban promote {tid}"
    args = argparse.Namespace(task_id=tid, ids=[], reason=[], json=True, force=False, dry_run=False)
    assert cli._cmd_promote(args) == 0
    assert json.loads(capsys.readouterr().out)["promoted"] is True
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None
    assert kb.claim_task(conn, tid, claimer="remote-host:123")
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"


def test_old_diagnostic_is_explicitly_last_observed_not_live(conn, monkeypatch):
    tid = card(conn)
    kanban_db_dispatch._stamp_respawn_guard_event(conn, tid, "active_pr")
    future = time.time() + 90000
    monkeypatch.setattr(kb.time, "time", lambda: future)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None
    diag = next(d for d in diagnostics(conn, tid) if d.kind == "respawn_guard_hold")
    assert "last observed" in diag.title.lower()
    assert "may have cleared" in diag.detail.lower()


def test_promote_preserves_auth_failures_and_duplicate_claim_protection(conn):
    tid = card(conn)
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET consecutive_failures=1, last_failure_error='401 Unauthorized' WHERE id=?", (tid,))
    authorize(conn, tid)
    assert kb.get_task(conn, tid).consecutive_failures == 1
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "blocker_auth"
    assert kb.claim_task(conn, tid, claimer="remote-host:123")
    assert kb.claim_task(conn, tid, claimer="remote-host:124") is None
    assert not kb.promote_task(conn, tid, actor="operator")[0]


def test_protocol_exit_retries_remain_bounded_and_never_complete_draft(conn, monkeypatch):
    tid = card(conn)
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    host = kb._claimer_id().split(":", 1)[0]
    for attempt in range(kanban_db_dispatch._PROTOCOL_VIOLATION_FAILURE_LIMIT):
        authorize(conn, tid)
        assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None
        assert kb.claim_task(conn, tid, claimer=f"{host}:test")
        pid = 991000 + attempt
        kanban_db_dispatch._set_worker_pid(conn, tid, pid)
        kanban_db_dispatch._record_worker_exit(pid, 0)
        after_launch_grace = time.time() + kb._resolve_crash_grace_seconds() + 1
        monkeypatch.setattr(kb.time, "time", lambda: after_launch_grace)
        assert tid in kanban_db_dispatch.detect_crashed_workers(conn)
        assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"
    assert kb.get_task(conn, tid).status == "blocked"
    for _ in range(3):
        kb.recompute_ready(conn)
        result = kanban_db_dispatch.dispatch_once(conn, dry_run=True)
        assert not result.spawned
        assert kb.get_task(conn, tid).status == "blocked"
    assert not any(e.kind == "completed" for e in kb.list_events(conn, tid))
    assert len([e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]) == 1


def test_existing_review_child_is_released_and_reused_after_rework(conn, monkeypatch):
    from plugins.kanban.dashboard import plugin_api as api
    tid = card(conn)
    review_id = kb.create_task(conn, title="Sole independent review", assignee="pr-reviewer", parents=[tid])
    assert kb.get_task(conn, review_id).status == "todo"
    assert kb.claim_task(conn, review_id, claimer="remote-host:124") is None
    # The implementation phase completes to release its pre-created reviewer;
    # this is NOT a product-delivery PASS or merge declaration.
    first = kb.claim_task(conn, tid, claimer="remote-host:123")
    assert kb.complete_task(conn, tid, summary="Implementation phase ready for review", expected_run_id=first.current_run_id)
    kb.recompute_ready(conn)
    assert kb.get_task(conn, review_id).status == "ready"
    review = kb.claim_task(conn, review_id, claimer="remote-host:124")
    kb.add_comment(conn, tid, "pr-reviewer", "BLOCK candidate A: fix ordering")
    assert kb.complete_task(conn, review_id, summary="BLOCK candidate A; findings returned", expected_run_id=review.current_run_id)
    # Same supported dashboard transition used to reopen this real card.
    api.update_task(tid, api.UpdateTaskBody(status="ready"), board="default")
    assert kb.get_task(conn, review_id).status == "todo"
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None
    second = kb.claim_task(conn, tid, claimer="remote-host:123")
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) in {"active_pr", "recent_success"}
    assert kb.complete_task(conn, tid, summary="Candidate B ready; still awaiting exact-head review", expected_run_id=second.current_run_id)
    kb.recompute_ready(conn)
    assert kb.get_task(conn, review_id).status == "ready"
    assert kb.claim_task(conn, review_id, claimer="remote-host:124")
    assert len(kb.list_tasks(conn)) == 2
    assert not any(d.kind == "review_dependency_deadlock" for d in diagnostics(conn, tid))


def test_origin_subscription_and_event_cursor_survive_continuation(conn):
    tid = card(conn)
    origin = dict(task_id=tid, platform="discord", chat_id="fixture-channel", thread_id="fixture-thread")
    kanban_db_notify.add_notify_sub(conn, **origin)
    first = kb.claim_task(conn, tid, claimer="remote-host:123")
    assert kb.request_review(conn, tid, reviewer="pr-reviewer", summary="Candidate A", expected_run_id=first.current_run_id)
    _, _, events = kanban_db_notify.claim_unseen_events_for_sub(conn, **origin, kinds=("review_requested", "changes_requested"))
    assert [e.kind for e in events] == ["review_requested"]
    assert not kanban_db_notify.claim_unseen_events_for_sub(conn, **origin, kinds=("review_requested", "changes_requested"))[2]
    review = kb.claim_review_task(conn, tid, claimer="remote-host:124")
    assert kb.request_changes(conn, tid, reason="Fix finding", expected_run_id=review.current_run_id)[0]
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None
    assert kb.claim_task(conn, tid, claimer="remote-host:123")
    subs = kanban_db_notify.list_notify_subs(conn, tid)
    assert len(subs) == 1
    assert subs[0]["chat_id"] == origin["chat_id"]
    assert subs[0]["thread_id"] == origin["thread_id"]
    _, _, events = kanban_db_notify.claim_unseen_events_for_sub(conn, **origin, kinds=("review_requested", "changes_requested"))
    assert [e.kind for e in events] == ["changes_requested"]
    assert not kanban_db_notify.claim_unseen_events_for_sub(conn, **origin, kinds=("review_requested", "changes_requested"))[2]


def test_real_isolated_cli_hold_promote_and_one_continuation(conn, tmp_path):
    import subprocess
    import sys

    tid = kb.create_task(conn, title="CLI smoke", assignee="default")
    kb.add_comment(conn, tid, "default", PR)
    db = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    assert db.parent == tmp_path
    # An empty PATH intentionally makes gh unavailable: fail-closed PR liveness,
    # not a fabricated provider response. No agents or network calls are spawned.
    env = {
        "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "home"),
        "HERMES_KANBAN_HOME": str(tmp_path / "home"),
        "HERMES_KANBAN_DB": str(db), "PATH": str(tmp_path / "empty-bin"),
        "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
    }
    root = Path(__file__).resolve().parents[2]

    def run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "kanban", *args],
            cwd=root, env=env, capture_output=True, text=True, timeout=40,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        return result.stdout

    before = json.loads(run("dispatch", "--dry-run", "--json"))
    assert before["spawned"] == []
    assert {"task_id": tid, "reason": "active_pr"} in before["respawn_guarded"]
    assert "active_pr" in run("dispatch", "--dry-run")
    promoted = json.loads(run("promote", tid, "--json"))
    assert promoted["promoted"] is True
    after = json.loads(run("dispatch", "--dry-run", "--json"))
    assert [t["task_id"] for t in after["spawned"]] == [tid]
    assert kb.claim_task(conn, tid, claimer="remote-host:123")
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_expires = 1 WHERE id = ?", (tid,))
    repeated = json.loads(run("dispatch", "--dry-run", "--json"))
    assert repeated["spawned"] == []
    assert {"task_id": tid, "reason": "active_pr"} in repeated["respawn_guarded"]


@pytest.mark.parametrize("kind,payload", [
    ("reclaimed", {"manual": False}),
    ("reclaimed", None),
    ("promoted", {"status": "ready"}),
    ("status", {"status": "blocked"}),
    ("status", {"status": "todo", "reason": "ancestor_reopened"}),
    ("changes_requested", {"status": "ready"}),
    ("unblocked", ["unexpected non-object payload"]),
])
def test_ambiguous_or_automatic_events_do_not_authorize(conn, kind, payload):
    tid = card(conn)
    with kanban_db_connect.write_txn(conn):
        kb._append_event(conn, tid, kind, payload)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"


def test_legacy_comment_tie_fails_closed_but_later_authorization_works(conn, monkeypatch):
    tid = card(conn)
    with kanban_db_connect.write_txn(conn):
        conn.execute("UPDATE task_events SET payload = '{}' WHERE task_id=? AND kind='commented'", (tid,))
    comment_time = kb.list_comments(conn, tid)[0].created_at
    monkeypatch.setattr(kb.time, "time", lambda: comment_time)
    authorize(conn, tid)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) == "active_pr"
    monkeypatch.setattr(kb.time, "time", lambda: comment_time + 1)
    authorize(conn, tid)
    assert kanban_db_dispatch.check_respawn_guard(conn, tid) is None


@pytest.mark.parametrize("pending", ["Draft PR", "CI failed", "Review absent"])
def test_dispatch_does_not_synthesize_delivery_completion(conn, pending):
    tid = card(conn)
    kb.add_comment(conn, tid, "integrator", pending)
    for _ in range(3):
        result = kanban_db_dispatch.dispatch_once(conn, dry_run=True)
        assert not result.spawned
        assert kb.get_task(conn, tid).status == "ready"
    assert not any(e.kind == "completed" for e in kb.list_events(conn, tid))
