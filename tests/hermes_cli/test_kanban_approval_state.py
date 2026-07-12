"""Durable approval-state tests for detached Kanban workers."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import threading
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


def _claimed_task(conn, *, title="approval", parents=()):
    task_id = kb.create_task(
        conn,
        title=title,
        assignee="worker",
        parents=parents,
    )
    task = kb.claim_task(conn, task_id, claimer="test:owner")
    assert task is not None
    return task


def _request(conn, task, *, raw_action="rm -rf /tmp/example", route=None):
    return kb.request_task_approval(
        conn,
        task_id=task.id,
        action_kind="terminal",
        action_digest=kb.kanban_action_digest(
            "terminal",
            raw_action,
            "local",
            workdir="/tmp/workspace",
        ),
        display_target="rm -rf /tmp/[redacted]",
        description="Delete generated files",
        worker_session_id="20260712_120000_abcdef",
        expected_run_id=task.current_run_id,
        expected_claim_lock=task.claim_lock,
        profile="worker",
        route=route,
        timeout_seconds=300,
    )


def test_action_digest_binds_execution_context(tmp_path):
    base = kb.kanban_action_digest(
        "terminal", "dangerous command", "docker", workdir=str(tmp_path),
    )
    assert base == kb.kanban_action_digest(
        "terminal", "dangerous command", "DOCKER", workdir=str(tmp_path),
    )
    assert base != kb.kanban_action_digest(
        "terminal", "changed command", "docker", workdir=str(tmp_path),
    )
    assert base != kb.kanban_action_digest(
        "terminal",
        "dangerous command",
        "docker",
        has_host_access=True,
        workdir=str(tmp_path),
    )
    assert base != kb.kanban_action_digest(
        "terminal", "dangerous command", "local", workdir=str(tmp_path),
    )


def test_request_parks_exact_run_without_counting_failure_and_redacts_event(
    kanban_home,
):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        kb.add_notify_sub(
            conn,
            task_id=task.id,
            platform="tui",
            chat_id="local-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=task.id,
            platform="telegram",
            chat_id="chat-1",
            thread_id="thread-1",
            user_id="user-1",
            notifier_profile="default",
        )

        request = _request(conn, task)

        assert request is not None
        assert request["status"] == "pending"
        assert request["platform"] == "telegram"
        assert request["chat_id"] == "chat-1"
        parked = kb.get_task(conn, task.id)
        assert parked.status == "blocked"
        assert parked.current_run_id is None
        assert parked.claim_lock is None
        assert parked.worker_pid is None
        assert parked.consecutive_failures == 0
        run = kb.get_run(conn, task.current_run_id)
        assert run.status == "approval_pending"
        assert run.outcome == "approval_pending"

        event = kb.list_events(conn, task.id)[-1]
        assert event.kind == "approval_requested"
        assert event.payload["request_id"] == request["id"]
        serialized = json.dumps(event.payload)
        assert "rm -rf /tmp/example" not in serialized
        assert request["action_digest"] not in serialized


def test_late_pid_registration_cannot_reanimate_parked_worker(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        request = _request(conn, task)
        assert request is not None

        assert kb._set_worker_pid(conn, task.id, 424242) is False

        parked = kb.get_task(conn, task.id)
        assert parked.status == "blocked"
        assert parked.current_run_id is None
        assert parked.worker_pid is None
        assert "spawned" not in [event.kind for event in kb.list_events(conn, task.id)]


def test_late_pid_registration_cannot_attach_to_successor_run(kanban_home):
    with kb.connect() as conn:
        first = _claimed_task(conn)
        first_run_id = first.current_run_id
        first_claim_lock = first.claim_lock
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, current_run_id = NULL WHERE id = ?",
                (first.id,),
            )
            conn.execute(
                "UPDATE task_runs SET status = 'released', outcome = 'released', "
                "ended_at = 1 WHERE id = ?",
                (first_run_id,),
            )
        successor = kb.claim_task(conn, first.id, claimer="test:successor")
        assert successor is not None
        assert successor.current_run_id != first_run_id

        assert kb._set_worker_pid(
            conn,
            first.id,
            424242,
            expected_run_id=first_run_id,
            expected_claim_lock=first_claim_lock,
        ) is False
        assert kb.get_task(conn, first.id).worker_pid is None
        assert kb.get_run(conn, successor.current_run_id).worker_pid is None


def test_request_is_owner_cas_and_idempotent(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        stale = kb.request_task_approval(
            conn,
            task_id=task.id,
            action_kind="terminal",
            action_digest="sha256:stale",
            display_target="dangerous command",
            worker_session_id="session",
            expected_run_id=task.current_run_id + 1,
            expected_claim_lock=task.claim_lock,
            profile="worker",
        )
        assert stale is None
        assert kb.get_task(conn, task.id).status == "running"

        first = _request(conn, task)
        second = _request(conn, task)
        assert first is not None
        assert second is not None
        assert first["id"] == second["id"]


def test_pending_request_rejects_manual_ready_claim(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        request = _request(
            conn,
            task,
            route={
                "platform": "telegram",
                "chat_id": "shared-chat",
                "thread_id": "thread-1",
                "notifier_profile": "default",
            },
        )
        assert request is not None
        conn.execute(
            "UPDATE tasks SET status = 'ready', block_kind = NULL WHERE id = ?",
            (task.id,),
        )

        assert kb.claim_task(conn, task.id, claimer="manual:claimer") is None
        assert kb.get_task(conn, task.id).status == "blocked"


def test_route_bound_decision_requeues_binds_and_consumes_once(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        request = _request(
            conn,
            task,
            route={
                "platform": "telegram",
                "chat_id": "chat-1",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "notifier_profile": "default",
            },
        )
        assert request is not None
        wrong_route = kb.decide_task_approval(
            conn,
            request["id"],
            "approve",
            platform="telegram",
            chat_id="other-chat",
        )
        assert wrong_route is None
        assert kb.get_task_approval(conn, request["id"])["status"] == "pending"

        approved = kb.decide_task_approval(
            conn,
            request["id"],
            "approve",
            platform="telegram",
            chat_id="chat-1",
            thread_id="thread-1",
            user_id="user-1",
            notifier_profile="default",
            decided_by="user-1",
        )
        assert approved is not None
        assert approved["status"] == "approved"
        assert approved["grant_nonce"]
        assert kb.get_task(conn, task.id).status == "ready"

        resumed = kb.claim_task(conn, task.id, claimer="test:resume")
        assert resumed is not None
        assert resumed.approval_request_id == request["id"]
        assert resumed.approval_grant_nonce == approved["grant_nonce"]
        assert resumed.approval_worker_session_id == "20260712_120000_abcdef"
        assert kb.claim_task(conn, task.id, claimer="test:loser") is None
        assert kb.get_task_approval(conn, request["id"])["status"] == "approved"

        consume_args = {
            "request_id": request["id"],
            "grant_nonce": approved["grant_nonce"],
            "task_id": task.id,
            "resume_run_id": resumed.current_run_id,
            "profile": "worker",
            "action_digest": request["action_digest"],
        }
        assert kb.consume_task_approval(conn, **consume_args) is True
        assert kb.consume_task_approval(conn, **consume_args) is False


def test_approved_task_is_parent_gated(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        kb.complete_task(conn, parent, result="temporarily done")
        task = _claimed_task(conn, parents=[parent])
        request = _request(conn, task)
        assert request is not None
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parent,))

        approved = kb.decide_task_approval(conn, request["id"], "approve")

        assert approved is not None
        assert kb.get_task(conn, task.id).status == "todo"


@pytest.mark.parametrize("terminal_state", ["denied", "expired"])
def test_denied_or_expired_request_stays_blocked_across_recompute(
    kanban_home,
    terminal_state,
):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        request = _request(conn, task)
        if terminal_state == "denied":
            decided = kb.decide_task_approval(conn, request["id"], "deny")
            assert decided["status"] == "denied"
        else:
            conn.execute(
                "UPDATE kanban_approval_requests SET expires_at = 1 WHERE id = ?",
                (request["id"],),
            )
            result = kb.dispatch_once(conn, spawn_fn=lambda *_args, **_kwargs: None)
            assert result.expired_approvals == 1

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, task.id).status == "blocked"


def test_exact_grant_has_one_winner_across_connections(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        request = _request(conn, task)
        approved = kb.decide_task_approval(conn, request["id"], "approve")
        resumed = kb.claim_task(conn, task.id, claimer="test:resume")
        args = {
            "request_id": request["id"],
            "grant_nonce": approved["grant_nonce"],
            "task_id": task.id,
            "resume_run_id": resumed.current_run_id,
            "profile": "worker",
            "action_digest": request["action_digest"],
        }

    def consume():
        with kb.connect() as thread_conn:
            return kb.consume_task_approval(thread_conn, **args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: consume(), range(2)))
    assert sorted(results) == [False, True]


def test_competing_gateway_decisions_have_one_cas_winner(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        request = _request(
            conn,
            task,
            route={
                "platform": "telegram",
                "chat_id": "shared-chat",
                "thread_id": "thread-1",
                "notifier_profile": "default",
            },
        )
        request_id = request["id"]

    barrier = threading.Barrier(2)

    def decide(decision):
        with kb.connect() as thread_conn:
            barrier.wait(timeout=5)
            return kb.decide_task_approval(
                thread_conn,
                request_id,
                decision,
                platform=request["platform"],
                chat_id=request["chat_id"],
                thread_id=request["thread_id"],
                notifier_profile=request["notifier_profile"],
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(decide, ("approve", "deny")))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0]["status"] in {"approved", "denied"}
    with kb.connect() as conn:
        final = kb.get_task_approval(conn, request_id)
        assert final["status"] == winners[0]["status"]
        expected_task_status = "ready" if final["status"] == "approved" else "blocked"
        assert kb.get_task(conn, task.id).status == expected_task_status


def test_changed_resumed_action_supersedes_bound_grant_and_parks_again(
    kanban_home,
):
    with kb.connect() as conn:
        first_run = _claimed_task(conn)
        first = _request(conn, first_run)
        approved = kb.decide_task_approval(conn, first["id"], "approve")
        resumed = kb.claim_task(conn, first_run.id, claimer="test:resume")
        assert resumed.approval_grant_nonce == approved["grant_nonce"]

        replacement = _request(
            conn,
            resumed,
            raw_action="rm -rf /tmp/different-target",
        )

        assert replacement is not None
        assert replacement["id"] != first["id"]
        assert replacement["status"] == "pending"
        assert kb.get_task_approval(conn, first["id"])["status"] == "cancelled"
        parked = kb.get_task(conn, first_run.id)
        assert parked.status == "blocked"
        assert parked.current_run_id is None
        assert parked.consecutive_failures == 0


def test_post_spawn_run_end_cancels_unconsumed_bound_grant(kanban_home):
    with kb.connect() as conn:
        first_run = _claimed_task(conn)
        request = _request(conn, first_run)
        kb.decide_task_approval(conn, request["id"], "approve")
        resumed = kb.claim_task(conn, first_run.id, claimer="test:resume")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
                (first_run.id,),
            )
            kb._end_run(
                conn,
                first_run.id,
                outcome="crashed",
                status="crashed",
                error="synthetic child crash",
            )

        after = kb.get_task_approval(conn, request["id"])
        assert after["status"] == "cancelled"
        assert after["grant_nonce"] is None
        assert after["resume_run_id"] == resumed.current_run_id


def test_dispatch_spawn_failure_unbinds_and_rotates_grant(kanban_home):
    profile = kanban_home / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    with kb.connect() as conn:
        task = _claimed_task(conn)
        request = _request(conn, task)
        approved = kb.decide_task_approval(conn, request["id"], "approve")

        def fail_spawn(*_args, **_kwargs):
            raise OSError("synthetic spawn failure")

        kb.dispatch_once(conn, spawn_fn=fail_spawn, failure_limit=5)

        after = kb.get_task_approval(conn, request["id"])
        assert after["status"] == "approved"
        assert after["resume_run_id"] is None
        assert after["grant_nonce"] != approved["grant_nonce"]
        assert kb.get_task(conn, task.id).consecutive_failures == 1


def test_approved_resume_bypasses_generic_respawn_heuristics(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        request = _request(conn, task)
        approved = kb.decide_task_approval(conn, request["id"], "approve")
        assert approved["status"] == "approved"
        conn.execute(
            "UPDATE tasks SET last_failure_error = '403 authentication denied' "
            "WHERE id = ?",
            (task.id,),
        )
        kb.add_comment(
            conn,
            task.id,
            "worker",
            "Review https://github.com/example/project/pull/123",
        )

        assert kb.check_respawn_guard(conn, task.id) is None


def test_default_spawn_scrubs_parent_authority_and_resumes_session(
    kanban_home,
    monkeypatch,
    tmp_path,
):
    profile = kanban_home / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    for key in (
        "HERMES_EXEC_ASK",
        "HERMES_CRON_SESSION",
        "HERMES_INTERACTIVE",
        "HERMES_YOLO_MODE",
        "HERMES_GATEWAY_SESSION",
        "_HERMES_GATEWAY",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_TUI_QUERY",
    ):
        monkeypatch.setenv(key, "inherited")
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = kb.Task(
        id="t_resume",
        title="resume",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=1,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=str(workspace),
        claim_lock="test:resume",
        claim_expires=None,
        tenant=None,
        current_run_id=8,
        approval_request_id="ka_request",
        approval_grant_nonce="opaque-nonce",
        approval_worker_session_id="20260712_120000_abcdef",
    )

    kb._default_spawn(task, str(workspace))

    env = captured["env"]
    for key in (
        "HERMES_EXEC_ASK",
        "HERMES_CRON_SESSION",
        "HERMES_INTERACTIVE",
        "HERMES_YOLO_MODE",
        "HERMES_GATEWAY_SESSION",
        "_HERMES_GATEWAY",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_TUI_QUERY",
    ):
        assert key not in env
    assert env["HERMES_KANBAN_SESSION"] == "1"
    assert env["HERMES_KANBAN_APPROVAL_ID"] == "ka_request"
    assert env["HERMES_KANBAN_APPROVAL_NONCE"] == "opaque-nonce"
    cmd = captured["cmd"]
    assert cmd.index("--resume") < cmd.index("chat")
    assert cmd[cmd.index("--resume") + 1] == "20260712_120000_abcdef"
    assert "resume kanban task t_resume after its human approval" in cmd[-1]

    # A fresh run must not inherit a stale grant exported by its launcher.
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_ID", "stale-request")
    monkeypatch.setenv("HERMES_KANBAN_APPROVAL_NONCE", "stale-nonce")
    task.approval_request_id = None
    task.approval_grant_nonce = None
    task.approval_worker_session_id = None
    kb._default_spawn(task, str(workspace))
    assert "HERMES_KANBAN_APPROVAL_ID" not in captured["env"]
    assert "HERMES_KANBAN_APPROVAL_NONCE" not in captured["env"]
    assert "--resume" not in captured["cmd"]
