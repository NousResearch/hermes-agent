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


_DEFAULT_ROUTE = {
    "platform": "telegram",
    "chat_id": "chat-1",
    "thread_id": "thread-1",
    "user_id": "user-1",
    "notifier_profile": "default",
}


def _claimed_task(conn, *, title="approval", parents=(), route=None):
    task_id = kb.create_task(
        conn,
        title=title,
        assignee="worker",
        parents=parents,
        _trusted_gateway_origin=route or _DEFAULT_ROUTE,
    )
    task = kb.claim_task(conn, task_id, claimer="test:owner")
    assert task is not None
    return task


def _request(conn, task, *, raw_action="rm -rf /tmp/example"):
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


def test_manual_notification_subscription_cannot_grant_approval_authority(
    kanban_home,
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="manual route", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="attacker-chat",
            user_id="attacker",
            notifier_profile="default",
        )
        claimed = kb.claim_task(conn, task_id, claimer="test:owner")
        assert claimed is not None

        unavailable = _request(conn, claimed)

        assert unavailable is not None
        assert unavailable["status"] == "unavailable"
        assert unavailable["id"].startswith("kau_")
        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.get_run(conn, claimed.current_run_id).outcome == (
            "approval_unavailable"
        )
        assert kb.list_task_approvals(conn, task_id=task_id) == []
        assert kb.get_task_approval_route(conn, task_id) is None
        assert kb.recompute_ready(conn) == 0


def test_approval_origin_requires_exact_user_principal(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="anonymous", assignee="worker")
        with pytest.raises(ValueError, match="user_id"):
            kb._bind_task_approval_route(
                conn,
                task_id=task_id,
                platform="telegram",
                chat_id="chat-1",
                user_id="",
                notifier_profile="default",
            )
        assert kb.get_task_approval_route(conn, task_id) is None


def test_removed_origin_subscription_fails_before_request_is_persisted(
    kanban_home,
):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        assert kb.remove_notify_sub(
            conn,
            task_id=task.id,
            platform="telegram",
            chat_id="chat-1",
            thread_id="thread-1",
        ) is True

        unavailable = _request(conn, task)

        assert unavailable is not None
        assert unavailable["status"] == "unavailable"
        assert unavailable["approval_unavailable"] is True
        assert unavailable["id"].startswith("kau_")
        parked = kb.get_task(conn, task.id)
        assert parked.status == "blocked"
        assert parked.current_run_id is None
        assert parked.claim_lock is None
        run = kb.get_run(conn, task.current_run_id)
        assert run.status == "approval_pending"
        assert run.outcome == "approval_unavailable"
        assert kb.list_task_approvals(conn, task_id=task.id) == []
        assert kb.list_pending_unnotified_task_approvals(
            conn,
            task_id=task.id,
        ) == []
        assert kb.decide_task_approval(
            conn,
            unavailable["id"],
            "approve",
        ) is None
        assert kb.list_events(conn, task.id)[-1].kind == "approval_unavailable"
        assert kb.recompute_ready(conn) == 0


def test_approval_origin_is_immutable_and_idempotent(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="immutable origin",
            assignee="worker",
            _trusted_gateway_origin=_DEFAULT_ROUTE,
        )
        original = kb.get_task_approval_route(conn, task_id)
        assert original is not None
        assert kb._bind_task_approval_route(
            conn,
            task_id=task_id,
            **_DEFAULT_ROUTE,
        ) is True
        assert kb._bind_task_approval_route(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="attacker-chat",
            thread_id="thread-1",
            user_id="attacker",
            notifier_profile="default",
        ) is False
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            thread_id="thread-1",
            user_id="attacker",
            notifier_profile="attacker-profile",
        )
        origin_sub = next(
            sub
            for sub in kb.list_notify_subs(conn, task_id)
            if sub["chat_id"] == "chat-1"
        )
        assert origin_sub["user_id"] == "user-1"
        assert origin_sub["notifier_profile"] == "default"
        assert kb.get_task_approval_route(conn, task_id) == original


def test_idempotent_create_cannot_claim_unbound_or_foreign_task(kanban_home):
    with kb.connect() as conn:
        unbound_id = kb.create_task(
            conn,
            title="unbound",
            assignee="worker",
            idempotency_key="unbound-key",
        )
        with pytest.raises(ValueError, match="not owned"):
            kb.create_task(
                conn,
                title="retry",
                assignee="worker",
                idempotency_key="unbound-key",
                _trusted_gateway_origin=_DEFAULT_ROUTE,
            )
        assert kb.get_task_approval_route(conn, unbound_id) is None

        bound_id = kb.create_task(
            conn,
            title="bound",
            assignee="worker",
            idempotency_key="bound-key",
            _trusted_gateway_origin=_DEFAULT_ROUTE,
        )
        foreign = {**_DEFAULT_ROUTE, "chat_id": "foreign-chat"}
        with pytest.raises(ValueError, match="not owned"):
            kb.create_task(
                conn,
                title="foreign retry",
                assignee="worker",
                idempotency_key="bound-key",
                _trusted_gateway_origin=foreign,
            )
        assert kb.get_task_approval_route(conn, bound_id)["chat_id"] == "chat-1"


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


def test_approval_notification_state_is_durable_and_idempotent(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        request = _request(conn, task)
        assert request is not None
        assert request["notified_at"] is None
        assert [
            item["id"]
            for item in kb.list_pending_unnotified_task_approvals(
                conn,
                task_id=task.id,
            )
        ] == [request["id"]]

        assert kb.mark_task_approval_notified(
            conn,
            request["id"],
            notified_at=1234,
        ) is True
        # A duplicate post-send acknowledgement is harmless.
        assert kb.mark_task_approval_notified(
            conn,
            request["id"],
            notified_at=5678,
        ) is True

        stored = kb.get_task_approval(conn, request["id"])
        assert stored is not None
        assert stored["notified_at"] == 1234
        assert kb.list_pending_unnotified_task_approvals(
            conn,
            task_id=task.id,
        ) == []


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
        task = _claimed_task(
            conn,
            route={
                "platform": "telegram",
                "chat_id": "shared-chat",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "notifier_profile": "default",
            },
        )
        request = _request(conn, task)
        assert request is not None
        conn.execute(
            "UPDATE tasks SET status = 'ready', block_kind = NULL WHERE id = ?",
            (task.id,),
        )

        assert kb.claim_task(conn, task.id, claimer="manual:claimer") is None
        assert kb.get_task(conn, task.id).status == "blocked"


def test_route_bound_decision_requeues_binds_and_consumes_once(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(
            conn,
            route={
                "platform": "telegram",
                "chat_id": "chat-1",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "notifier_profile": "default",
            },
        )
        request = _request(conn, task)
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


def test_review_approval_restores_review_lane_and_binds_exact_grant(
    kanban_home,
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review approval",
            assignee="worker",
            _trusted_gateway_origin=_DEFAULT_ROUTE,
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        review_run = kb.claim_review_task(
            conn,
            task_id,
            claimer="test:review-owner",
        )
        assert review_run is not None
        request = _request(conn, review_run)
        assert request is not None and request["status"] == "pending"

        approved = kb.decide_task_approval(
            conn,
            request["id"],
            "approve",
            **_DEFAULT_ROUTE,
        )

        assert approved is not None and approved["status"] == "approved"
        assert kb.get_task(conn, task_id).status == "review"
        assert kb.claim_task(conn, task_id, claimer="wrong:ready-lane") is None

        resumed = kb.claim_review_task(
            conn,
            task_id,
            claimer="test:review-resume",
        )
        assert resumed is not None
        assert resumed.approval_request_id == request["id"]
        assert resumed.approval_grant_nonce == approved["grant_nonce"]
        assert resumed.approval_worker_session_id == "20260712_120000_abcdef"
        stored = kb.get_task_approval(conn, request["id"])
        assert stored["resume_run_id"] == resumed.current_run_id
        assert kb.consume_task_approval(
            conn,
            request_id=request["id"],
            grant_nonce=approved["grant_nonce"],
            task_id=task_id,
            resume_run_id=resumed.current_run_id,
            profile="worker",
            action_digest=request["action_digest"],
        ) is True


def test_expired_unbound_review_grant_blocks_review_dispatch(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="expired review approval",
            assignee="worker",
            _trusted_gateway_origin=_DEFAULT_ROUTE,
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        review_run = kb.claim_review_task(conn, task_id, claimer="test:review")
        request = _request(conn, review_run)
        approved = kb.decide_task_approval(conn, request["id"], "approve")
        assert approved is not None
        assert kb.get_task(conn, task_id).status == "review"
        conn.execute(
            "UPDATE kanban_approval_requests SET expires_at = 1 WHERE id = ?",
            (request["id"],),
        )

        assert kb.expire_task_approvals(conn, now=2) == 1

        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.claim_review_task(conn, task_id, claimer="must:not-spawn") is None
        assert kb.get_task_approval(conn, request["id"])["status"] == "expired"


@pytest.mark.parametrize("failure_stage", ["workspace", "spawn"])
def test_review_resume_failure_requeues_review_and_preserves_unbound_grant(
    kanban_home,
    monkeypatch,
    failure_stage,
):
    profile = kanban_home / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title=f"review {failure_stage} failure",
            assignee="worker",
            workspace_kind=("worktree" if failure_stage == "workspace" else "scratch"),
            _trusted_gateway_origin=_DEFAULT_ROUTE,
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        initial = kb.claim_review_task(conn, task_id, claimer="test:review")
        assert initial is not None
        request = _request(conn, initial)
        assert request is not None
        approved = kb.decide_task_approval(conn, request["id"], "approve")
        assert approved is not None
        original_nonce = approved["grant_nonce"]

        spawn_calls = []

        if failure_stage == "workspace":
            def fail_workspace(*_args, **_kwargs):
                raise OSError("synthetic workspace failure")

            monkeypatch.setattr(kb, "_resolve_worktree_workspace", fail_workspace)

            def spawn_fn(*_args, **_kwargs):
                spawn_calls.append(True)
                raise AssertionError("spawn reached after workspace failure")
        else:
            def spawn_fn(*_args, **_kwargs):
                spawn_calls.append(True)
                raise OSError("synthetic Popen failure")

        kb.dispatch_once(conn, spawn_fn=spawn_fn, failure_limit=5)

        after = kb.get_task(conn, task_id)
        grant = kb.get_task_approval(conn, request["id"])
        assert after.status == "review"
        assert after.current_run_id is None
        assert grant["status"] == "approved"
        assert grant["resume_run_id"] is None
        assert grant["grant_nonce"] != original_nonce
        assert kb.claim_task(conn, task_id, claimer="wrong:ready-lane") is None
        resumed = kb.claim_review_task(conn, task_id, claimer="test:retry-review")
        assert resumed is not None
        assert resumed.approval_request_id == request["id"]
        assert resumed.approval_grant_nonce == grant["grant_nonce"]
        assert bool(spawn_calls) is (failure_stage == "spawn")


def test_stale_claim_reaper_returns_review_run_to_review_lane(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="stale review worker",
            assignee="worker",
            _trusted_gateway_origin=_DEFAULT_ROUTE,
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        claimed = kb.claim_review_task(conn, task_id, claimer="remote:review")
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET claim_expires = 1 WHERE id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE task_runs SET claim_expires = 1 WHERE id = ?",
            (claimed.current_run_id,),
        )

        assert kb.release_stale_claims(conn) == 1

        requeued = kb.get_task(conn, task_id)
        ended = kb.get_run(conn, claimed.current_run_id)
        assert requeued.status == "review"
        assert requeued.current_run_id is None
        assert ended.outcome == "reclaimed"
        assert kb.claim_task(conn, task_id, claimer="wrong:ready-lane") is None
        assert kb.claim_review_task(conn, task_id, claimer="test:review-retry")


def test_review_crash_at_failure_threshold_blocks_instead_of_respawning(
    kanban_home,
    monkeypatch,
):
    host = kb._claimer_id().split(":", 1)[0]
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="crashed review worker",
            assignee="worker",
            _trusted_gateway_origin=_DEFAULT_ROUTE,
        )
        conn.execute(
            "UPDATE tasks SET status = 'review', max_retries = 1 WHERE id = ?",
            (task_id,),
        )
        claimed = kb.claim_review_task(
            conn,
            task_id,
            claimer=f"{host}:review",
        )
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET worker_pid = 424242, started_at = 1 WHERE id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = 424242, started_at = 1 WHERE id = ?",
            (claimed.current_run_id,),
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(
            kb,
            "_classify_worker_exit",
            lambda _pid: ("nonzero_exit", 1),
        )

        assert kb.detect_crashed_workers(conn) == [task_id]

        blocked = kb.get_task(conn, task_id)
        assert blocked.status == "blocked"
        assert blocked.current_run_id is None
        assert blocked.consecutive_failures == 1
        assert task_id in kb.detect_crashed_workers._last_auto_blocked
        assert kb.claim_review_task(conn, task_id, claimer="must:not-respawn") is None
        assert kb.list_events(conn, task_id)[-1].kind == "gave_up"


@pytest.mark.parametrize("failure_stage", ["workspace", "spawn"])
@pytest.mark.parametrize("failure_limit", [1, 5])
def test_stale_dispatch_failure_cannot_mutate_successor_run(
    kanban_home,
    monkeypatch,
    failure_stage,
    failure_limit,
):
    profile = kanban_home / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title=f"stale {failure_stage} failure",
            assignee="worker",
        )
        successor = {}

        def replace_claimed_run():
            active = kb.get_task(conn, task_id)
            assert active.status == "running"
            assert kb.reclaim_task(conn, task_id, reason="synthetic race")
            claimed_b = kb.claim_task(conn, task_id, claimer="successor:B")
            assert claimed_b is not None
            successor["task"] = claimed_b

        if failure_stage == "workspace":
            def fail_workspace(_task, *, board=None):
                del board
                replace_claimed_run()
                raise OSError("stale workspace failure from run A")

            monkeypatch.setattr(kb, "resolve_workspace", fail_workspace)

            def spawn_fn(*_args, **_kwargs):
                raise AssertionError("spawn reached after workspace failure")
        else:
            def spawn_fn(*_args, **_kwargs):
                replace_claimed_run()
                raise OSError("stale Popen failure from run A")

        kb.dispatch_once(conn, spawn_fn=spawn_fn, failure_limit=failure_limit)

        claimed_b = successor["task"]
        after = kb.get_task(conn, task_id)
        run_b = kb.get_run(conn, claimed_b.current_run_id)
        assert after.status == "running"
        assert after.current_run_id == claimed_b.current_run_id
        assert after.claim_lock == "successor:B"
        assert after.consecutive_failures == 0
        assert run_b.status == "running"
        assert run_b.ended_at is None
        assert not any(
            event.kind == "gave_up"
            and event.run_id == claimed_b.current_run_id
            for event in kb.list_events(conn, task_id)
        )


@pytest.mark.parametrize("max_retries", [1, 5])
def test_timeout_failure_accounting_cannot_mutate_successor_run(
    kanban_home,
    monkeypatch,
    max_retries,
):
    host = kb._claimer_id().split(":", 1)[0]
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="timeout accounting race",
            assignee="worker",
            max_retries=max_retries,
            max_runtime_seconds=1,
        )
        claimed_a = kb.claim_task(conn, task_id, claimer=f"{host}:run-a")
        assert claimed_a is not None
        assert kb._set_worker_pid(
            conn,
            task_id,
            424242,
            expected_run_id=claimed_a.current_run_id,
            expected_claim_lock=claimed_a.claim_lock,
        )
        conn.execute(
            "UPDATE tasks SET started_at = 1 WHERE id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = 1 WHERE id = ?",
            (claimed_a.current_run_id,),
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        original_record_failure = kb._record_task_failure
        successor = {}

        def claim_successor_before_accounting(*args, **kwargs):
            claimed_b = kb.claim_task(conn, task_id, claimer="successor:B")
            assert claimed_b is not None
            successor["task"] = claimed_b
            return original_record_failure(*args, **kwargs)

        monkeypatch.setattr(
            kb,
            "_record_task_failure",
            claim_successor_before_accounting,
        )

        assert kb.enforce_max_runtime(
            conn,
            signal_fn=lambda _pid, _signal: None,
        ) == [task_id]

        claimed_b = successor["task"]
        after = kb.get_task(conn, task_id)
        run_a = kb.get_run(conn, claimed_a.current_run_id)
        run_b = kb.get_run(conn, claimed_b.current_run_id)
        assert run_a.outcome == "timed_out"
        assert after.status == "running"
        assert after.current_run_id == claimed_b.current_run_id
        assert after.claim_lock == "successor:B"
        assert after.consecutive_failures == 0
        assert run_b.status == "running"
        assert run_b.ended_at is None
        assert not any(
            event.kind == "gave_up" and event.run_id == claimed_a.current_run_id
            for event in kb.list_events(conn, task_id)
        )


def test_rate_limited_review_retry_obeys_respawn_cooldown(
    kanban_home,
    monkeypatch,
):
    profile = kanban_home / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    host = kb._claimer_id().split(":", 1)[0]
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="rate-limited review",
            assignee="worker",
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        claimed = kb.claim_review_task(
            conn,
            task_id,
            claimer=f"{host}:review",
        )
        assert claimed is not None
        assert kb._set_worker_pid(
            conn,
            task_id,
            424242,
            expected_run_id=claimed.current_run_id,
            expected_claim_lock=claimed.claim_lock,
        )
        conn.execute(
            "UPDATE tasks SET started_at = 1 WHERE id = ?",
            (task_id,),
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(
            kb,
            "_classify_worker_exit",
            lambda _pid: ("rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE),
        )
        monkeypatch.setattr(
            kb,
            "_resolve_rate_limit_cooldown_seconds",
            lambda: 3600,
        )

        assert kb.detect_crashed_workers(conn) == []
        assert kb.get_task(conn, task_id).status == "review"
        spawned = []
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: spawned.append(True),
        )

        assert spawned == []
        assert (task_id, "rate_limit_cooldown") in result.respawn_guarded
        assert kb.get_task(conn, task_id).status == "review"
        assert kb.list_events(conn, task_id)[-1].kind == "respawn_guarded"


def test_legacy_claim_event_without_source_status_resumes_ready(kanban_home):
    with kb.connect() as conn:
        task = _claimed_task(conn)
        claimed_event = conn.execute(
            "SELECT id, payload FROM task_events "
            "WHERE task_id = ? AND run_id = ? AND kind = 'claimed'",
            (task.id, task.current_run_id),
        ).fetchone()
        payload = json.loads(claimed_event["payload"])
        payload.pop("source_status", None)
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE id = ?",
            (json.dumps(payload), claimed_event["id"]),
        )

        request = _request(conn, task)
        approved = kb.decide_task_approval(conn, request["id"], "approve")

        assert approved is not None
        assert kb.get_task(conn, task.id).status == "ready"


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
        task = _claimed_task(
            conn,
            route={
                "platform": "telegram",
                "chat_id": "shared-chat",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "notifier_profile": "default",
            },
        )
        request = _request(conn, task)
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
        owner_bootstrap_nonce="owner-bootstrap",
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
    assert env["HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE"] == "owner-bootstrap"
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
