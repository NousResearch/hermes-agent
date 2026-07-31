"""Native dynamic Kanban workflow aggregation contracts."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import doctor
from hermes_cli import kanban_db as kb


@pytest.fixture
def workflow_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "workflow.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    conn = kb.connect(path)
    try:
        yield conn
    finally:
        conn.close()
        kb._INITIALIZED_PATHS.discard(str(path.resolve()))


def _actor(tenant="tenant-a", capabilities=("workflow.manage",)):
    return kb.KanbanActorContext(
        principal_id="svc:orchestrator",
        profile_name="orchestrator",
        board_identity="board:test",
        tenant=tenant,
        capabilities=frozenset(capabilities),
        source_kind="orchestrator",
    )


def _worker_actor(*, task_id, run_id, claim_lock, tenant="tenant-a"):
    return kb.KanbanActorContext(
        principal_id="worker:x_qa",
        profile_name="x_qa",
        source_kind="dispatcher_worker",
        board_identity="board:test",
        tenant=tenant,
        capabilities=frozenset(("workflow.outcome",)),
        task_scope=task_id,
        run_id=run_id,
        claim_lock=claim_lock,
    )


def _race_outcome_worker(db_path, task_id, outcome, mutation_id, start, results):
    conn = kb.connect(Path(db_path))
    try:
        start.wait(10)
        response = kb.record_workflow_outcome(
            conn, workflow_id="wf_race", task_id=task_id, outcome=outcome,
            actor=_actor(capabilities=("workflow.outcome",)),
            mutation_id=mutation_id, expected_version=1,
        )
        results.put(("ok", outcome, response["workflow"]["state"]))
    except Exception as exc:
        results.put(("error", outcome, type(exc).__name__, str(exc)))
    finally:
        conn.close()


def _new_workflow(workflow_db, workflow_id="wf_test"):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    kb.create_workflow(
        workflow_db,
        workflow_id=workflow_id,
        name="release",
        tenant="tenant-a",
        designated_acceptance_task_id=acceptance,
        actor=_actor(),
        mutation_id=f"create-{workflow_id}",
    )
    return acceptance


def test_workflow_subscription_claim_retry_dead_letter_and_resume(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    kb.create_workflow(
        workflow_db, workflow_id="wf_notify", name="notify", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=_actor(), mutation_id="create-notify",
        subscription={
            "platform": "telegram", "chat_id": "origin", "notifier_profile": "default",
            "target_states": ["PASS"],
        },
    )
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_notify", task_id=acceptance, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="notify-pass", expected_version=1,
    )
    old, claimed, events = kb.claim_workflow_events_for_subscription(
        workflow_db, workflow_id="wf_notify", role="origin"
    )
    assert old < claimed
    assert [event["kind"] for event in events] == ["aggregate_changed"]

    retry = kb.fail_workflow_delivery(
        workflow_db, workflow_id="wf_notify", role="origin",
        claimed_cursor=claimed, old_cursor=old, error_class="TemporaryError",
        max_retries=2, retry_delay_seconds=0,
    )
    assert retry["dead_lettered_at"] is None
    _, claimed_again, _ = kb.claim_workflow_events_for_subscription(
        workflow_db, workflow_id="wf_notify", role="origin"
    )
    dead = kb.fail_workflow_delivery(
        workflow_db, workflow_id="wf_notify", role="origin",
        claimed_cursor=claimed_again, old_cursor=old, error_class="PermanentError",
        max_retries=2, retry_delay_seconds=0,
    )
    assert dead["dead_lettered_at"] is not None
    assert kb.count_workflow_subscriptions(workflow_db) == 1

    kb.resume_workflow_subscription(
        workflow_db, workflow_id="wf_notify", role="origin",
        actor=_actor(capabilities=("workflow.admin",)),
    )
    _, _, resumed_events = kb.claim_workflow_events_for_subscription(
        workflow_db, workflow_id="wf_notify", role="origin"
    )
    assert resumed_events
    completed = kb.complete_workflow_delivery(
        workflow_db, workflow_id="wf_notify", role="origin"
    )
    assert completed["retry_count"] == 0
    assert completed["next_attempt_at"] is None
    assert completed["last_error_class"] is None


def test_workflow_subscription_claim_stops_at_first_target_transition(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    kb.create_workflow(
        workflow_db, workflow_id="wf_transition", name="notify", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=_actor(),
        mutation_id="create-transition",
        subscription={
            "platform": "telegram", "chat_id": "origin", "notifier_profile": "default",
            "target_states": ["PASS"],
        },
    )
    passed = kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_transition", task_id=acceptance, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="transition-pass", expected_version=1,
    )
    pass_event = workflow_db.execute(
        "SELECT * FROM kanban_workflow_events "
        "WHERE workflow_id='wf_transition' AND mutation_id='transition-pass' "
        "AND kind='aggregate_changed'"
    ).fetchone()
    assert pass_event is not None
    next_acceptance = kb.create_task(
        workflow_db, title="accept generation 2", assignee="orchestrator", tenant="tenant-a"
    )
    remediation = kb.create_task(
        workflow_db, title="remediate generation 2", assignee="builder", tenant="tenant-a"
    )
    reverification = kb.create_task(
        workflow_db, title="verify generation 2", assignee="x_qa", tenant="tenant-a"
    )
    reopened = kb.reopen_workflow(
        workflow_db,
        workflow_id="wf_transition",
        designated_acceptance_task_id=next_acceptance,
        members=[
            {"task_id": next_acceptance, "stage_key": "acceptance-2",
             "stage_role": "acceptance", "required": True},
            {"task_id": remediation, "stage_key": "remediation-2",
             "stage_role": "remediation", "required": True},
            {"task_id": reverification, "stage_key": "reverification-2",
             "stage_role": "reverification", "required": True},
        ],
        actor=_actor(capabilities=("workflow.admin",)),
        mutation_id="transition-reopen", expected_version=2,
        reason="remediation and re-verification cycle",
    )

    old, claimed, events = kb.claim_workflow_events_for_subscription(
        workflow_db, workflow_id="wf_transition", role="origin"
    )

    assert old < pass_event["id"]
    assert claimed == pass_event["id"]
    assert [event["id"] for event in events] == [pass_event["id"]]
    assert events[0]["payload"]["resulting_state"] == "PASS"
    assert claimed < reopened["workflow"]["last_event_id"]


def test_workflow_delivery_retry_uses_bounded_exponential_backoff(
    workflow_db, monkeypatch,
):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    kb.create_workflow(
        workflow_db, workflow_id="wf_backoff", name="notify", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=_actor(),
        mutation_id="create-backoff",
        subscription={
            "platform": "telegram", "chat_id": "origin", "notifier_profile": "default",
            "target_states": ["PASS"],
        },
    )
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_backoff", task_id=acceptance, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="backoff-pass", expected_version=1,
    )
    clock = [1_000]
    monkeypatch.setattr(kb.time, "time", lambda: clock[0])
    old, claimed, events = kb.claim_workflow_events_for_subscription(
        workflow_db, workflow_id="wf_backoff"
    )
    assert events

    observed_delays = []
    for _ in (10, 20, 25, 25):
        failed = kb.fail_workflow_delivery(
            workflow_db, workflow_id="wf_backoff",
            claimed_cursor=claimed, old_cursor=old, error_class="TemporaryError",
            max_retries=6, retry_delay_seconds=10, max_retry_delay_seconds=25,
        )
        observed_delays.append(failed["next_attempt_at"] - clock[0])
        clock[0] = failed["next_attempt_at"]
        old, claimed, events = kb.claim_workflow_events_for_subscription(
            workflow_db, workflow_id="wf_backoff"
        )
        assert events

    assert observed_delays == [10, 20, 25, 25]


def test_workflow_subscription_admin_can_skip_one_event_and_disable(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    kb.create_workflow(
        workflow_db, workflow_id="wf_admin_notify", name="notify", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=_actor(),
        mutation_id="create-admin-notify",
        subscription={
            "platform": "telegram", "chat_id": "origin", "notifier_profile": "default",
            "target_states": ["PASS"],
        },
    )
    passed = kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_admin_notify", task_id=acceptance, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="admin-notify-pass", expected_version=1,
    )
    aggregate_event = next(
        event for event in kb.get_workflow(
            workflow_db, "wf_admin_notify",
            actor=_actor(capabilities=("workflow.read",)),
        )["events"]
        if event["kind"] == "aggregate_changed"
    )

    skipped = kb.skip_workflow_subscription_event(
        workflow_db, workflow_id="wf_admin_notify", event_id=aggregate_event["id"],
        actor=_actor(capabilities=("workflow.admin",)), mutation_id="skip-pass-delivery",
        expected_version=passed["workflow"]["version"], reason="operator disposition",
    )
    assert skipped["subscription"]["last_event_id"] == aggregate_event["id"]
    assert skipped["workflow"]["version"] == 3
    assert any(
        event["kind"] == "subscription_event_skipped"
        and event["mutation_id"] == "skip-pass-delivery"
        for event in skipped["events"]
    )
    replayed = kb.replay_workflow_events(
        workflow_db, workflow_id="wf_admin_notify",
        actor=_actor(capabilities=("workflow.read",)),
    )
    assert replayed["state"] == "PASS"
    assert replayed["version"] == 3
    assert kb.workflow_integrity_report(
        workflow_db, workflow_id="wf_admin_notify"
    )["ok"] is True
    retried = kb.skip_workflow_subscription_event(
        workflow_db, workflow_id="wf_admin_notify", event_id=aggregate_event["id"],
        actor=_actor(capabilities=("workflow.admin",)), mutation_id="skip-pass-delivery",
        expected_version=passed["workflow"]["version"], reason="operator disposition",
    )
    assert retried == skipped

    disabled = kb.disable_workflow_subscription(
        workflow_db, workflow_id="wf_admin_notify",
        actor=_actor(capabilities=("workflow.admin",)), reason="retired destination",
    )
    assert disabled["disabled_at"] is not None
    old, new, events = kb.claim_workflow_events_for_subscription(
        workflow_db, workflow_id="wf_admin_notify"
    )
    assert old == new
    assert events == []


def test_workflow_subscription_skip_rejects_leaping_over_next_target_transition(
    workflow_db,
):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    kb.create_workflow(
        workflow_db, workflow_id="wf_skip_order", name="notify", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=_actor(),
        mutation_id="create-skip-order",
        subscription={
            "platform": "telegram", "chat_id": "origin", "notifier_profile": "default",
            "target_states": ["PASS"],
        },
    )
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_skip_order", task_id=acceptance, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="skip-order-pass-one", expected_version=1,
    )
    next_acceptance = kb.create_task(
        workflow_db, title="accept generation 2", assignee="orchestrator", tenant="tenant-a"
    )
    remediation = kb.create_task(
        workflow_db, title="remediate generation 2", assignee="builder", tenant="tenant-a"
    )
    reverification = kb.create_task(
        workflow_db, title="verify generation 2", assignee="x_qa", tenant="tenant-a"
    )
    kb.reopen_workflow(
        workflow_db,
        workflow_id="wf_skip_order",
        designated_acceptance_task_id=next_acceptance,
        members=[
            {"task_id": next_acceptance, "stage_key": "acceptance-2",
             "stage_role": "acceptance", "required": True},
            {"task_id": remediation, "stage_key": "remediation-2",
             "stage_role": "remediation", "required": True},
            {"task_id": reverification, "stage_key": "reverification-2",
             "stage_role": "reverification", "required": True},
        ],
        actor=_actor(capabilities=("workflow.admin",)),
        mutation_id="skip-order-reopen", expected_version=2,
        reason="remediation and re-verification cycle",
    )
    outcome_actor = _actor(capabilities=("workflow.outcome",))
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_skip_order", task_id=remediation, outcome="PASS",
        actor=outcome_actor, mutation_id="skip-order-remediation-pass",
        expected_version=3,
    )
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_skip_order", task_id=reverification, outcome="PASS",
        actor=outcome_actor, mutation_id="skip-order-reverification-pass",
        expected_version=4,
    )
    passed = kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_skip_order", task_id=next_acceptance, outcome="PASS",
        actor=outcome_actor, mutation_id="skip-order-pass-two", expected_version=5,
    )
    aggregate_events = workflow_db.execute(
        "SELECT id,payload FROM kanban_workflow_events "
        "WHERE workflow_id='wf_skip_order' AND kind='aggregate_changed' ORDER BY id"
    ).fetchall()
    assert [json.loads(event["payload"])["resulting_state"] for event in aggregate_events] == [
        "PASS", "ACTIVE", "PASS",
    ]
    first_pass_event, _, second_pass_event = aggregate_events
    subscription_before = dict(workflow_db.execute(
        "SELECT * FROM kanban_workflow_subscriptions "
        "WHERE workflow_id='wf_skip_order' AND role='origin'"
    ).fetchone())
    event_count_before = workflow_db.execute(
        "SELECT count(*) FROM kanban_workflow_events WHERE workflow_id='wf_skip_order'"
    ).fetchone()[0]
    mutation_count_before = workflow_db.execute(
        "SELECT count(*) FROM kanban_workflow_mutations WHERE workflow_id='wf_skip_order'"
    ).fetchone()[0]

    with pytest.raises(kb.WorkflowConflictError, match="next pending target transition"):
        kb.skip_workflow_subscription_event(
            workflow_db, workflow_id="wf_skip_order", event_id=second_pass_event["id"],
            actor=_actor(capabilities=("workflow.admin",)), mutation_id="skip-pass-two-early",
            expected_version=passed["workflow"]["version"], reason="operator disposition",
        )

    subscription_after = dict(workflow_db.execute(
        "SELECT * FROM kanban_workflow_subscriptions "
        "WHERE workflow_id='wf_skip_order' AND role='origin'"
    ).fetchone())
    workflow_after = workflow_db.execute(
        "SELECT version FROM kanban_workflows WHERE id='wf_skip_order'"
    ).fetchone()
    assert subscription_after == subscription_before
    assert workflow_after["version"] == passed["workflow"]["version"]
    assert workflow_db.execute(
        "SELECT count(*) FROM kanban_workflow_events WHERE workflow_id='wf_skip_order'"
    ).fetchone()[0] == event_count_before
    assert workflow_db.execute(
        "SELECT count(*) FROM kanban_workflow_mutations WHERE workflow_id='wf_skip_order'"
    ).fetchone()[0] == mutation_count_before

    skipped = kb.skip_workflow_subscription_event(
        workflow_db, workflow_id="wf_skip_order", event_id=first_pass_event["id"],
        actor=_actor(capabilities=("workflow.admin",)), mutation_id="skip-pass-one",
        expected_version=passed["workflow"]["version"], reason="operator disposition",
    )
    assert skipped["subscription"]["last_event_id"] == first_pass_event["id"]
    assert kb.workflow_integrity_report(
        workflow_db, workflow_id="wf_skip_order"
    )["ok"] is True


def test_workflow_subscription_admin_actions_fail_closed(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    created = kb.create_workflow(
        workflow_db, workflow_id="wf_admin_closed", name="notify", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=_actor(),
        mutation_id="create-admin-closed",
        subscription={
            "platform": "telegram", "chat_id": "origin", "notifier_profile": "default",
        },
    )
    with pytest.raises(kb.WorkflowAuthorizationError):
        kb.disable_workflow_subscription(
            workflow_db, workflow_id="wf_admin_closed",
            actor=_actor(capabilities=("workflow.manage",)), reason="forged admin",
        )
    with pytest.raises(kb.WorkflowAuthorizationError):
        kb.skip_workflow_subscription_event(
            workflow_db, workflow_id="wf_admin_closed",
            event_id=created["workflow"]["last_event_id"],
            actor=_actor(capabilities=("workflow.manage",)), mutation_id="forged-skip",
            expected_version=created["workflow"]["version"], reason="forged admin",
        )


def test_replay_and_integrity_report_detect_materialized_state_corruption(workflow_db):
    acceptance = _new_workflow(workflow_db)
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_test", task_id=acceptance, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="pass-for-replay", expected_version=1,
    )
    replayed = kb.replay_workflow_events(
        workflow_db, workflow_id="wf_test",
        actor=_actor(capabilities=("workflow.read",)),
    )
    assert replayed["state"] == "PASS"
    assert replayed["version"] == 2
    assert kb.workflow_integrity_report(workflow_db, workflow_id="wf_test")["ok"] is True

    workflow_db.execute("UPDATE kanban_workflows SET state='ACTIVE' WHERE id='wf_test'")
    report = kb.workflow_integrity_report(workflow_db, workflow_id="wf_test")
    assert report["ok"] is False
    assert any("replay" in error for error in report["errors"])

    workflow_db.execute(
        "UPDATE kanban_workflow_events SET payload_schema_version=999 "
        "WHERE workflow_id='wf_test' AND id=(SELECT max(id) FROM kanban_workflow_events)"
    )
    with pytest.raises(kb.WorkflowIntegrityError, match="schema"):
        kb.replay_workflow_events(
            workflow_db, workflow_id="wf_test",
            actor=_actor(capabilities=("workflow.read",)),
        )


def test_replay_bootstraps_identity_and_aggregate_without_materialized_rows(workflow_db):
    acceptance = _new_workflow(workflow_db)
    implementation = kb.create_task(
        workflow_db, title="implementation", assignee="builder", tenant="tenant-a"
    )
    kb.add_workflow_member(
        workflow_db, workflow_id="wf_test", task_id=implementation,
        stage_key="implementation", stage_role="implementation", required=True,
        actor=_actor(), mutation_id="add-for-empty-replay", expected_version=1,
    )
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_test", task_id=implementation, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="pass-for-empty-replay", expected_version=2,
    )

    workflow_db.commit()
    workflow_db.execute("PRAGMA foreign_keys=OFF")
    for table in (
        "kanban_workflow_outcomes", "kanban_workflow_subscriptions",
        "kanban_workflow_members", "kanban_workflow_generations", "kanban_workflows",
    ):
        workflow_db.execute(f"DELETE FROM {table} WHERE workflow_id='wf_test'" if table != "kanban_workflows"
                            else "DELETE FROM kanban_workflows WHERE id='wf_test'")
    workflow_db.commit()
    workflow_db.execute("PRAGMA foreign_keys=ON")

    replay = kb.replay_workflow_events(
        workflow_db, workflow_id="wf_test",
        actor=_actor(capabilities=("workflow.read",)),
    )

    assert replay["tenant"] == "tenant-a"
    assert replay["board_identity"] == "board:test"
    assert replay["state"] == "ACTIVE"
    assert replay["version"] == 3
    assert replay["generations"]["1"]["designated_acceptance_task_id"] == acceptance
    assert replay["members"][f"1:{implementation}"]["stage_key"] == "implementation"
    assert replay["outcomes"][f"1:{implementation}"]["outcome"] == "PASS"

    with pytest.raises(kb.WorkflowAuthorizationError, match="tenant"):
        kb.replay_workflow_events(
            workflow_db, workflow_id="wf_test",
            actor=_actor(tenant="tenant-b", capabilities=("workflow.read",)),
        )


def test_replay_reconstructs_create_and_replaced_subscription_destinations(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    kb.create_workflow(
        workflow_db, workflow_id="wf_subscription_replay", name="release",
        tenant="tenant-a", designated_acceptance_task_id=acceptance,
        actor=_actor(), mutation_id="create-subscription-replay",
        subscription={
            "platform": "telegram", "chat_id": "chat-original",
            "chat_type": "group", "thread_id": "thread-original",
            "user_id": "user-original", "notifier_profile": "default",
            "delivery_metadata": {"topic": "release"},
            "target_states": ["PASS"],
        },
    )
    created_replay = kb.replay_workflow_events(
        workflow_db, workflow_id="wf_subscription_replay",
        actor=_actor(capabilities=("workflow.read",)),
    )
    assert created_replay["subscription"]["chat_id"] == "chat-original"

    kb.set_workflow_subscription(
        workflow_db, workflow_id="wf_subscription_replay", platform="discord",
        chat_id="chat-replacement", chat_type="channel", thread_id="thread-replacement",
        user_id="user-replacement", notifier_profile="qa",
        delivery_metadata={"channel_name": "release-qa"}, target_states=["NEEDS_INPUT", "PASS"],
        actor=_actor(capabilities=("workflow.admin",)),
        mutation_id="replace-subscription", expected_version=1,
    )

    replay = kb.replay_workflow_events(
        workflow_db, workflow_id="wf_subscription_replay",
        actor=_actor(capabilities=("workflow.read",)),
    )
    assert replay["subscription"] == {
        "role": "origin", "platform": "discord", "chat_id": "chat-replacement",
        "chat_type": "channel", "thread_id": "thread-replacement",
        "user_id": "user-replacement", "notifier_profile": "qa",
        "delivery_metadata": {"channel_name": "release-qa"},
        "target_states": ["NEEDS_INPUT", "PASS"], "tenant": "tenant-a",
    }


def test_create_subscription_metadata_is_canonical_and_idempotent(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    subscription = {
        "platform": "telegram", "chat_id": "chat-canonical",
        "notifier_profile": "default",
        "delivery_metadata": {"scalar": "kept", "count": 1, "flag": True},
        "target_states": ["PASS"],
    }
    created = kb.create_workflow(
        workflow_db, workflow_id="wf_create_metadata", name="release",
        tenant="tenant-a", designated_acceptance_task_id=acceptance,
        actor=_actor(), mutation_id="create-metadata", subscription=subscription,
    )
    retried = kb.create_workflow(
        workflow_db, workflow_id="wf_create_metadata", name="release",
        tenant="tenant-a", designated_acceptance_task_id=acceptance,
        actor=_actor(), mutation_id="create-metadata",
        subscription={
            **subscription,
            "delivery_metadata": {"flag": True, "count": 1, "scalar": "kept"},
        },
    )

    replay = kb.replay_workflow_events(
        workflow_db, workflow_id="wf_create_metadata",
        actor=_actor(capabilities=("workflow.read",)),
    )
    materialized = workflow_db.execute(
        "SELECT delivery_metadata FROM kanban_workflow_subscriptions "
        "WHERE workflow_id='wf_create_metadata'"
    ).fetchone()

    assert retried == created
    assert replay["subscription"]["delivery_metadata"] == {
        "count": 1, "flag": True, "scalar": "kept",
    }
    assert json.loads(materialized["delivery_metadata"]) == replay["subscription"]["delivery_metadata"]
    assert kb.workflow_integrity_report(
        workflow_db, workflow_id="wf_create_metadata"
    )["ok"] is True


@pytest.mark.parametrize(
    "delivery_metadata",
    [
        {"nested": {"x": 1}},
        {"none": None},
        {"items": [1, 2]},
    ],
)
def test_create_rejects_unsupported_subscription_metadata_atomically(
    workflow_db, delivery_metadata,
):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )

    with pytest.raises(ValueError, match="delivery metadata"):
        kb.create_workflow(
            workflow_db, workflow_id="wf_invalid_create_metadata", name="release",
            tenant="tenant-a", designated_acceptance_task_id=acceptance,
            actor=_actor(), mutation_id="invalid-create-metadata",
            subscription={
                "platform": "telegram", "chat_id": "chat-invalid",
                "notifier_profile": "default", "delivery_metadata": delivery_metadata,
            },
        )

    assert workflow_db.execute(
        "SELECT 1 FROM kanban_workflows WHERE id='wf_invalid_create_metadata'"
    ).fetchone() is None
    assert workflow_db.execute(
        "SELECT 1 FROM kanban_workflow_events WHERE workflow_id='wf_invalid_create_metadata'"
    ).fetchone() is None
    assert workflow_db.execute(
        "SELECT 1 FROM kanban_workflow_mutations WHERE workflow_id='wf_invalid_create_metadata'"
    ).fetchone() is None


@pytest.mark.parametrize(
    "delivery_metadata",
    [
        {"nested": {"x": 1}},
        {"none": None},
        {"items": [1, 2]},
    ],
)
def test_set_subscription_rejects_unsupported_metadata_without_mutation(
    workflow_db, delivery_metadata,
):
    _new_workflow(workflow_db, workflow_id="wf_invalid_set_metadata")

    with pytest.raises(ValueError, match="delivery metadata"):
        kb.set_workflow_subscription(
            workflow_db, workflow_id="wf_invalid_set_metadata", platform="discord",
            chat_id="chat-invalid", notifier_profile="default",
            delivery_metadata=delivery_metadata,
            actor=_actor(capabilities=("workflow.admin",)),
            mutation_id="invalid-set-metadata", expected_version=1,
        )

    replay = kb.replay_workflow_events(
        workflow_db, workflow_id="wf_invalid_set_metadata",
        actor=_actor(capabilities=("workflow.read",)),
    )
    assert replay["version"] == 1
    assert replay["subscription"] is None
    assert workflow_db.execute(
        "SELECT 1 FROM kanban_workflow_mutations "
        "WHERE workflow_id='wf_invalid_set_metadata' AND mutation_id='invalid-set-metadata'"
    ).fetchone() is None
    assert kb.workflow_integrity_report(
        workflow_db, workflow_id="wf_invalid_set_metadata"
    )["ok"] is True


def test_set_subscription_metadata_is_canonical_and_idempotent(workflow_db):
    _new_workflow(workflow_db, workflow_id="wf_set_metadata")
    kwargs = {
        "workflow_id": "wf_set_metadata", "platform": "discord",
        "chat_id": "chat-canonical", "notifier_profile": "default",
        "actor": _actor(capabilities=("workflow.admin",)),
        "mutation_id": "set-metadata", "expected_version": 1,
    }
    changed = kb.set_workflow_subscription(
        workflow_db, delivery_metadata={"scalar": "kept", "count": 1}, **kwargs,
    )
    retried = kb.set_workflow_subscription(
        workflow_db, delivery_metadata={"count": 1, "scalar": "kept"}, **kwargs,
    )

    replay = kb.replay_workflow_events(
        workflow_db, workflow_id="wf_set_metadata",
        actor=_actor(capabilities=("workflow.read",)),
    )
    materialized = workflow_db.execute(
        "SELECT delivery_metadata FROM kanban_workflow_subscriptions "
        "WHERE workflow_id='wf_set_metadata'"
    ).fetchone()

    assert retried == changed
    assert replay["subscription"]["delivery_metadata"] == {"count": 1, "scalar": "kept"}
    assert json.loads(materialized["delivery_metadata"]) == replay["subscription"]["delivery_metadata"]
    assert kb.workflow_integrity_report(
        workflow_db, workflow_id="wf_set_metadata"
    )["ok"] is True


def test_integrity_detects_subscription_destination_divergence(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    kb.create_workflow(
        workflow_db, workflow_id="wf_subscription_integrity", name="release",
        tenant="tenant-a", designated_acceptance_task_id=acceptance,
        actor=_actor(), mutation_id="create-subscription-integrity",
        subscription={
            "platform": "telegram", "chat_id": "chat-canonical",
            "notifier_profile": "default", "target_states": ["PASS"],
        },
    )
    assert kb.workflow_integrity_report(
        workflow_db, workflow_id="wf_subscription_integrity"
    )["ok"] is True

    workflow_db.execute(
        "UPDATE kanban_workflow_subscriptions SET chat_id='chat-corrupted' "
        "WHERE workflow_id='wf_subscription_integrity'"
    )

    report = kb.workflow_integrity_report(
        workflow_db, workflow_id="wf_subscription_integrity"
    )
    assert report["ok"] is False
    assert any("subscription" in error for error in report["errors"])


def test_replay_and_integrity_reject_missing_mutation_ledger_entry(workflow_db):
    _new_workflow(workflow_db)
    workflow_db.execute(
        "DELETE FROM kanban_workflow_mutations "
        "WHERE workflow_id='wf_test' AND mutation_id='create-wf_test'"
    )

    with pytest.raises(kb.WorkflowIntegrityError, match="mutation ledger"):
        kb.replay_workflow_events(
            workflow_db, workflow_id="wf_test",
            actor=_actor(capabilities=("workflow.read",)),
        )
    report = kb.workflow_integrity_report(workflow_db, workflow_id="wf_test")
    assert report["ok"] is False
    assert any("mutation ledger" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("tamper_sql", "expected_error"),
    [
        (
            "UPDATE kanban_workflow_mutations SET request_digest='corrupt' "
            "WHERE workflow_id='wf_test' AND mutation_id='pass-ledger'",
            "request digest",
        ),
        (
            "UPDATE kanban_workflow_mutations SET canonical_event_id=last_event_id "
            "WHERE workflow_id='wf_test' AND mutation_id='pass-ledger'",
            "canonical event",
        ),
        (
            "UPDATE kanban_workflow_mutations SET last_event_id=first_event_id "
            "WHERE workflow_id='wf_test' AND mutation_id='pass-ledger'",
            "event range",
        ),
        (
            "UPDATE kanban_workflow_mutations SET response_json='{}' "
            "WHERE workflow_id='wf_test' AND mutation_id='pass-ledger'",
            "response",
        ),
    ],
)
def test_replay_rejects_corrupt_mutation_ledger_linkage(
    workflow_db, tamper_sql, expected_error,
):
    acceptance = _new_workflow(workflow_db)
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_test", task_id=acceptance, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="pass-ledger", expected_version=1,
    )
    workflow_db.execute(tamper_sql)

    with pytest.raises(kb.WorkflowIntegrityError, match=expected_error):
        kb.replay_workflow_events(
            workflow_db, workflow_id="wf_test",
            actor=_actor(capabilities=("workflow.read",)),
        )


@pytest.mark.parametrize(
    ("section", "field", "corrupt_value"),
    [
        ("members", "stage_key", "CORRUPTED_STAGE"),
        ("generation", "designated_acceptance_task_id", "CORRUPTED_TASK"),
        ("subscription", "notifier_profile", "CORRUPTED_PROFILE"),
        ("outcomes", "outcome", "CORRUPTED_OUTCOME"),
        ("events", "kind", "CORRUPTED_EVENT"),
    ],
)
def test_replay_integrity_doctor_and_retry_reject_corrupt_ledger_response_sections(
    workflow_db, section, field, corrupt_value,
):
    acceptance = _new_workflow(workflow_db)
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_test", task_id=acceptance, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="pass-semantic-response", expected_version=1,
    )
    subscription_kwargs = {
        "workflow_id": "wf_test", "platform": "discord", "chat_id": "chat-integrity",
        "notifier_profile": "default", "delivery_metadata": {"channel": "release"},
        "target_states": ["PASS"],
        "actor": _actor(capabilities=("workflow.admin",)),
        "mutation_id": "subscription-semantic-response", "expected_version": 2,
    }
    kb.set_workflow_subscription(workflow_db, **subscription_kwargs)
    ledger = workflow_db.execute(
        "SELECT response_json FROM kanban_workflow_mutations "
        "WHERE workflow_id='wf_test' AND mutation_id='subscription-semantic-response'"
    ).fetchone()
    response = json.loads(ledger["response_json"])
    target = response[section][0] if isinstance(response[section], list) else response[section]
    target[field] = corrupt_value
    workflow_db.execute(
        "UPDATE kanban_workflow_mutations SET response_json=? "
        "WHERE workflow_id='wf_test' AND mutation_id='subscription-semantic-response'",
        (json.dumps(response),),
    )

    with pytest.raises(kb.WorkflowIntegrityError, match="ledger response"):
        kb.replay_workflow_events(
            workflow_db, workflow_id="wf_test",
            actor=_actor(capabilities=("workflow.read",)),
        )
    report = kb.workflow_integrity_report(workflow_db, workflow_id="wf_test")
    assert report["ok"] is False
    assert any("ledger response" in error for error in report["errors"])
    diagnostics = doctor._check_kanban_workflow_health(
        workflow_db.execute("PRAGMA database_list").fetchone()["file"]
    )
    assert any("integrity mismatch" in diagnostic for diagnostic in diagnostics)
    with pytest.raises(kb.WorkflowIntegrityError, match="ledger response"):
        kb.set_workflow_subscription(workflow_db, **subscription_kwargs)


def test_qa_outcome_requires_active_dispatch_run_and_claim_binding(workflow_db):
    _new_workflow(workflow_db)
    qa_task = kb.create_task(
        workflow_db, title="independent QA", assignee="x_qa", tenant="tenant-a"
    )
    kb.add_workflow_member(
        workflow_db, workflow_id="wf_test", task_id=qa_task,
        stage_key="qa", stage_role="qa", required=True,
        actor=_actor(), mutation_id="add-qa", expected_version=1,
    )
    with pytest.raises(kb.WorkflowAuthorizationError, match="binding"):
        kb.record_workflow_outcome(
            workflow_db, workflow_id="wf_test", task_id=qa_task, outcome="PASS",
            actor=_actor(capabilities=("workflow.outcome",)),
            mutation_id="unbound-qa-pass", expected_version=2,
        )

    claim = "claim-qa"
    cur = workflow_db.execute(
        "INSERT INTO task_runs (task_id,profile,status,claim_lock,claim_expires,started_at) "
        "VALUES (?,?, 'running', ?, ?, ?)",
        (qa_task, "x_qa", claim, int(time.time()) + 300, int(time.time())),
    )
    run_id = int(cur.lastrowid)
    workflow_db.execute(
        "UPDATE tasks SET status='running',current_run_id=?,claim_lock=?,claim_expires=? "
        "WHERE id=?",
        (run_id, claim, int(time.time()) + 300, qa_task),
    )
    result = kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_test", task_id=qa_task, outcome="PASS",
        actor=_worker_actor(task_id=qa_task, run_id=run_id, claim_lock=claim),
        mutation_id="bound-qa-pass", expected_version=2, run_id=run_id,
    )
    assert result["workflow"]["state"] == "ACTIVE"


def test_remediation_outcome_is_distinct_from_task_status_and_requires_followup(workflow_db):
    acceptance = _new_workflow(workflow_db)
    workflow_db.execute("UPDATE tasks SET status='done' WHERE id=?", (acceptance,))
    result = kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=acceptance,
        outcome="REMEDIATION_REQUIRED",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="needs-remediation",
        expected_version=1,
        summary="independent QA found a defect",
    )
    assert kb.get_task(workflow_db, acceptance).status == "done"
    assert result["workflow"]["state"] == "NEEDS_INPUT"


def test_terminal_generation_requires_explicit_reopen(workflow_db):
    acceptance = _new_workflow(workflow_db)
    kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=acceptance,
        outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="pass-generation-one",
        expected_version=1,
    )
    next_acceptance = kb.create_task(
        workflow_db, title="accept generation 2", assignee="orchestrator", tenant="tenant-a"
    )
    remediation = kb.create_task(
        workflow_db, title="remediate generation 2", assignee="builder", tenant="tenant-a"
    )
    reverification = kb.create_task(
        workflow_db, title="verify generation 2", assignee="x_qa", tenant="tenant-a"
    )
    reopened = kb.reopen_workflow(
        workflow_db,
        workflow_id="wf_test",
        designated_acceptance_task_id=next_acceptance,
        members=[
            {
                "task_id": next_acceptance,
                "stage_key": "acceptance-2",
                "stage_role": "acceptance",
                "required": True,
            },
            {
                "task_id": remediation,
                "stage_key": "remediation-2",
                "stage_role": "remediation",
                "required": True,
            },
            {
                "task_id": reverification,
                "stage_key": "reverification-2",
                "stage_role": "reverification",
                "required": True,
            },
        ],
        actor=_actor(capabilities=("workflow.admin",)),
        mutation_id="reopen-generation-two",
        expected_version=2,
        reason="remediation and re-verification cycle",
    )
    assert reopened["workflow"]["active_generation"] == 2
    assert reopened["workflow"]["state"] == "ACTIVE"
    assert reopened["generation"]["designated_acceptance_task_id"] == next_acceptance
    old = workflow_db.execute(
        "SELECT generation_state,superseded_by_generation FROM kanban_workflow_generations "
        "WHERE workflow_id='wf_test' AND generation=1"
    ).fetchone()
    assert tuple(old) == ("PASS", 2)


def test_outcomes_reduce_required_members_to_pass_with_event_batch(workflow_db):
    acceptance = _new_workflow(workflow_db)
    implementation = kb.create_task(
        workflow_db, title="implement", assignee="builder", tenant="tenant-a"
    )
    kb.add_workflow_member(
        workflow_db,
        workflow_id="wf_test",
        task_id=implementation,
        stage_key="implementation",
        stage_role="implementation",
        required=True,
        actor=_actor(),
        mutation_id="add-implementation",
        expected_version=1,
    )
    outcome_actor = _actor(capabilities=("workflow.outcome",))

    implementation_pass = kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=implementation,
        outcome="PASS",
        actor=outcome_actor,
        mutation_id="impl-pass",
        expected_version=2,
        summary="implementation complete",
    )
    assert implementation_pass["workflow"]["state"] == "ACTIVE"

    acceptance_pass = kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=acceptance,
        outcome="PASS",
        actor=outcome_actor,
        mutation_id="accept-pass",
        expected_version=3,
        summary="accepted",
    )
    assert acceptance_pass["workflow"]["state"] == "PASS"
    assert acceptance_pass["generation"]["generation_state"] == "PASS"
    rows = workflow_db.execute(
        "SELECT batch_seq,event_role,kind FROM kanban_workflow_events "
        "WHERE workflow_id='wf_test' AND mutation_id='accept-pass' ORDER BY batch_seq"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (0, "canonical", "outcome_recorded"),
        (1, "derived", "aggregate_changed"),
    ]

    retry = kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=acceptance,
        outcome="PASS",
        actor=outcome_actor,
        mutation_id="accept-pass",
        expected_version=3,
        summary="accepted",
    )
    assert retry == acceptance_pass

    with pytest.raises(kb.WorkflowConflictError, match="reopened"):
        kb.add_workflow_member(
            workflow_db,
            workflow_id="wf_test",
            task_id=kb.create_task(
                workflow_db, title="late", assignee="builder", tenant="tenant-a"
            ),
            stage_key="late",
            stage_role="remediation",
            required=True,
            actor=_actor(),
            mutation_id="late-add",
            expected_version=4,
        )


def test_links_touching_enrolled_tasks_fail_closed_on_tenant_mismatch(workflow_db):
    acceptance = _new_workflow(workflow_db)
    foreign = kb.create_task(
        workflow_db, title="foreign", assignee="worker", tenant="tenant-b"
    )
    kb.add_notify_sub(
        workflow_db, task_id=acceptance, platform="telegram", chat_id="origin"
    )

    with pytest.raises(kb.WorkflowIntegrityError, match="tenant"):
        kb.link_tasks(workflow_db, acceptance, foreign)
    assert workflow_db.execute(
        "SELECT COUNT(*) FROM task_links WHERE parent_id=? AND child_id=?",
        (acceptance, foreign),
    ).fetchone()[0] == 0
    assert kb.list_notify_subs(workflow_db, foreign) == []

    with pytest.raises(kb.WorkflowIntegrityError, match="tenant"):
        kb.create_task(
            workflow_db,
            title="cross-tenant child",
            assignee="worker",
            tenant="tenant-b",
            parents=[acceptance],
        )
    assert workflow_db.execute(
        "SELECT COUNT(*) FROM tasks WHERE title='cross-tenant child'"
    ).fetchone()[0] == 0


def test_add_member_is_explicit_cas_idempotent_and_tenant_safe(workflow_db):
    _new_workflow(workflow_db)
    implementation = kb.create_task(
        workflow_db, title="implement", assignee="builder", tenant="tenant-a"
    )
    foreign = kb.create_task(
        workflow_db, title="foreign", assignee="builder", tenant="tenant-b"
    )

    first = kb.add_workflow_member(
        workflow_db,
        workflow_id="wf_test",
        task_id=implementation,
        stage_key="implementation",
        stage_role="implementation",
        required=True,
        actor=_actor(),
        mutation_id="add-implementation",
        expected_version=1,
    )
    retry = kb.add_workflow_member(
        workflow_db,
        workflow_id="wf_test",
        task_id=implementation,
        stage_key="implementation",
        stage_role="implementation",
        required=True,
        actor=_actor(),
        mutation_id="add-implementation",
        expected_version=1,
    )

    assert retry == first
    assert first["workflow"]["version"] == 2
    assert {member["task_id"] for member in first["members"]} == {
        implementation,
        first["generation"]["designated_acceptance_task_id"],
    }
    assert workflow_db.execute(
        "SELECT COUNT(*) FROM task_links WHERE parent_id=? OR child_id=?",
        (implementation, implementation),
    ).fetchone()[0] == 0

    with pytest.raises(kb.WorkflowConflictError, match="expected version 1"):
        kb.add_workflow_member(
            workflow_db,
            workflow_id="wf_test",
            task_id=foreign,
            stage_key="foreign",
            stage_role="implementation",
            required=True,
            actor=_actor(),
            mutation_id="stale-add",
            expected_version=1,
        )
    with pytest.raises(kb.WorkflowIntegrityError, match="tenant"):
        kb.add_workflow_member(
            workflow_db,
            workflow_id="wf_test",
            task_id=foreign,
            stage_key="foreign",
            stage_role="implementation",
            required=True,
            actor=_actor(),
            mutation_id="foreign-add",
            expected_version=2,
        )
    assert workflow_db.execute(
        "SELECT COUNT(*) FROM kanban_workflow_members WHERE task_id=?", (foreign,)
    ).fetchone()[0] == 0


def test_workflow_create_is_idempotent_and_designates_acceptance(workflow_db):
    acceptance = kb.create_task(
        workflow_db,
        title="accept",
        assignee="orchestrator",
        tenant="tenant-a",
        session_id="provenance-only",
    )
    actor = kb.KanbanActorContext(
        principal_id="svc:orchestrator",
        profile_name="orchestrator",
        board_identity="board:test",
        tenant="tenant-a",
        capabilities=frozenset({"workflow.manage"}),
        source_kind="orchestrator",
    )

    first = kb.create_workflow(
        workflow_db,
        workflow_id="wf_test",
        name="release",
        tenant="tenant-a",
        designated_acceptance_task_id=acceptance,
        actor=actor,
        mutation_id="mut-create",
    )
    retry = kb.create_workflow(
        workflow_db,
        workflow_id="wf_test",
        name="release",
        tenant="tenant-a",
        designated_acceptance_task_id=acceptance,
        actor=actor,
        mutation_id="mut-create",
    )

    assert retry == first
    assert first["workflow"]["state"] == "ACTIVE"
    assert first["workflow"]["version"] == 1
    assert first["generation"]["designated_acceptance_task_id"] == acceptance
    assert first["members"] == [
        {
            "task_id": acceptance,
            "stage_key": "acceptance",
            "stage_role": "acceptance",
            "required": True,
        }
    ]
    assert workflow_db.execute(
        "SELECT COUNT(*) FROM kanban_workflow_events WHERE workflow_id='wf_test'"
    ).fetchone()[0] == 1
    assert workflow_db.execute(
        "SELECT COUNT(*) FROM kanban_workflow_mutations WHERE workflow_id='wf_test'"
    ).fetchone()[0] == 1

    with pytest.raises(kb.WorkflowConflictError, match="mutation_id"):
        kb.create_workflow(
            workflow_db,
            workflow_id="wf_test",
            name="different request",
            tenant="tenant-a",
            designated_acceptance_task_id=acceptance,
            actor=actor,
            mutation_id="mut-create",
        )


def test_workflow_schema_is_additive_and_idempotent(workflow_db, tmp_path):
    expected = {
        "kanban_workflows",
        "kanban_workflow_generations",
        "kanban_workflow_members",
        "kanban_workflow_outcomes",
        "kanban_workflow_events",
        "kanban_workflow_mutations",
        "kanban_workflow_subscriptions",
    }
    names = {
        row[0]
        for row in workflow_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'kanban_workflow%'"
        )
    }
    assert expected <= names

    # Reopening a new-schema board must not rewrite legacy task rows.
    task_id = kb.create_task(
        workflow_db, title="legacy task", assignee="worker", tenant="tenant-a"
    )
    workflow_db.close()
    path = tmp_path / "workflow.db"
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    kb.init_db(path)
    with kb.connect_closing(path) as reopened:
        assert kb.get_task(reopened, task_id).title == "legacy task"
        assert expected <= {
            row[0]
            for row in reopened.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'kanban_workflow%'"
            )
        }


def test_legacy_board_copy_gains_workflow_schema_without_task_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "legacy-copy.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    with kb.connect_closing(path) as conn:
        task_id = kb.create_task(
            conn, title="legacy preserved", assignee="worker", tenant="tenant-a"
        )
    raw = sqlite3.connect(path)
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        for table in (
            "kanban_workflow_subscriptions", "kanban_workflow_mutations",
            "kanban_workflow_outcomes", "kanban_workflow_members",
            "kanban_workflow_events", "kanban_workflow_generations",
            "kanban_workflows",
        ):
            raw.execute(f"DROP TABLE {table}")
        raw.commit()
    finally:
        raw.close()
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    with kb.connect_closing(path) as upgraded:
        assert kb.get_task(upgraded, task_id).title == "legacy preserved"
        assert upgraded.execute(
            "SELECT COUNT(*) FROM kanban_workflows"
        ).fetchone()[0] == 0


def test_member_removal_is_explicit_audited_and_independent_of_unlink(workflow_db):
    acceptance = _new_workflow(workflow_db)
    implementation = kb.create_task(
        workflow_db, title="implement", assignee="builder", tenant="tenant-a"
    )
    kb.add_workflow_member(
        workflow_db, workflow_id="wf_test", task_id=implementation,
        stage_key="implementation", stage_role="implementation", required=True,
        actor=_actor(), mutation_id="add-before-remove", expected_version=1,
    )
    kb.link_tasks(workflow_db, implementation, acceptance)

    removed = kb.remove_workflow_member(
        workflow_db, workflow_id="wf_test", task_id=implementation,
        actor=_actor(), mutation_id="remove-implementation", expected_version=2,
        reason="stage replaced",
    )

    assert {member["task_id"] for member in removed["members"]} == {acceptance}
    assert workflow_db.execute(
        "SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?",
        (implementation, acceptance),
    ).fetchone() is not None
    event = workflow_db.execute(
        "SELECT kind,payload FROM kanban_workflow_events "
        "WHERE workflow_id='wf_test' AND mutation_id='remove-implementation'"
    ).fetchone()
    assert event["kind"] == "member_removed"
    assert "stage replaced" in event["payload"]


def test_cancel_is_terminal_idempotent_and_requires_admin(workflow_db):
    _new_workflow(workflow_db)
    with pytest.raises(kb.WorkflowAuthorizationError):
        kb.cancel_workflow(
            workflow_db, workflow_id="wf_test",
            actor=_actor(capabilities=("workflow.manage",)),
            mutation_id="cancel-forged", expected_version=1, reason="stop",
        )

    cancelled = kb.cancel_workflow(
        workflow_db, workflow_id="wf_test",
        actor=_actor(capabilities=("workflow.admin",)),
        mutation_id="cancel-valid", expected_version=1, reason="operator cancelled",
    )
    retry = kb.cancel_workflow(
        workflow_db, workflow_id="wf_test",
        actor=_actor(capabilities=("workflow.admin",)),
        mutation_id="cancel-valid", expected_version=1, reason="operator cancelled",
    )
    assert retry == cancelled
    assert cancelled["workflow"]["state"] == "CANCELLED"
    assert cancelled["generation"]["generation_state"] == "CANCELLED"
    with pytest.raises(kb.WorkflowConflictError, match="terminal"):
        kb.add_workflow_member(
            workflow_db, workflow_id="wf_test",
            task_id=kb.create_task(
                workflow_db, title="late", assignee="builder", tenant="tenant-a"
            ),
            stage_key="late", stage_role="other", required=True,
            actor=_actor(), mutation_id="late-after-cancel", expected_version=2,
        )


def test_subscription_destination_is_workflow_owned_and_admin_mutable(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a",
        session_id="task-provenance",
    )
    kb.create_workflow(
        workflow_db, workflow_id="wf_destination", name="delivery", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=_actor(), mutation_id="create-destination",
        subscription={
            "platform": "api_server", "chat_id": "origin-session",
            "notifier_profile": "default",
        },
    )
    workflow_db.execute(
        "UPDATE tasks SET session_id='mutated-task-session' WHERE id=?", (acceptance,)
    )
    updated = kb.set_workflow_subscription(
        workflow_db, workflow_id="wf_destination",
        platform="api_server", chat_id="new-origin-session",
        notifier_profile="default", actor=_actor(capabilities=("workflow.admin",)),
        mutation_id="move-destination", expected_version=1,
    )
    assert updated["subscription"]["chat_id"] == "new-origin-session"
    created = next(event for event in updated["events"] if event["kind"] == "workflow_created")
    assert created["payload"]["mutation"]["member"]["task_session_id"] == "task-provenance"
    assert updated["subscription"]["chat_id"] != "mutated-task-session"


def test_idempotent_retry_still_requires_current_actor_authorization(workflow_db):
    _new_workflow(workflow_db)
    member = kb.create_task(
        workflow_db, title="implementation", assignee="builder", tenant="tenant-a"
    )
    kb.add_workflow_member(
        workflow_db, workflow_id="wf_test", task_id=member,
        stage_key="implementation", stage_role="implementation", required=True,
        actor=_actor(), mutation_id="authorized-add", expected_version=1,
    )
    foreign_board_actor = kb.KanbanActorContext(
        principal_id="svc:foreign", profile_name="orchestrator",
        board_identity="board:other", tenant="tenant-a",
        capabilities=frozenset({"workflow.manage"}), source_kind="orchestrator",
    )
    with pytest.raises(kb.WorkflowAuthorizationError, match="board"):
        kb.add_workflow_member(
            workflow_db, workflow_id="wf_test", task_id=member,
            stage_key="implementation", stage_role="implementation", required=True,
            actor=foreign_board_actor, mutation_id="authorized-add", expected_version=1,
        )


def test_workflow_actor_board_identity_is_immutable(workflow_db):
    _new_workflow(workflow_db)
    member = kb.create_task(
        workflow_db, title="implementation", assignee="builder", tenant="tenant-a"
    )
    foreign_board_actor = kb.KanbanActorContext(
        principal_id="svc:orchestrator",
        profile_name="orchestrator",
        board_identity="board:other",
        tenant="tenant-a",
        capabilities=frozenset({"workflow.manage"}),
        source_kind="orchestrator",
    )
    with pytest.raises(kb.WorkflowAuthorizationError, match="board"):
        kb.add_workflow_member(
            workflow_db,
            workflow_id="wf_test",
            task_id=member,
            stage_key="implementation",
            stage_role="implementation",
            required=True,
            actor=foreign_board_actor,
            mutation_id="foreign-board-add",
            expected_version=1,
        )


def test_add_member_reduces_existing_remediation_obligation(workflow_db):
    acceptance = _new_workflow(workflow_db)
    result = kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=acceptance,
        outcome="REMEDIATION_REQUIRED",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="remediation-without-followup",
        expected_version=1,
    )
    assert result["workflow"]["state"] == "NEEDS_INPUT"
    remediation = kb.create_task(
        workflow_db, title="fix", assignee="builder", tenant="tenant-a"
    )
    added = kb.add_workflow_member(
        workflow_db,
        workflow_id="wf_test",
        task_id=remediation,
        stage_key="remediation",
        stage_role="remediation",
        required=True,
        actor=_actor(),
        mutation_id="add-followup",
        expected_version=2,
    )
    assert added["workflow"]["state"] == "REMEDIATION_REQUIRED"
    assert [
        event["kind"]
        for event in added["events"]
        if event["mutation_id"] == "add-followup"
    ] == ["member_added", "aggregate_changed"]


def test_second_required_acceptance_member_is_rejected(workflow_db):
    _new_workflow(workflow_db)
    second = kb.create_task(
        workflow_db, title="second acceptance", assignee="orchestrator", tenant="tenant-a"
    )
    with pytest.raises(kb.WorkflowIntegrityError, match="exactly one"):
        kb.add_workflow_member(
            workflow_db,
            workflow_id="wf_test",
            task_id=second,
            stage_key="acceptance-two",
            stage_role="acceptance",
            required=True,
            actor=_actor(),
            mutation_id="second-acceptance",
            expected_version=1,
        )


def test_reopen_requires_remediation_and_reverification_obligations(workflow_db):
    acceptance = _new_workflow(workflow_db)
    kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=acceptance,
        outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="pass-before-reopen-obligation",
        expected_version=1,
    )
    next_acceptance = kb.create_task(
        workflow_db, title="accept again", assignee="orchestrator", tenant="tenant-a"
    )
    with pytest.raises(kb.WorkflowIntegrityError, match="remediation.*reverification"):
        kb.reopen_workflow(
            workflow_db,
            workflow_id="wf_test",
            designated_acceptance_task_id=next_acceptance,
            members=[{
                "task_id": next_acceptance,
                "stage_key": "acceptance-2",
                "stage_role": "acceptance",
                "required": True,
            }],
            actor=_actor(capabilities=("workflow.admin",)),
            mutation_id="invalid-reopen",
            expected_version=2,
            reason="post-pass defect",
        )


def test_stage_superseded_is_terminal_and_replayable(workflow_db):
    acceptance = _new_workflow(workflow_db)
    result = kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=acceptance,
        outcome="SUPERSEDED",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="supersede-generation",
        expected_version=1,
        summary="replaced by a different workflow",
    )
    assert result["workflow"]["state"] == "SUPERSEDED"
    replay = kb.replay_workflow_events(
        workflow_db,
        workflow_id="wf_test",
        actor=_actor(capabilities=("workflow.read",)),
    )
    assert replay["state"] == "SUPERSEDED"


def test_replay_reconstructs_members_outcomes_and_designation_from_events(workflow_db):
    acceptance = _new_workflow(workflow_db)
    implementation = kb.create_task(
        workflow_db, title="implementation", assignee="builder", tenant="tenant-a"
    )
    kb.add_workflow_member(
        workflow_db,
        workflow_id="wf_test",
        task_id=implementation,
        stage_key="implementation",
        stage_role="implementation",
        required=True,
        actor=_actor(),
        mutation_id="add-for-replay",
        expected_version=1,
    )
    kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=implementation,
        outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="outcome-for-replay",
        expected_version=2,
    )
    workflow_db.execute(
        "UPDATE kanban_workflow_members SET stage_key='corrupted' "
        "WHERE workflow_id='wf_test' AND task_id=?",
        (implementation,),
    )
    replay = kb.replay_workflow_events(
        workflow_db,
        workflow_id="wf_test",
        actor=_actor(capabilities=("workflow.read",)),
    )
    assert replay["generations"]["1"]["designated_acceptance_task_id"] == acceptance
    assert replay["members"][f"1:{implementation}"]["stage_key"] == "implementation"
    assert replay["outcomes"][f"1:{implementation}"]["outcome"] == "PASS"
    assert len(replay["outcome_history"]) == 1
    replayed_outcome = next(iter(replay["outcome_history"].values()))
    assert replayed_outcome["task_id"] == implementation
    assert replayed_outcome["outcome"] == "PASS"

    workflow_db.execute(
        "UPDATE kanban_workflow_outcomes SET outcome='NEEDS_INPUT' "
        "WHERE workflow_id='wf_test' AND task_id=?",
        (implementation,),
    )
    report = kb.workflow_integrity_report(workflow_db, workflow_id="wf_test")
    assert report["ok"] is False
    assert any("outcomes differ" in error for error in report["errors"])


def test_outcome_supersession_replay_rejects_forks_and_preserves_history(workflow_db):
    _new_workflow(workflow_db)
    member = kb.create_task(
        workflow_db, title="implementation", assignee="builder", tenant="tenant-a"
    )
    kb.add_workflow_member(
        workflow_db, workflow_id="wf_test", task_id=member,
        stage_key="implementation", stage_role="implementation", required=True,
        actor=_actor(), mutation_id="add-supersession-member", expected_version=1,
    )
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_test", task_id=member, outcome="NEEDS_INPUT",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="first-outcome", expected_version=2,
    )
    first_id = workflow_db.execute(
        "SELECT id FROM kanban_workflow_outcomes WHERE workflow_id='wf_test' AND task_id=?",
        (member,),
    ).fetchone()["id"]
    kb.record_workflow_outcome(
        workflow_db, workflow_id="wf_test", task_id=member, outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="superseding-outcome", expected_version=3,
        supersedes_outcome_id=first_id,
    )
    replay = kb.replay_workflow_events(
        workflow_db, workflow_id="wf_test",
        actor=_actor(capabilities=("workflow.read",)),
    )
    assert len(replay["outcome_history"]) == 2
    assert replay["outcomes"][f"1:{member}"]["outcome"] == "PASS"
    with pytest.raises(kb.WorkflowConflictError, match="unique effective head"):
        kb.record_workflow_outcome(
            workflow_db, workflow_id="wf_test", task_id=member,
            outcome="NEEDS_INPUT", actor=_actor(capabilities=("workflow.outcome",)),
            mutation_id="forked-outcome", expected_version=4,
            supersedes_outcome_id=first_id,
        )


def test_integrity_failure_is_a_durable_derived_event(workflow_db):
    acceptance = _new_workflow(workflow_db)
    workflow_db.execute(
        "UPDATE kanban_workflow_members SET tenant_snapshot='tenant-b' "
        "WHERE workflow_id='wf_test' AND task_id=?",
        (acceptance,),
    )
    result = kb.record_workflow_outcome(
        workflow_db,
        workflow_id="wf_test",
        task_id=acceptance,
        outcome="PASS",
        actor=_actor(capabilities=("workflow.outcome",)),
        mutation_id="integrity-failure",
        expected_version=1,
    )
    assert result["workflow"]["state"] == "NEEDS_INPUT"
    kinds = [
        row["kind"]
        for row in workflow_db.execute(
            "SELECT kind FROM kanban_workflow_events "
            "WHERE workflow_id='wf_test' AND mutation_id='integrity-failure' "
            "ORDER BY batch_seq"
        )
    ]
    assert kinds == ["outcome_recorded", "integrity_failed", "aggregate_changed"]


def test_separate_process_pass_remediation_race_has_one_cas_winner(workflow_db):
    acceptance = kb.create_task(
        workflow_db, title="accept", assignee="orchestrator", tenant="tenant-a"
    )
    kb.create_workflow(
        workflow_db, workflow_id="wf_race", name="race", tenant="tenant-a",
        designated_acceptance_task_id=acceptance, actor=_actor(),
        mutation_id="create-race",
    )
    db_path = workflow_db.execute("PRAGMA database_list").fetchone()["file"]
    workflow_db.commit()
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_race_outcome_worker,
            args=(db_path, acceptance, outcome, mutation_id, start, results),
        )
        for outcome, mutation_id in (
            ("PASS", "race-pass"),
            ("REMEDIATION_REQUIRED", "race-remediation"),
        )
    ]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    assert sum(result[0] == "ok" for result in observed) == 1
    loser = next(result for result in observed if result[0] == "error")
    assert loser[2] == "WorkflowConflictError"
    workflow = kb.get_workflow(
        workflow_db, "wf_race", actor=_actor(capabilities=("workflow.read",)),
    )
    assert workflow["workflow"]["version"] == 2
    race_events = [
        event for event in workflow["events"]
        if event["mutation_id"] in {"race-pass", "race-remediation"}
    ]
    assert {event["mutation_id"] for event in race_events} in (
        {"race-pass"}, {"race-remediation"}
    )
    assert [event["batch_seq"] for event in race_events] == list(range(len(race_events)))


def test_workflow_queries_use_scope_and_due_indexes(workflow_db):
    _new_workflow(workflow_db)

    def plan(sql, params=()):
        return " ".join(
            row["detail"] for row in workflow_db.execute(
                "EXPLAIN QUERY PLAN " + sql, params
            )
        )

    assert "idx_workflow_members_reduce" in plan(
        "SELECT task_id FROM kanban_workflow_members "
        "WHERE workflow_id=? AND generation=? AND required=1 "
        "AND removed_event_id IS NULL",
        ("wf_test", 1),
    )
    assert "idx_workflow_events_tail" in plan(
        "SELECT id FROM kanban_workflow_events WHERE workflow_id=? AND id>? ORDER BY id",
        ("wf_test", 0),
    )
    assert "idx_workflow_subscriptions_due" in plan(
        "SELECT workflow_id FROM kanban_workflow_subscriptions "
        "WHERE disabled_at IS NULL AND dead_lettered_at IS NULL "
        "AND next_attempt_at IS NULL"
    )
    assert "idx_workflow_members_task" in plan(
        "SELECT workflow_id FROM kanban_workflow_members WHERE task_id=?",
        ("missing",),
    )


def test_integrity_report_detects_orphan_workflow_rows(workflow_db):
    workflow_db.commit()
    workflow_db.execute("PRAGMA foreign_keys=OFF")
    workflow_db.execute(
        "INSERT INTO kanban_workflow_events "
        "(workflow_id,generation,kind,mutation_id,batch_seq,event_role,actor_principal,"
        "expected_version,payload_schema_version,payload,created_at) "
        "VALUES ('wf_orphan',1,'orphan','orphan-mut',0,'canonical','bad',0,1,'{}',1)"
    )
    report = kb.workflow_integrity_report(workflow_db)
    assert report["ok"] is False
    assert any("orphan" in error for error in report["errors"])
