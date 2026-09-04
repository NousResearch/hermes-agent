"""Durable cron execution-ledger behavior."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def test_scheduled_fire_identity_rejects_private_job_id_without_stringifying():
    import cron.executions as executions

    class PrivateJobId:
        def __str__(self):
            raise AssertionError("private job id was stringified")

        def __repr__(self):
            raise AssertionError("private job id was represented")

    with __import__("pytest").raises(ValueError, match="job_id must be a string"):
        executions.scheduled_fire_identity(
            PrivateJobId(), "2026-08-23T20:00:00+00:00"
        )


def test_create_execution_rejects_private_identity_without_stringifying(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)

    class PrivateIdentity:
        def __str__(self):
            raise AssertionError("private identity was stringified")

        def __repr__(self):
            raise AssertionError("private identity was represented")

    with __import__("pytest").raises(ValueError, match="job_id must be a string"):
        executions.create_execution(PrivateIdentity(), source="direct")
    with __import__("pytest").raises(ValueError, match="source must be a string"):
        executions.create_execution("safe-job", source=PrivateIdentity())


def test_list_executions_rejects_private_filters_without_stringifying(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)

    class PrivateFilter:
        def __str__(self):
            raise AssertionError("private filter was stringified")

        def __repr__(self):
            raise AssertionError("private filter was represented")

    with __import__("pytest").raises(ValueError, match="job_id must be a string"):
        executions.list_executions(job_id=PrivateFilter())
    with __import__("pytest").raises(
        ValueError, match="before_claimed_at must be a string"
    ):
        executions.list_executions(before_claimed_at=PrivateFilter())
    with __import__("pytest").raises(ValueError, match="limit must be an integer"):
        executions.list_executions(limit=True)


def test_latest_executions_rejects_private_ids_before_hashing(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    class PrivateJobId:
        def __hash__(self):
            raise AssertionError("private job id was hashed")

        def __eq__(self, _other):
            raise AssertionError("private job id was compared")

        def __str__(self):
            raise AssertionError("private job id was stringified")

        def __repr__(self):
            raise AssertionError("private job id was represented")

    with __import__("pytest").raises(ValueError, match="job_id must be a string"):
        executions.latest_executions([PrivateJobId()])
    with __import__("pytest").raises(ValueError, match="job_ids must be a list"):
        executions.latest_executions(("safe",))


def test_execution_transitions_are_durable(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    claimed = executions.create_execution("job-1", source="builtin")
    assert claimed["status"] == "claimed"
    assert claimed["claimed_at"]
    assert claimed["started_at"] is None
    assert claimed["finished_at"] is None

    running = executions.mark_execution_running(claimed["id"])
    assert running["status"] == "running"
    assert running["started_at"]

    completed = executions.finish_execution(claimed["id"], success=True)
    assert completed["status"] == "completed"
    assert completed["finished_at"]
    assert completed["error"] is None

    persisted = executions.list_executions(job_id="job-1")
    assert persisted == [completed]


def test_execution_claim_persists_fire_identity_before_dispatch(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    manual = executions.create_execution("manual-job", source="manual")
    assert manual["fire_identity"] == manual["id"]

    scheduled = executions.create_execution(
        "scheduled-job", source="builtin", fire_identity="scheduled-fire-2026-08-22T19:00Z",
    )
    assert scheduled["fire_identity"] == "scheduled-fire-2026-08-22T19:00Z"


def test_execution_fire_identity_binds_once_before_receipt_plan(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("bind-job", source="external")

    bound = executions.bind_execution_fire_identity(execution["id"], "claim-fire")
    assert bound["fire_identity"] == "claim-fire"
    with __import__("pytest").raises(ValueError, match="already bound"):
        executions.bind_execution_fire_identity(execution["id"], "different-fire")


def test_new_execution_without_transport_plan_is_not_attempted_unconfirmed(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    local = executions.create_execution("local-only", source="builtin")
    executions.finish_execution(local["id"], success=True)

    assert executions.receipt_summary(local["id"]) == {
        "delivered": 0, "failed": 0, "unknown": 0, "targets_delivered": 0,
    }


def test_receipt_plan_preregisters_unknown_attempt_without_persisting_content(monkeypatch, tmp_path):
    """Dispatch may begin only after every content-free receipt row is durable."""
    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("job-receipt", source="builtin")

    attempts = executions.preregister_receipt_plan(
        execution["id"],
        fire_identity="fire-1",
        components=[
            {
                "target": {"platform": "matrix", "chat_id": "!room:example.org"},
                "component": "text",
                "ordinal": 0,
                "content": "do not persist this report body",
            },
            {
                "target": {"platform": "matrix", "chat_id": "!room:example.org"},
                "component": "media",
                "ordinal": 1,
                "content": "logical media item identity",
            },
        ],
    )

    assert len(attempts) == 2
    assert {row["outcome"] for row in attempts} == {"unknown"}
    summary = executions.receipt_summary(execution["id"])
    assert summary == {"delivered": 0, "failed": 0, "unknown": 2, "targets_delivered": 0}

    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        dumped = "\n".join(str(row) for row in conn.execute(
            "SELECT content_hash FROM delivery_components"
        ))
    assert "do not persist this report body" not in dumped
    assert "logical media item identity" not in dumped


def test_receipt_plan_rejects_private_non_string_content_without_stringifying(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("hostile-content", source="direct")

    class PrivateContent:
        def __str__(self):
            raise AssertionError("private content was stringified")

        def __repr__(self):
            raise AssertionError("private content was represented")

    with __import__("pytest").raises(ValueError, match="content must be a string"):
        executions.preregister_receipt_plan(
            execution["id"],
            fire_identity=execution["fire_identity"],
            components=[{
                "target": {"platform": "telegram", "chat_id": "123"},
                "component": "text", "ordinal": 0, "content": PrivateContent(),
            }],
        )
    assert executions.receipt_summary(execution["id"])["unknown"] == 0


def test_receipt_plan_rejects_container_and_text_subclasses_before_magic_methods(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("hostile-plan", source="direct")

    class HostileList(list):
        def __iter__(self):
            raise AssertionError("hostile list was iterated")

        def __len__(self):
            raise AssertionError("hostile list length was evaluated")

    class HostileDict(dict):
        def get(self, *_args, **_kwargs):
            raise AssertionError("hostile mapping get was called")

        def __getitem__(self, _key):
            raise AssertionError("hostile mapping item access was called")

    class HostileText(str):
        def __bool__(self):
            raise AssertionError("hostile text truthiness was evaluated")

        def __len__(self):
            raise AssertionError("hostile text length was evaluated")

        def __eq__(self, _other):
            raise AssertionError("hostile text was compared")

        def __hash__(self):
            raise AssertionError("hostile text was hashed")

    valid = {
        "target": {"platform": "telegram", "chat_id": "123"},
        "component": "text", "ordinal": 0, "content": "safe",
    }
    invalid_plans = (
        HostileList([valid]),
        [HostileDict(valid)],
        [{**valid, "target": HostileDict(valid["target"])}],
        [{**valid, "content": HostileText("safe")}],
    )
    for invalid in invalid_plans:
        with __import__("pytest").raises(ValueError):
            executions.preregister_receipt_plan(
                execution["id"],
                fire_identity=execution["fire_identity"],
                components=invalid,
            )

    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        assert conn.execute("SELECT COUNT(*) FROM delivery_targets").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM delivery_components").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM delivery_attempts").fetchone()[0] == 0


def test_receipt_transition_only_records_typed_ack_and_target_is_all_components(monkeypatch, tmp_path):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("job-ack", source="builtin")
    attempts = executions.preregister_receipt_plan(
        execution["id"],
        fire_identity="fire-2",
        components=[
            {
                "target": {"platform": "telegram", "chat_id": "123", "thread_id": "8"},
                "component": "text",
                "ordinal": 0,
                "content": "text component",
            },
            {
                "target": {"platform": "telegram", "chat_id": "123", "thread_id": "8"},
                "component": "media",
                "ordinal": 1,
                "content": "media component",
            },
        ],
    )
    receipt = TransportReceipt(
        outcome="delivered",
        provider_message_id="42",
        requested_target=TransportTarget("telegram", "123", "8"),
        actual_target=TransportTarget("telegram", "123", None),
    )

    assert executions.record_transport_receipt(attempts[0]["id"], receipt) is True
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 1,
        "failed": 0,
        "unknown": 1,
        "targets_delivered": 0,
    }
    assert executions.record_transport_receipt(attempts[0]["id"], receipt) is False


def test_requested_target_is_not_delivered_when_ack_landed_at_fallback_target(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("fallback-target", source="builtin")
    attempts = executions.preregister_receipt_plan(
        execution["id"],
        fire_identity="fallback-fire",
        components=[{
            "target": {"platform": "telegram", "chat_id": "123", "thread_id": "8"},
            "component": "text",
            "ordinal": 0,
            "content": "fallback-bound text",
        }],
    )
    receipt = TransportReceipt(
        outcome="delivered",
        provider_message_id="fallback-provider-id",
        requested_target=TransportTarget("telegram", "123", "8"),
        actual_target=TransportTarget("telegram", "123", None),
        component="text",
        ordinal=0,
    )

    assert executions.record_transport_receipt(attempts[0]["id"], receipt) is True
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 1,
        "failed": 0,
        "unknown": 0,
        "targets_delivered": 0,
    }


def test_requested_target_is_delivered_when_every_ack_matches_target(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("exact-target", source="builtin")
    attempts = executions.preregister_receipt_plan(
        execution["id"],
        fire_identity="exact-fire",
        components=[{
            "target": {"platform": "telegram", "chat_id": "123", "thread_id": "8"},
            "component": "text",
            "ordinal": 0,
            "content": "exact-target text",
        }],
    )
    target = TransportTarget("telegram", "123", "8")
    receipt = TransportReceipt(
        outcome="delivered",
        provider_message_id="exact-provider-id",
        requested_target=target,
        actual_target=target,
        component="text",
        ordinal=0,
    )

    assert executions.record_transport_receipt(attempts[0]["id"], receipt) is True
    assert executions.receipt_summary(execution["id"])["targets_delivered"] == 1


def test_unknown_observation_requires_exact_binding_without_upgrading_outcome(
    monkeypatch, tmp_path
):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("observe-unknown", source="direct")
    attempts = executions.preregister_receipt_plan(
        execution["id"],
        fire_identity=execution["fire_identity"],
        components=[{
            "target": {"platform": "bot-chat", "chat_id": "research"},
            "component": "text", "ordinal": 0, "content": "planned query",
        }],
    )
    wrong = TransportReceipt(
        outcome="unknown",
        requested_target=TransportTarget("bot-chat", "other-profile"),
    )
    with __import__("pytest").raises(ValueError, match="does not match"):
        executions.observe_transport_unknown(attempts[0]["id"], wrong)

    with executions._transaction() as conn:
        assert conn.execute(
            "SELECT observed_at FROM delivery_attempts WHERE id=?",
            (attempts[0]["id"],),
        ).fetchone()["observed_at"] is None

    exact = TransportReceipt(
        outcome="unknown",
        requested_target=TransportTarget("bot-chat", "research"),
    )
    assert executions.observe_transport_unknown(attempts[0]["id"], exact) is True
    assert executions.receipt_summary(execution["id"]) == {
        "delivered": 0, "failed": 0, "unknown": 1, "targets_delivered": 0,
    }


def test_receipt_plan_is_idempotent_only_for_same_fire_and_receipt_binds_target(monkeypatch, tmp_path):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("job-plan", source="builtin")
    plan = [{
        "target": {"platform": "telegram", "chat_id": "123"},
        "component": "text", "ordinal": 0, "content": "secret report body",
    }]
    first = executions.preregister_receipt_plan(execution["id"], fire_identity="fire-a", components=plan)
    assert executions.preregister_receipt_plan(
        execution["id"], fire_identity="fire-a", components=plan
    ) == first
    with __import__("pytest").raises(ValueError, match="conflicting"):
        executions.preregister_receipt_plan(execution["id"], fire_identity="fire-b", components=plan)

    wrong_target = TransportReceipt(
        outcome="delivered", provider_message_id="1",
        requested_target=TransportTarget("telegram", "other"),
        actual_target=TransportTarget("telegram", "other"),
    )
    with __import__("pytest").raises(ValueError, match="requested_target"):
        executions.record_transport_receipt(first[0]["id"], wrong_target)

    independent = executions.create_execution("job-plan", source="builtin")
    assert executions.preregister_receipt_plan(
        independent["id"], fire_identity="fire-independent", components=plan,
    )
    conflicting = executions.create_execution("job-plan", source="recovery")
    changed_plan = [dict(plan[0], content="different content")]
    with __import__("pytest").raises(ValueError, match="conflicting.*fire_identity"):
        executions.preregister_receipt_plan(
            conflicting["id"], fire_identity="fire-a", components=changed_plan,
        )


def test_mutated_duck_typed_actual_target_cannot_mark_attempt_delivered(monkeypatch, tmp_path):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("typed-boundary", source="builtin")
    attempts = executions.preregister_receipt_plan(
        execution["id"], fire_identity="typed-fire", components=[{
            "target": {"platform": "telegram", "chat_id": "123"},
            "component": "text", "ordinal": 0, "content": "bounded",
        }],
    )
    target = TransportTarget("telegram", "123")
    receipt = TransportReceipt(
        outcome="delivered", provider_message_id="provider-1",
        requested_target=target, actual_target=target,
    )
    object.__setattr__(receipt, "actual_target", object())
    with __import__("pytest").raises(TypeError, match="TransportTarget"):
        executions.record_transport_receipt(attempts[0]["id"], receipt)
    assert executions.receipt_summary(execution["id"])["unknown"] == 1


def test_receipt_persistence_rejects_subclasses_before_magic_methods(
    monkeypatch, tmp_path
):
    from datetime import datetime, timezone
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    target = TransportTarget("telegram", "123")

    class HostileReceipt(TransportReceipt):
        def __getattribute__(self, name):
            try:
                armed = object.__getattribute__(self, "_armed")
            except AttributeError:
                armed = False
            if armed and name not in {"__class__", "_armed"}:
                raise AssertionError("hostile receipt attribute was read")
            return super().__getattribute__(name)

    subclass_execution = executions.create_execution("receipt-subclass", source="direct")
    subclass_attempt = executions.preregister_receipt_plan(
        subclass_execution["id"], fire_identity="receipt-subclass-fire",
        components=[{
            "target": {"platform": "telegram", "chat_id": "123"},
            "component": "text", "ordinal": 0, "content": "bounded",
        }],
    )[0]
    hostile_receipt = HostileReceipt(outcome="unknown", requested_target=target)
    object.__setattr__(hostile_receipt, "_armed", True)
    with __import__("pytest").raises(ValueError, match="TransportReceipt"):
        executions.observe_transport_unknown(subclass_attempt["id"], hostile_receipt)

    class HostileText(str):
        def __hash__(self):
            raise AssertionError("hostile outcome was hashed")

        def __eq__(self, _other):
            raise AssertionError("hostile outcome was compared")

    record_execution = executions.create_execution("mutated-outcome", source="direct")
    record_attempt = executions.preregister_receipt_plan(
        record_execution["id"], fire_identity="mutated-outcome-fire",
        components=[{
            "target": {"platform": "telegram", "chat_id": "123"},
            "component": "text", "ordinal": 0, "content": "bounded",
        }],
    )[0]
    delivered = TransportReceipt(
        outcome="delivered", requested_target=target, actual_target=target,
        provider_message_id="provider-id",
    )
    object.__setattr__(delivered, "outcome", HostileText("delivered"))
    with __import__("pytest").raises(ValueError, match="outcome"):
        executions.record_transport_receipt(record_attempt["id"], delivered)

    class HostileDateTime(datetime):
        def astimezone(self, *_args, **_kwargs):
            raise AssertionError("hostile datetime conversion was called")

        def isoformat(self, *_args, **_kwargs):
            raise AssertionError("hostile datetime formatting was called")

    observe_execution = executions.create_execution("mutated-observed", source="direct")
    observe_attempt = executions.preregister_receipt_plan(
        observe_execution["id"], fire_identity="mutated-observed-fire",
        components=[{
            "target": {"platform": "telegram", "chat_id": "123"},
            "component": "text", "ordinal": 0, "content": "bounded",
        }],
    )[0]
    unknown = TransportReceipt(outcome="unknown", requested_target=target)
    object.__setattr__(unknown, "observed_at", HostileDateTime.now(timezone.utc))
    with __import__("pytest").raises(ValueError, match="timezone-aware"):
        executions.observe_transport_unknown(observe_attempt["id"], unknown)

    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        rows = conn.execute(
            "SELECT outcome, observed_at FROM delivery_attempts ORDER BY id"
        ).fetchall()
    assert rows and all(row == ("unknown", None) for row in rows)


def test_mutated_typed_target_fields_cannot_mark_attempt_delivered(monkeypatch, tmp_path):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    for target_field in ("requested_target", "actual_target"):
        execution = executions.create_execution(
            f"mutated-{target_field}", source="builtin",
        )
        attempts = executions.preregister_receipt_plan(
            execution["id"], fire_identity=f"fire-{target_field}", components=[{
                "target": {"platform": "telegram", "chat_id": "123"},
                "component": "text", "ordinal": 0, "content": "bounded",
            }],
        )
        receipt = TransportReceipt(
            outcome="delivered",
            provider_message_id=f"provider-{target_field}",
            requested_target=TransportTarget("telegram", "123"),
            actual_target=TransportTarget("telegram", "123"),
        )
        object.__setattr__(getattr(receipt, target_field), "thread_id", "")

        with __import__("pytest").raises(ValueError, match="thread_id"):
            executions.record_transport_receipt(attempts[0]["id"], receipt)
        assert executions.receipt_summary(execution["id"])["unknown"] == 1


def test_mutated_failed_receipt_cannot_persist_provider_evidence_or_unknown_kind(
    monkeypatch, tmp_path,
):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    target = TransportTarget("telegram", "123")
    mutations = (
        ("provider_message_id", "provider-id", "provider"),
        ("actual_target", target, "actual_target"),
        ("failure_kind", "unbounded_kind", "failure_kind"),
    )

    for ordinal, (field, value, match) in enumerate(mutations):
        execution = executions.create_execution(
            f"failed-boundary-{ordinal}", source="builtin",
        )
        attempts = executions.preregister_receipt_plan(
            execution["id"], fire_identity=f"failed-fire-{ordinal}", components=[{
                "target": {"platform": "telegram", "chat_id": "123"},
                "component": "text", "ordinal": 0, "content": "bounded",
            }],
        )
        receipt = TransportReceipt(
            outcome="failed",
            requested_target=target,
            component="text",
            ordinal=0,
            failure_kind="pre_dispatch",
        )
        object.__setattr__(receipt, field, value)

        with __import__("pytest").raises(ValueError, match=match):
            executions.record_transport_receipt(attempts[0]["id"], receipt)
        assert executions.receipt_summary(execution["id"])["unknown"] == 1


def test_mutated_delivered_receipt_cannot_persist_failure_kind(monkeypatch, tmp_path):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("delivered-boundary", source="builtin")
    attempts = executions.preregister_receipt_plan(
        execution["id"], fire_identity="delivered-fire", components=[{
            "target": {"platform": "telegram", "chat_id": "123"},
            "component": "text", "ordinal": 0, "content": "bounded",
        }],
    )
    target = TransportTarget("telegram", "123")
    receipt = TransportReceipt(
        outcome="delivered",
        provider_message_id="provider-id",
        requested_target=target,
        actual_target=target,
        component="text",
        ordinal=0,
    )
    object.__setattr__(receipt, "failure_kind", "pre_dispatch")

    with __import__("pytest").raises(ValueError, match="failure_kind"):
        executions.record_transport_receipt(attempts[0]["id"], receipt)
    assert executions.receipt_summary(execution["id"])["unknown"] == 1


def test_receipt_plan_and_persistence_reject_boolean_ordinals(monkeypatch, tmp_path):
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    rejected = executions.create_execution("bool-plan", source="builtin")
    with __import__("pytest").raises(ValueError, match="ordinal"):
        executions.preregister_receipt_plan(
            rejected["id"], fire_identity="bool-plan-fire", components=[{
                "target": {"platform": "telegram", "chat_id": "123"},
                "component": "text", "ordinal": True, "content": "bounded",
            }],
        )

    execution = executions.create_execution("bool-receipt", source="builtin")
    attempts = executions.preregister_receipt_plan(
        execution["id"], fire_identity="bool-receipt-fire", components=[{
            "target": {"platform": "telegram", "chat_id": "123"},
            "component": "text", "ordinal": 0, "content": "bounded",
        }],
    )
    target = TransportTarget("telegram", "123")
    receipt = TransportReceipt(
        outcome="delivered", provider_message_id="provider-1",
        requested_target=target, actual_target=target, ordinal=0,
    )
    object.__setattr__(receipt, "ordinal", False)
    with __import__("pytest").raises(ValueError, match="ordinal"):
        executions.record_transport_receipt(attempts[0]["id"], receipt)
    assert executions.receipt_summary(execution["id"])["unknown"] == 1


def test_partial_confirmation_blocks_a_second_plan_for_the_same_fire(monkeypatch, tmp_path):
    """Unknown never authorizes replay after any component may have dispatched."""
    from gateway.platforms.base import TransportReceipt, TransportTarget

    executions = _point_ledger(monkeypatch, tmp_path)
    first_execution = executions.create_execution("job-partial", source="builtin")
    plan = [
        {
            "target": {"platform": "telegram", "chat_id": "123"},
            "component": "text", "ordinal": 0, "content": "report",
        },
        {
            "target": {"platform": "telegram", "chat_id": "123"},
            "component": "media", "ordinal": 0, "content": "/private/report.pdf",
        },
    ]
    attempts = executions.preregister_receipt_plan(
        first_execution["id"], fire_identity="fire-partial", components=plan,
    )
    target = TransportTarget("telegram", "123")
    assert executions.record_transport_receipt(
        attempts[0]["id"],
        TransportReceipt(
            outcome="delivered", provider_message_id="provider-1",
            requested_target=target, actual_target=target,
            component="text", ordinal=0,
        ),
    ) is True

    recovery = executions.create_execution("job-partial", source="recovery")
    with __import__("pytest").raises(ValueError, match="already attempted"):
        executions.preregister_receipt_plan(
            recovery["id"], fire_identity="fire-partial", components=plan,
        )


def test_receipt_schema_upgrade_is_singleton_and_idempotent(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    legacy = executions.create_execution("legacy-receipt", source="builtin")
    executions.finish_execution(legacy["id"], success=True)

    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        conn.execute("UPDATE executions SET receipt_state=NULL WHERE id=?", (legacy["id"],))
        conn.execute("DROP TABLE receipt_schema")
        conn.execute("CREATE TABLE receipt_schema(version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO receipt_schema(version) VALUES (1), (2)")

    assert executions.receipt_summary(legacy["id"])["unknown"] == 1
    assert executions.receipt_summary(legacy["id"])["unknown"] == 1
    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(receipt_schema)")]
        rows = list(conn.execute("SELECT singleton, version FROM receipt_schema"))
        synthetic = conn.execute("SELECT COUNT(*) FROM delivery_attempts").fetchone()[0]
    assert columns == ["singleton", "version"]
    assert rows == [(1, executions._RECEIPT_SCHEMA_VERSION)]
    assert synthetic == 0


def test_execution_and_receipt_database_never_persists_payloads_paths_or_raw_errors(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("privacy-job", source="builtin")
    executions.preregister_receipt_plan(
        execution["id"], fire_identity="privacy-fire",
        components=[
            {
                "target": {"platform": "matrix", "chat_id": "!room:example.org"},
                "component": "text", "ordinal": 0,
                "content": "PRIVATE_REPORT_BODY_SENTINEL",
            },
            {
                "target": {"platform": "matrix", "chat_id": "!room:example.org"},
                "component": "media", "ordinal": 0,
                "content": "/private/media/PAYSLIP_SENTINEL.pdf",
            },
        ],
    )
    failed = executions.finish_execution(
        execution["id"], success=False,
        error="RAW_PROVIDER_EXCEPTION_SENTINEL user@example.org",
    )
    assert failed["error"] is None
    assert failed["error_kind"] == "execution_failed"

    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        dump = "\n".join(conn.iterdump())
    assert "PRIVATE_REPORT_BODY_SENTINEL" not in dump
    assert "PAYSLIP_SENTINEL" not in dump
    assert "RAW_PROVIDER_EXCEPTION_SENTINEL" not in dump
    assert "user@example.org" not in dump

def test_execution_can_be_loaded_by_exact_attempt_id(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    first = executions.create_execution("same-job", source="builtin")
    second = executions.create_execution("same-job", source="builtin")

    assert executions.get_execution(first["id"]) == first
    assert executions.get_execution(second["id"]) == second
    assert executions.get_execution("missing") is None


def test_fresh_external_handoff_is_not_recovered_before_worker_adopts(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("handoff-job", source="builtin")
    assert executions.mark_execution_handoff_pending(record["id"]) is not None

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)

    assert executions.recover_interrupted_executions() == 0
    assert executions.get_execution(record["id"])["status"] == "claimed"
    adopted = executions.adopt_claimed_execution(record["id"])
    assert adopted["status"] == "running"
    assert adopted["handoff_pending"] == 0


def test_stale_external_handoff_is_recovered_unknown(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("handoff-job", source="builtin")
    pending = executions.mark_execution_handoff_pending(record["id"])

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    monkeypatch.setattr(
        executions.time,
        "time",
        lambda: pending["handoff_started_at"]
        + executions.HANDOFF_ADOPTION_GRACE_SECONDS
        + 1,
    )

    assert executions.recover_interrupted_executions() == 1
    recovered = executions.get_execution(record["id"])
    assert recovered["status"] == "unknown"
    assert recovered["handoff_pending"] == 0


def test_recovery_does_not_overwrite_concurrent_worker_adoption(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("adoption-race", source="builtin")
    pending = executions.mark_execution_handoff_pending(record["id"])
    assert pending is not None
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-scheduler")
    monkeypatch.setattr(
        executions.time,
        "time",
        lambda: pending["handoff_started_at"]
        + executions.HANDOFF_ADOPTION_GRACE_SECONDS
        + 1,
    )

    def adopt_while_liveness_is_checked(_pid, _started_at):
        monkeypatch.setattr(executions, "_PROCESS_ID", "external-worker")
        monkeypatch.setattr(executions.os, "getpid", lambda: 4242)
        monkeypatch.setattr(executions, "_process_start_time", lambda _pid: 9876)
        assert executions.adopt_claimed_execution(record["id"]) is not None
        return False

    monkeypatch.setattr(executions, "_owner_is_live", adopt_while_liveness_is_checked)

    assert executions.recover_interrupted_executions() == 0
    current = executions.get_execution(record["id"])
    assert current is not None
    assert current["status"] == "running"
    assert current["process_id"] == "external-worker"
    assert current["pid"] == 4242


def test_foreign_process_cannot_start_or_finish_execution(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("owner-fence", source="builtin")
    original_process_id = executions._PROCESS_ID
    original_pid = record["pid"]

    monkeypatch.setattr(executions, "_PROCESS_ID", "foreign-process")
    monkeypatch.setattr(executions.os, "getpid", lambda: original_pid + 1)
    assert executions.mark_execution_running(record["id"]) is None
    assert executions.finish_execution(record["id"], success=True) is None

    monkeypatch.setattr(executions, "_PROCESS_ID", original_process_id)
    monkeypatch.setattr(executions.os, "getpid", lambda: original_pid)
    assert executions.mark_execution_running(record["id"]) is not None
    assert executions.finish_execution(record["id"], success=True) is not None


def test_execution_ledger_follows_the_current_profile_home(monkeypatch, tmp_path):
    import cron.executions as executions

    current_home = {"path": tmp_path / "default"}
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(executions, "get_hermes_home", lambda: current_home["path"])

    default_row = executions.create_execution("default-job", source="builtin")
    current_home["path"] = tmp_path / "worker"
    worker_row = executions.create_execution("worker-job", source="builtin")

    assert executions.list_executions() == [worker_row]
    current_home["path"] = tmp_path / "default"
    assert executions.list_executions() == [default_row]
    assert (tmp_path / "default" / "cron" / "executions.db").is_file()
    assert (tmp_path / "worker" / "cron" / "executions.db").is_file()


def test_terminal_execution_cannot_be_rewritten(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("immutable", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)

    assert executions.finish_execution(
        record["id"], success=False, error="late writer"
    ) is None
    assert executions.latest_execution("immutable")["status"] == "completed"


def test_retention_bounds_terminal_history_but_preserves_inflight(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 3)
    inflight = executions.create_execution("live", source="builtin")
    executions.mark_execution_running(inflight["id"])
    for index in range(8):
        row = executions.create_execution(f"done-{index}", source="builtin")
        executions.finish_execution(row["id"], success=True)

    records = executions.list_executions(limit=100)
    assert len([row for row in records if row["status"] == "completed"]) == 3
    assert executions.latest_execution("live")["status"] == "running"


def test_receipt_preregistration_is_concurrently_idempotent(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    execution = executions.create_execution("concurrent-plan", source="builtin")
    plan = [{
        "target": {"platform": "matrix", "chat_id": "!room:example.org"},
        "component": "text", "ordinal": 0, "content": "same content",
    }]

    def register():
        return executions.preregister_receipt_plan(
            execution["id"], fire_identity="same-fire", components=plan,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: register(), range(2)))
    assert results[0] == results[1]


def test_execution_retention_cascades_receipt_rows(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 1)
    old = executions.create_execution("old-receipt", source="builtin")
    executions.preregister_receipt_plan(
        old["id"], fire_identity="old-fire",
        components=[{
            "target": {"platform": "telegram", "chat_id": "123"},
            "component": "text", "ordinal": 0, "content": "old",
        }],
    )
    executions.finish_execution(old["id"], success=True)
    new = executions.create_execution("new-receipt", source="builtin")
    executions.finish_execution(new["id"], success=True)

    with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM delivery_targets WHERE execution_id=?", (old["id"],)
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM delivery_components").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM delivery_attempts").fetchone()[0] == 0

def test_recently_finished_long_running_execution_survives_retention(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 1)
    long_running = executions.create_execution("long-running", source="builtin")
    assert executions.mark_execution_running(long_running["id"]) is not None
    newer = executions.create_execution("newer", source="builtin")
    assert executions.finish_execution(newer["id"], success=True) is not None

    finished = executions.finish_execution(long_running["id"], success=True)

    assert finished is not None
    assert finished["status"] == "completed"
    assert executions.get_execution(long_running["id"])["status"] == "completed"
    assert executions.get_execution(newer["id"]) is None


def test_corrupt_store_fails_closed_without_overwrite(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True)
    executions.EXECUTIONS_FILE.write_bytes(b"not a sqlite database")

    with __import__("pytest").raises(sqlite3.DatabaseError):
        executions.create_execution("new", source="builtin")
    assert executions.EXECUTIONS_FILE.read_bytes() == b"not a sqlite database"


def test_cron_runs_cli_prints_execution_history(monkeypatch, tmp_path, capsys):
    executions = _point_ledger(monkeypatch, tmp_path)
    row = executions.create_execution("cli-job", source="builtin")
    executions.finish_execution(row["id"], success=False, error="boom")
    from hermes_cli.cron import cron_runs

    cron_runs("cli-job", limit=10)

    output = capsys.readouterr().out
    assert row["id"] in output
    assert "failed" in output
    assert "Failure kind: execution_failed" in output
    assert "boom" not in output
    assert "Receipt: delivered=0 failed=0 unknown=0" in output


def test_quick_backup_includes_execution_ledger():
    from hermes_cli.backup import _QUICK_STATE_FILES

    assert "cron/executions.db" in _QUICK_STATE_FILES


def test_failed_execution_keeps_only_bounded_error_category(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    record = executions.create_execution("job-2", source="external")
    failed = executions.finish_execution(record["id"], success=False, error="provider exploded")

    assert failed["status"] == "failed"
    assert failed["error"] is None
    assert failed["error_kind"] == "execution_failed"


def test_recovery_does_not_mark_live_process_execution_unknown(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("still-live", source="builtin")
    executions.mark_execution_running(record["id"])

    assert executions.recover_interrupted_executions() == 0
    assert executions.latest_execution("still-live")["status"] == "running"


def test_restart_marks_interrupted_execution_unknown_without_requeue(tmp_path):
    """Real temp-HERMES_HOME subprocess restart: in-flight is audit-only unknown."""
    home = tmp_path / "home"
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)

    create = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cron.executions import create_execution, mark_execution_running; "
            "r=create_execution('restart-job', source='builtin'); "
            "mark_execution_running(r['id']); print(r['id'])",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    execution_id = create.stdout.strip()

    recover = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from cron.executions import recover_interrupted_executions, list_executions; "
            "print(recover_interrupted_executions()); "
            "print(json.dumps(list_executions(job_id='restart-job'))) ",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = recover.stdout.strip().splitlines()
    assert lines[0] == "1"
    records = json.loads(lines[1])
    assert len(records) == 1
    assert records[0]["id"] == execution_id
    assert records[0]["status"] == "unknown"
    assert records[0]["finished_at"]
    assert records[0]["error"] is None
    assert records[0]["error_kind"] == "interrupted"
    # Recovery only classifies the old attempt. It must not manufacture a new
    # claimed record (which would imply an automatic retry).
    assert [r["status"] for r in records] == ["unknown"]


def test_generic_submit_failure_finishes_attempt_and_releases_guard(monkeypatch):
    import cron.scheduler as scheduler

    class BrokenPool:
        def submit(self, _callable):
            raise ValueError("executor rejected")

    finished = []
    monkeypatch.setattr(
        scheduler, "create_execution",
        lambda *_args, **_kwargs: {"id": "exec-submit-fail"},
    )
    monkeypatch.setattr(
        scheduler, "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [{"id": "submit-fail"}])
    monkeypatch.setattr(scheduler, "claim_job_for_fire", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "_get_parallel_pool", lambda _workers: BrokenPool())

    assert scheduler.tick(verbose=False, sync=False) == 0
    assert finished == [
        ("exec-submit-fail", {
            "success": False,
            "error": "Executor dispatch failed: executor rejected",
        })
    ]
    assert "submit-fail" not in scheduler.get_running_job_ids()


def test_run_one_job_records_running_then_terminal(monkeypatch):
    import cron.scheduler as scheduler

    events = []
    run_execution_ids = []
    monkeypatch.setattr(
        scheduler,
        "mark_execution_running",
        lambda execution_id: events.append(("running", execution_id)) or {},
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: events.append(("finish", execution_id, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)

    def fake_run_job(job, *, defer_agent_teardown=None, execution_id=None, **_kw):
        run_execution_ids.append(execution_id)
        return True, "output", "response", None

    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    assert scheduler.run_one_job({"id": "job-3", "execution_id": "exec-3"}) is True
    assert run_execution_ids == ["exec-3"]
    assert events[0] == ("running", "exec-3")
    assert events[-1][0:2] == ("finish", "exec-3")
    assert events[-1][2]["success"] is True


def test_direct_run_passes_one_execution_and_fire_identity_to_delivery(monkeypatch):
    """Manual/direct fires must not lose the durable receipt identity."""
    import cron.scheduler as scheduler
    from cron.executions import scheduled_fire_identity

    captured = {}
    fire_at = "2026-08-22T19:00:00+00:00"

    def create(job_id, *, source, **kwargs):
        captured["created"] = (job_id, source, kwargs.get("fire_identity"))
        return {"id": "exec-direct", "fire_identity": kwargs.get("fire_identity")}

    monkeypatch.setattr(scheduler, "create_execution", create)
    def bind(execution_id, fire_identity):
        captured["bound"] = (execution_id, fire_identity)
        return {"id": execution_id, "fire_identity": fire_identity}

    monkeypatch.setattr(scheduler, "bind_execution_fire_identity", bind)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a, **_kw: {})
    monkeypatch.setattr(scheduler, "finish_execution", lambda *_a, **_kw: None)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        scheduler,
        "fire_claim_fence",
        lambda *_a, **_kw: __import__("contextlib").nullcontext(True),
    )
    monkeypatch.setattr(
        scheduler, "run_job",
        lambda _job, *, defer_agent_teardown=None, **_kw: (True, "output", "response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_a, **_kw: None)

    def deliver(_job, _content, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)

    assert scheduler.run_one_job({
        "id": "manual-identity",
        "deliver": "local",
        "fire_claim": {"by": "manual-owner", "at": fire_at, "fire_at": fire_at},
    }) is True
    expected = scheduled_fire_identity("manual-identity", fire_at)
    assert captured == {
        "created": ("manual-identity", "direct", None),
        "bound": ("exec-direct", expected),
        "adapters": None,
        "loop": None,
        "execution_id": "exec-direct",
        "fire_identity": expected,
        "for_failure": False,
    }


def test_existing_execution_binding_is_verified_from_ledger_before_running(
    monkeypatch, tmp_path
):
    """A job snapshot cannot prove that its persisted execution is bound."""
    import contextlib
    import cron.scheduler as scheduler

    executions = _point_ledger(monkeypatch, tmp_path)
    fire_at = "2026-08-22T19:00:00+00:00"
    expected = executions.scheduled_fire_identity("existing-direct", fire_at)
    execution = executions.create_execution("existing-direct", source="direct")
    assert execution["fire_identity"] == execution["id"]

    def mark_running(execution_id):
        persisted = executions.list_executions(job_id="existing-direct")
        assert persisted[0]["id"] == execution_id
        assert persisted[0]["fire_identity"] == expected

    monkeypatch.setattr(scheduler, "mark_execution_running", mark_running)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *_a, **_kw: None)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        scheduler,
        "fire_claim_fence",
        lambda *_a, **_kw: contextlib.nullcontext(True),
    )
    monkeypatch.setattr(
        scheduler, "run_job",
        lambda _job, *, defer_agent_teardown=None, **_kw: (
            True, "output", "response", None,
        ),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: None)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_a, **_kw: None)

    assert scheduler.run_one_job({
        "id": "existing-direct",
        "execution_id": execution["id"],
        # Adversarial stale/forged snapshot: it claims the expected identity,
        # while the authoritative ledger is still default-bound to execution_id.
        "fire_identity": expected,
        "fire_claim": {
            "by": "manual-owner", "at": fire_at, "fire_at": fire_at,
        },
    }) is True


def test_builtin_tick_binds_due_timestamp_through_claim_to_execution(monkeypatch):
    import cron.scheduler as scheduler
    from cron.executions import scheduled_fire_identity

    due_at = "2026-08-22T19:00:00+00:00"
    acquired_at = "2026-08-22T20:00:00+00:00"
    due_job = {
        "id": "builtin-fire-identity", "name": "identity",
        "schedule": {"kind": "interval", "seconds": 3600},
        "next_run_at": due_at, "enabled": True,
    }
    captured = {}

    def create(job_id, *, source, **kwargs):
        captured["created"] = (job_id, source, kwargs.get("fire_identity"))
        return {"id": "exec-builtin", "fire_identity": kwargs.get("fire_identity")}

    def bind(execution_id, fire_identity):
        captured["bound"] = (execution_id, fire_identity)
        return {"id": execution_id, "fire_identity": fire_identity}

    def run(claimed_job, **_kwargs):
        captured["run"] = claimed_job
        return True

    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [due_job])
    monkeypatch.setattr(scheduler, "advance_next_runs", lambda _ids: None)
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    monkeypatch.setattr(scheduler, "create_execution", create)
    monkeypatch.setattr(scheduler, "bind_execution_fire_identity", bind, raising=False)
    monkeypatch.setattr(
        scheduler, "claim_job_for_fire",
        lambda _jid, **_kwargs: dict(
            due_job,
            next_run_at="2026-08-22T21:00:00+00:00",
            fire_claim={"by": "builtin-owner", "at": acquired_at, "fire_at": acquired_at},
        ),
    )
    monkeypatch.setattr(scheduler, "run_one_job", run)

    assert scheduler.tick(verbose=False, sync=True) == 1
    expected = scheduled_fire_identity(due_job["id"], acquired_at)
    assert captured["created"] == (due_job["id"], "builtin", None)
    assert captured["bound"] == ("exec-builtin", expected)
    assert captured["run"]["fire_identity"] == expected


def test_provider_start_recovers_interrupted_records_before_tick(monkeypatch):
    import cron.scheduler_provider as provider

    events = []
    stop = __import__("threading").Event()
    stop.set()
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: events.append("recover") or 0,
        raising=False,
    )
    monkeypatch.setattr("cron.jobs.record_ticker_heartbeat", lambda **_kwargs: events.append("heartbeat"))

    provider.InProcessCronScheduler().start(stop, interval=1)

    assert events[:2] == ["recover", "heartbeat"]


def test_external_provider_start_recovers_interrupted_records(monkeypatch):
    from plugins.cron_providers.chronos import ChronosCronScheduler

    provider = ChronosCronScheduler()
    provider._client = type("Client", (), {"arm": lambda self, **kwargs: None})()
    events = []
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: events.append("recover") or 0,
    )
    monkeypatch.setattr(provider, "reconcile", lambda: events.append("reconcile"))

    provider.start(__import__("threading").Event())

    assert events == ["recover", "reconcile"]


class _TrackingConnection:
    """Delegates to a real sqlite3.Connection while recording close() calls.

    sqlite3.Connection is a static C type: it has no per-instance __dict__
    and its class methods can't be monkeypatched, so open/close tracking is
    done via a delegating wrapper returned in place of the real connection.
    """

    def __init__(self, real, closed_ids):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_closed_ids", closed_ids)

    def close(self):
        self._closed_ids.append(id(self._real))
        self._real.close()

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._real.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def _count_open_connections(executions, monkeypatch):
    """Wrap sqlite3.connect to track open/close balance for the ledger module."""
    opened_ids = []
    closed_ids = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened_ids.append(id(conn))
        return _TrackingConnection(conn, closed_ids)

    monkeypatch.setattr(executions.sqlite3, "connect", tracking_connect)
    return opened_ids, closed_ids


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Regression for #69567: every ledger call must close its connection
    deterministically instead of relying on garbage collection."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    record = executions.create_execution("leak-check", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)
    executions.list_executions(job_id="leak-check")
    executions.latest_executions(["leak-check"])
    executions.recover_interrupted_executions()

    assert len(opened) == 6
    assert len(closed) == 6
    assert set(opened) == set(closed)


def test_early_return_still_closes_connection(monkeypatch, tmp_path):
    """mark_execution_running returns None mid-block on a bad transition;
    the connection must still be closed rather than leaked."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    assert executions.mark_execution_running("does-not-exist") is None

    assert len(opened) == 1
    assert len(closed) == 1


def test_exception_during_operation_still_closes_connection(monkeypatch, tmp_path):
    """A failing statement inside the transaction must roll back and close,
    not leak the connection."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    with __import__("pytest").raises(sqlite3.IntegrityError):
        with executions._transaction() as conn:
            conn.execute(
                "INSERT INTO executions (id, job_id, source, process_id, pid, "
                "status, claimed_at) VALUES ('x', 'x', 'x', 'x', 1, 'bogus-status', 'now')"
            )

    assert len(opened) == 1
    assert len(closed) == 1


def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    """If PRAGMA/DDL setup in _connect() fails after sqlite3.connect()
    succeeds, the partially-initialized connection must still be closed."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened_ids = []
    closed_ids = []
    real_connect = sqlite3.connect

    class _FailingSchemaConnection(_TrackingConnection):
        def execute(self, sql, *args, **kwargs):
            if "CREATE TABLE" in sql:
                raise sqlite3.OperationalError("simulated schema init failure")
            return self._real.execute(sql, *args, **kwargs)

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened_ids.append(id(conn))
        return _FailingSchemaConnection(conn, closed_ids)

    monkeypatch.setattr(executions.sqlite3, "connect", tracking_connect)

    with __import__("pytest").raises(sqlite3.OperationalError):
        executions.create_execution("init-fail", source="builtin")

    assert len(opened_ids) == 1
    assert len(closed_ids) == 1


def test_job_listing_exposes_latest_execution(monkeypatch, tmp_path):
    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    executions = _point_ledger(monkeypatch, tmp_path)

    job = jobs.create_job(prompt="audit me", schedule="every 1h", name="audit")
    record = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(record["id"])

    listed = jobs.list_jobs(include_disabled=True)
    assert listed[0]["latest_execution"]["id"] == record["id"]
    assert listed[0]["latest_execution"]["status"] == "running"
