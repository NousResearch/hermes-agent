"""The 'model-visible means logged' invariant."""

from __future__ import annotations

import pytest

from nova.audit import (
    MODEL_VISIBLE_KINDS,
    AuditEvent,
    AuditLog,
    NullAuditLog,
    new_correlation_id,
)
from nova.errors import AuditError


def test_model_visible_change_writes_intent_then_commit(audit):
    cid = new_correlation_id()
    with audit.model_visible_change("agent.materialized", correlation_id=cid, subject="a") as out:
        out["paths"] = ["config.yaml"]
    phases = [(e.kind, e.phase) for e in audit.read()]
    assert phases == [("agent.materialized", "intent"), ("agent.materialized", "committed")]


def test_commit_carries_what_actually_happened(audit):
    cid = new_correlation_id()
    with audit.model_visible_change("agent.materialized", correlation_id=cid, subject="a") as out:
        out["paths"] = ["config.yaml"]
    committed = [e for e in audit.read() if e.phase == "committed"][0]
    assert committed.detail["paths"] == ["config.yaml"]


def test_failure_is_recorded_and_reraised(audit):
    cid = new_correlation_id()
    with pytest.raises(ValueError):
        with audit.model_visible_change("agent.materialized", correlation_id=cid, subject="a"):
            raise ValueError("disk full")
    failed = [e for e in audit.read() if e.phase == "failed"]
    assert len(failed) == 1
    assert "disk full" in failed[0].error


def test_model_visible_kind_cannot_bypass_the_guard(audit):
    """The invariant is structural: record() refuses model-visible kinds."""
    with pytest.raises(AuditError, match="must be written through"):
        audit.record("agent.materialized", correlation_id=new_correlation_id())


def test_non_model_visible_kind_cannot_use_the_guard(audit):
    with pytest.raises(AuditError, match="not declared in MODEL_VISIBLE_KINDS"):
        with audit.model_visible_change("bundle.apply_started", correlation_id="c", subject="s"):
            pass


def test_every_model_visible_kind_is_reachable_through_the_guard(audit):
    """Guards against a kind being added to the set but never routed through the guard."""
    for kind in sorted(MODEL_VISIBLE_KINDS):
        with audit.model_visible_change(kind, correlation_id="c", subject="s"):
            pass
    kinds = {e.kind for e in audit.read()}
    assert kinds == set(MODEL_VISIBLE_KINDS)


def test_open_intent_is_visible_after_a_crash(tmp_path):
    """A process killed mid-change leaves an unanswered intent — the point of write-ahead."""
    log = AuditLog(tmp_path / "a.jsonl", tenant_id="t")
    crashed = AuditEvent(
        event_id="e1",
        correlation_id=new_correlation_id(),
        ts="2026-01-01T00:00:00.000Z",
        kind="agent.materialized",
        phase="intent",
        actor="test",
        tenant_id="t",
        model_visible=True,
        subject="a",
    )
    # Append the intent with no matching outcome, exactly as a hard kill would leave it.
    log.path.write_text(crashed.to_json() + "\n", encoding="utf-8")
    assert [e.subject for e in log.open_intents()] == ["a"]


def test_settled_changes_leave_no_open_intents(audit):
    cid = new_correlation_id()
    with audit.model_visible_change("agent.materialized", correlation_id=cid, subject="a"):
        pass
    assert audit.open_intents() == []


def test_events_round_trip_through_json(audit):
    cid = new_correlation_id()
    audit.record("bundle.apply_started", correlation_id=cid, subject="acme", detail={"n": 1})
    event = audit.read()[0]
    assert event.correlation_id == cid
    assert event.detail == {"n": 1}
    assert event.model_visible is False
    assert event.tenant_id == "acme"


def test_correlation_id_ties_events_together(audit):
    cid = new_correlation_id()
    audit.record("bundle.apply_started", correlation_id=cid, subject="acme")
    with audit.model_visible_change("agent.materialized", correlation_id=cid, subject="a"):
        pass
    assert {e.correlation_id for e in audit.read()} == {cid}


def test_null_log_discards_but_still_enforces(tmp_path):
    log = NullAuditLog(tenant_id="t")
    with log.model_visible_change("agent.materialized", correlation_id="c", subject="a"):
        pass
    assert log.read() == []
    with pytest.raises(AuditError):
        log.record("agent.materialized", correlation_id="c")


def test_log_is_created_with_restrictive_permissions(tmp_path):
    import os
    import stat

    log = AuditLog(tmp_path / "a.jsonl", tenant_id="t")
    log.record("bundle.apply_started", correlation_id="c", subject="s")
    mode = stat.S_IMODE(os.stat(log.path).st_mode)
    assert mode & 0o077 == 0, f"audit log is group/world accessible: {oct(mode)}"
