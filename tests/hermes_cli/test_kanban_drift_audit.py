"""CH-424 sustained Kanban drift audit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_closeout as closeout
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_drift_audit as audit
from hermes_cli.kanban import run_slash


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_audit_report_records_required_fields_and_non_authoritative_boundary(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="ready item", tenant="brain-os", public_id="CH-424")
        report = audit.audit_drift(
            conn,
            domain="brain-os",
            mode="informational",
            sampled_ids=["CH-424"],
            now=1_700_000_000,
        )

    payload = report.to_dict()
    assert payload["domain"] == "brain-os"
    assert payload["timestamp"] == 1_700_000_000
    assert "kanban.db" in payload["authority_layer"]
    assert "non_authoritative" in payload["authority_layer"]
    assert payload["sampled_ids"] == ["CH-424"]
    assert payload["result_class"] == "ok"
    assert payload["mode"] == "informational"
    assert payload["blocking"] is False
    assert "linear_compatibility_shadow" in payload["non_authoritative_surfaces"]
    assert task_id


def test_audit_classifies_only_allowed_mismatch_classes(kanban_home):
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="first",
            tenant="brain-os",
            public_id="CH-425",
            idempotency_key="linear:CH-425",
        )
        second = kb.create_task(conn, title="second", tenant="brain-os", public_id="CH-426")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET idempotency_key='linear:CH-425' WHERE id=?", (second,))
        report = audit.audit_drift(
            conn,
            domain="brain-os",
            mode="gating",
            projection_snapshot={
                "records": [
                    {
                        "surface": "wiki",
                        "authority": "projection_only; non-authoritative",
                        "generated_at": 1_600_000_000,
                        "task_ids": ["CH-425", "CH-NOPE"],
                        "body": "api_key=abc123",
                    }
                ]
            },
            now=1_700_000_000,
        )

    classes = {finding.result_class for finding in report.findings}
    assert classes <= set(audit.MISMATCH_CLASSES)
    assert {
        "duplicate_shadow",
        "missing_mapping",
        "stale_snapshot",
        "secret_hygiene",
    } <= classes
    assert report.blocking is True
    assert report.result_class in audit.MISMATCH_CLASSES
    assert first and second


def test_projection_authority_claim_fails_closed_and_blocks_flip_closeout(kanban_home):
    with kb.connect() as conn:
        kb.create_task(conn, title="sample", tenant="brain-os", public_id="CH-426")
        report = audit.audit_drift(
            conn,
            domain="brain-os",
            mode="gating",
            projection_snapshot={"records": [{"task_ids": ["CH-426"], "body": "authoritative mirror"}]},
            now=1_700_000_000,
        )

    assert report.result_class == "projection_authority_claim"
    assert report.blocking is True
    assert any(
        f.result_class == "projection_authority_claim" and f.blocks_flip_closeout
        for f in report.findings
    )

    verification = closeout.verify_closeout_transition(
        "closed",
        {
            "summary": "worker proof",
            "cleanup": {"proof": "clean", "worktree_clean": True},
            "no_pr_exception": {"policy": "docs-only", "reason": "no code"},
            "drift_audit": report.to_dict(),
        },
        current_phase="review_ready",
    )
    assert verification.allowed is False
    assert "drift_audit_projection_authority_claim" in verification.blockers


def test_projection_authority_claim_is_specific_fail_closed_class(kanban_home):
    with kb.connect() as conn:
        kb.create_task(conn, title="sample", tenant="brain-os", public_id="BO-007")
        report = audit.audit_drift(
            conn,
            domain="brain-os",
            mode="gating",
            projection_snapshot={
                "records": [
                    {
                        "surface": "linear_compatibility_shadow",
                        "authority": "linear authority",
                        "task_ids": ["BO-007"],
                        "body": "Linear compatibility projection claims canonical source of truth",
                    }
                ]
            },
            now=1_700_000_000,
        )

    assert report.result_class == "projection_authority_claim"
    assert report.blocking is True
    finding = report.findings[0]
    assert finding.result_class == "projection_authority_claim"
    assert finding.surface == "linear_compatibility_shadow"
    assert finding.blocks_flip_closeout is True

    blockers = audit.closeout_blocks_from_audit(report.to_dict())
    assert "drift_audit_projection_authority_claim" in blockers


def test_projection_lag_is_reported_without_promoting_projection_to_authority(kanban_home):
    with kb.connect() as conn:
        kb.create_task(conn, title="sample", tenant="brain-os", public_id="CH-427")
        report = audit.audit_drift(
            conn,
            domain="brain-os",
            mode="gating",
            projection_snapshot={
                "records": [
                    {
                        "authority": "projection_only; non-authoritative",
                        "generated_at": 1_700_000_000,
                        "task_ids": [],
                        "body": "projection_only summary",
                    }
                ]
            },
            now=1_700_000_010,
        )

    assert report.result_class == "projection_lag"
    assert report.blocking is False
    assert "non_authoritative" in report.authority_layer


def test_cli_drift_audit_json_is_compact_and_review_queue_safe(kanban_home):
    with kb.connect() as conn:
        kb.create_task(
            conn,
            title="review ready without proof",
            tenant="brain-os",
            public_id="CH-428",
            review_phase="review_ready",
        )

    out = run_slash("drift-audit --domain brain-os --mode gating --json")
    payload = json.loads(out)

    assert payload["schema"] == audit.AUDIT_SCHEMA
    assert payload["domain"] == "brain-os"
    assert payload["mode"] == "gating"
    assert payload["result_class"] == "closeout_gap"
    assert payload["blocking"] is True
    assert payload["sampled_ids"] == ["CH-428"]
    assert len(json.dumps(payload)) < 2500
    assert payload["findings"][0]["result_class"] == "closeout_gap"
