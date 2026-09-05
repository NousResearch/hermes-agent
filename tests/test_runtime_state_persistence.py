"""Focused tests for the provider-neutral AgentRuntime v1 persistence seam."""

from __future__ import annotations

import math

import pytest

from agent.runtime_api import (
    RuntimeFailurePhase,
    RuntimeStateEnvelope,
    RuntimeUsageReceipt,
)
from hermes_state import SessionDB


def _receipt(
    *,
    correlation_id: str | None = "turn-1",
    fallback_used: bool = False,
    failure_phase: RuntimeFailurePhase | None = None,
    selected_model: str | None = None,
    effective_model: str | None = None,
    canonical_model: str | None = None,
    model_resolution: str = "unknown",
) -> RuntimeUsageReceipt:
    return RuntimeUsageReceipt(
        runtime_id="example-runtime",
        provider="example-provider",
        model="example-model",
        billing_mode="subscription",
        cost_status="known",
        input_tokens=10,
        output_tokens=4,
        cache_read_tokens=2,
        cache_write_tokens=1,
        reasoning_tokens=3,
        replay_safe=True,
        correlation_id=correlation_id,
        fallback_used=fallback_used,
        failure_phase=failure_phase,
        selected_model=selected_model,
        effective_model=effective_model,
        canonical_model=canonical_model,
        model_resolution=model_resolution,
    )


def test_fresh_schema_contains_runtime_tables(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"runtime_session_state", "runtime_usage_receipts"} <= tables

        state_columns = {
            row[1]
            for row in db._conn.execute(
                "PRAGMA table_info(runtime_session_state)"
            ).fetchall()
        }
        receipt_columns = {
            row[1]
            for row in db._conn.execute(
                "PRAGMA table_info(runtime_usage_receipts)"
            ).fetchall()
        }
        assert state_columns == {
            "session_id",
            "runtime_id",
            "schema_version",
            "state_json",
            "updated_at",
        }
        assert {
            "id",
            "session_id",
            "runtime_id",
            "provider",
            "model",
            "selected_model",
            "effective_model",
            "canonical_model",
            "model_resolution",
            "billing_mode",
            "cost_status",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "replay_safe",
            "correlation_id",
            "fallback_used",
            "failure_phase",
            "recorded_at",
        } <= receipt_columns
    finally:
        db.close()


def test_runtime_state_update_and_read_are_scoped_by_session_and_runtime(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-a", source="cli")
        first = RuntimeStateEnvelope(
            runtime_id="example-runtime",
            schema_version=1,
            state={"sdk_session_id": "synthetic-sdk-session", "attempt": 1},
        )
        db.update_runtime_state("session-a", first)
        assert db.get_runtime_state("session-a", "example-runtime") == first
        assert db.get_runtime_state("session-a", "other-runtime") is None

        second = RuntimeStateEnvelope(
            runtime_id="example-runtime",
            schema_version=2,
            state={"attempt": 2, "nested": {"ready": True}},
        )
        db.update_runtime_state("session-a", second)
        assert db.get_runtime_state("session-a", "example-runtime") == second
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM runtime_session_state WHERE session_id = ?",
                ("session-a",),
            ).fetchone()[0]
            == 1
        )
    finally:
        db.close()


@pytest.mark.parametrize(
    "state",
    [
        ["state-must-be-an-object"],
        {"access_token": "synthetic-secret-placeholder"},
        {"value": math.nan},
    ],
)
def test_runtime_state_rejects_unsafe_payloads(tmp_path, state):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-a", source="cli")
        with pytest.raises(ValueError):
            db.update_runtime_state(
                "session-a",
                RuntimeStateEnvelope(
                    runtime_id="example-runtime",
                    schema_version=1,
                    state=state,
                ),
            )
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM runtime_session_state"
            ).fetchone()[0]
            == 0
        )
    finally:
        db.close()


def test_runtime_usage_receipts_are_append_only_and_correlated_retries_are_idempotent(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-a", source="cli")
        original = _receipt()
        assert db.record_runtime_usage_receipt("session-a", original) is True
        session_before = db.get_session("session-a")
        assert session_before["input_tokens"] == 0
        assert session_before["output_tokens"] == 0
        assert (
            db._conn.execute(
                "SELECT COUNT(*) FROM session_model_usage WHERE session_id = ?",
                ("session-a",),
            ).fetchone()[0]
            == 0
        )

        changed_retry = RuntimeUsageReceipt(
            **{
                **original.__dict__,
                "output_tokens": 999,
                "fallback_used": True,
                "failure_phase": RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
                "selected_model": "changed-selection",
                "effective_model": "changed-effective",
                "canonical_model": "changed-canonical",
                "model_resolution": "mismatch",
            }
        )
        assert db.record_runtime_usage_receipt("session-a", changed_retry) is False
        assert db.list_runtime_usage_receipts("session-a") == [original]

        without_correlation = _receipt(correlation_id=None)
        assert db.record_runtime_usage_receipt("session-a", without_correlation) is True
        assert db.record_runtime_usage_receipt("session-a", without_correlation) is True
        assert len(db.list_runtime_usage_receipts("session-a")) == 3
        assert len(db.list_runtime_usage_receipts("session-a", "example-runtime")) == 3
    finally:
        db.close()


def test_runtime_usage_receipt_round_trips_model_provenance(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-a", source="cli")
        receipt = RuntimeUsageReceipt(
            runtime_id="example-runtime",
            provider="example-provider",
            model="example-model-2026-09",
            billing_mode="subscription",
            cost_status="known",
            correlation_id="turn-with-provenance",
            selected_model="example-model",
            effective_model="example-model",
            canonical_model="example-model-2026-09",
            model_resolution="canonicalized",
        )

        assert db.record_runtime_usage_receipt("session-a", receipt) is True
        assert db.list_runtime_usage_receipts("session-a") == [receipt]
    finally:
        db.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_used", 1),
        ("failure_phase", "unsupported-phase"),
        ("selected_model", 1),
        ("effective_model", "bad\nmodel"),
        ("canonical_model", "x" * 1025),
        ("model_resolution", None),
    ],
)
def test_runtime_usage_receipts_reject_untyped_classifications(
    tmp_path, field, value
):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-a", source="cli")
        receipt = _receipt(**{field: value})
        with pytest.raises(ValueError, match=field):
            db.record_runtime_usage_receipt("session-a", receipt)
    finally:
        db.close()


def test_legacy_runtime_usage_receipts_reconcile_classification_columns(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("session-a", source="cli")
    db._conn.execute(
        "ALTER TABLE runtime_usage_receipts DROP COLUMN fallback_used"
    )
    db._conn.execute(
        "ALTER TABLE runtime_usage_receipts DROP COLUMN failure_phase"
    )
    for column in (
        "selected_model",
        "effective_model",
        "canonical_model",
        "model_resolution",
    ):
        db._conn.execute(
            f'ALTER TABLE runtime_usage_receipts DROP COLUMN "{column}"'
        )
    db._conn.execute(
        """INSERT INTO runtime_usage_receipts (
               session_id, runtime_id, provider, model, billing_mode,
               cost_status, input_tokens, output_tokens, cache_read_tokens,
               cache_write_tokens, reasoning_tokens, replay_safe,
               correlation_id, recorded_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "session-a",
            "example-runtime",
            "example-provider",
            "example-model",
            "subscription",
            "known",
            10,
            4,
            2,
            1,
            3,
            1,
            "legacy-turn",
            1.0,
        ),
    )
    db._conn.commit()
    db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        columns = {
            row[1]
            for row in reopened._conn.execute(
                "PRAGMA table_info(runtime_usage_receipts)"
            ).fetchall()
        }
        assert {
            "fallback_used",
            "failure_phase",
            "selected_model",
            "effective_model",
            "canonical_model",
            "model_resolution",
        } <= columns
        assert reopened.list_runtime_usage_receipts("session-a") == [
            _receipt(correlation_id="legacy-turn")
        ]
        raw = reopened._conn.execute(
            """SELECT selected_model, effective_model, canonical_model,
                      model_resolution
                 FROM runtime_usage_receipts"""
        ).fetchone()
        assert tuple(raw) == (None, None, None, "unknown")
    finally:
        reopened.close()


def test_runtime_state_and_receipts_are_inert_across_reopen(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("session-a", source="cli")
    state = RuntimeStateEnvelope(
        runtime_id="example-runtime",
        schema_version=1,
        state={"retained": True},
    )
    receipt = _receipt(
        fallback_used=True,
        failure_phase=RuntimeFailurePhase.AFTER_SIDE_EFFECTS,
    )
    db.update_runtime_state("session-a", state)
    db.record_runtime_usage_receipt("session-a", receipt)
    raw_receipt = db._conn.execute(
        "SELECT fallback_used, failure_phase FROM runtime_usage_receipts"
    ).fetchone()
    assert tuple(raw_receipt) == (1, "after_side_effects")
    db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened.get_runtime_state("session-a", "example-runtime") == state
        assert reopened.list_runtime_usage_receipts("session-a") == [receipt]
        assert (
            reopened._conn.execute(
                "SELECT COUNT(*) FROM runtime_usage_receipts"
            ).fetchone()[0]
            == 1
        )
    finally:
        reopened.close()
