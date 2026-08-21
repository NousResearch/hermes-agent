from __future__ import annotations

import json
import sqlite3

import pytest


def _conn() -> sqlite3.Connection:
    from hermes_state_common import SCHEMA_SQL

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('s-1', 'test', 0)"
    )
    return conn


def _event(**overrides):
    from agent.retry_ledger import normalize_fp_v1

    event = {
        "run_id": "run-1",
        "session_id": "s-1",
        "repo": "local",
        "branch": "main",
        "head_sha": "a" * 40,
        "task_id": "task",
        "objective_id": "objective",
        "authority_mode": "R0",
        "loop_iteration": 0,
        "attempt_number": 1,
        "strategy_mode": "same_strategy",
        "check_name": "unit",
        "check_type": "test",
        "result_state": "fail",
        "error_class": "rate_limit",
        "error_fingerprint": normalize_fp_v1({"kind": "rate"}),
        "decision": "retry",
        "decision_reason": "rate_limit",
        "tool_calls_count": 0,
        "tokens_used_input": None,
        "tokens_used_output": None,
        "estimated_cost_usd": None,
        "duration_ms": 1,
        "changed_paths": [],
        "stop_reason": "none",
        "created_at": 1.0,
    }
    event.update(overrides)
    return event


def test_schema_has_exact_v1_enums_nullable_usage_fk_and_index() -> None:
    from agent.retry_ledger import verify_retry_ledger_code

    conn = _conn()
    schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='retry_ledger_events'"
    ).fetchone()[0]
    assert "'R0', 'R1'" in schema
    assert (
        "'same_strategy', 'fresh_context', 'different_strategy', 'human_escalated'"
        in schema
    )
    assert "'test', 'lint', 'type', 'policy', 'build', 'custom'" in schema
    assert "'pass', 'fail', 'blocked', 'skipped', 'timeout'" in schema
    assert "'stop', 'retry', 'escalate'" in schema
    nullable = {
        row[1]: row[3] for row in conn.execute("PRAGMA table_info(retry_ledger_events)")
    }
    assert (
        nullable["tokens_used_input"]
        == nullable["tokens_used_output"]
        == nullable["estimated_cost_usd"]
        == 0
    )
    assert {
        row[2] for row in conn.execute("PRAGMA foreign_key_list(retry_ledger_events)")
    } == {"sessions"}
    assert "idx_retry_ledger_events_run_check" in {
        row[1] for row in conn.execute("PRAGMA index_list(retry_ledger_events)")
    }
    assert verify_retry_ledger_code(conn)["checks"]["schema_exact_enums"]


def test_writer_r0_contract_nulls_canonical_paths_sequences_and_append_only() -> None:
    from agent.retry_ledger import RetryLedgerValidationError, RetryLedgerWriter

    conn = _conn()
    writer = RetryLedgerWriter(conn)
    writer.append(_event())
    row = conn.execute(
        "SELECT tokens_used_input, tokens_used_output, estimated_cost_usd, changed_paths FROM retry_ledger_events"
    ).fetchone()
    assert tuple(row[:3]) == (None, None, None)
    assert json.loads(row[3]) == []
    with pytest.raises(RetryLedgerValidationError, match="R0-only"):
        writer.append(_event(run_id="r0", authority_mode="R1"))
    with pytest.raises(RetryLedgerValidationError, match="raw or unknown"):
        writer.append(_event(payload="diagnostic"))
    with pytest.raises(RetryLedgerValidationError, match="empty"):
        writer.append(_event(attempt_number=2, changed_paths=["x.py"]))
    with pytest.raises(RetryLedgerValidationError, match="contiguous"):
        writer.append(_event(attempt_number=3))
    writer.append(_event(check_name="other", loop_iteration=1))
    with pytest.raises(RetryLedgerValidationError, match="monotonic"):
        writer.append(_event(check_name="third", loop_iteration=0))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE retry_ledger_events SET decision='stop'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM retry_ledger_events")


def test_fingerprint_normalizes_arbitrary_safe_values_without_raw_output() -> None:
    from agent.retry_ledger import normalize_fp_v1

    mac_root = chr(47) + "Users"
    left = normalize_fp_v1({
        "message": {
            "value": f"Timeout {mac_root}/a/x.py:42 request_id=a after 12ms at 2026-01-01T00:00:00Z",
            "uuid": "123e4567-e89b-12d3-a456-426614174000",
        },
        "code": 429,
    })
    right = normalize_fp_v1({
        "code": "429",
        "message": {
            "uuid": "abcdefab-cdef-abcd-efab-cdefabcdefab",
            "value": f"timeout {mac_root}/b/y.py:7 request_id=b after 999 milliseconds at 2027-01-01T00:00:00Z",
        },
    })
    assert left == right
    assert left.startswith("efp/v1:") and "timeout" not in left


def test_decision_units_cover_all_caps_classes_and_duplicate_levels() -> None:
    from agent.retry_ledger import BudgetConfig, decide_r0_retry

    budget = BudgetConfig()
    base = {
        "loop_iteration": 0,
        "elapsed_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "attempts_for_check": 0,
    }
    assert (
        budget.max_iterations,
        budget.wall_clock_minutes,
        budget.max_input_tokens,
        budget.max_output_tokens,
    ) == (3, 20, 75_000, 12_000)
    for error in (
        "validation_schema",
        "policy_safety",
        "auth_permission",
        "scope_or_authority",
    ):
        assert decide_r0_retry(base, budget, error).stop_reason == error
    assert (
        decide_r0_retry(
            {**base, "attempts_for_check": 1}, budget, "transient_network"
        ).decision
        == "retry"
    )
    assert (
        decide_r0_retry(
            {**base, "attempts_for_check": 2}, budget, "rate_limit"
        ).stop_reason
        == "rate_limit_retry_limit"
    )
    assert (
        decide_r0_retry(
            {**base, "attempts_for_check": 1}, budget, "timeout"
        ).stop_reason
        == "timeout_retry_limit"
    )
    assert (
        decide_r0_retry(
            {**base, "fingerprint_count": 2}, budget, "duplicate_fingerprint"
        ).stop_reason
        == "duplicate_second"
    )
    assert (
        decide_r0_retry(
            {**base, "fingerprint_count": 3}, budget, "duplicate_fingerprint"
        ).stop_reason
        == "duplicate_third_hard_stop"
    )
    for state, reason in (
        ({"loop_iteration": 3}, "iteration_cap"),
        ({"elapsed_ms": 1_200_000}, "wall_clock_cap"),
        ({"input_tokens": 75_000}, "input_tokens_cap"),
        ({"output_tokens": 12_000}, "output_tokens_cap"),
    ):
        assert (
            decide_r0_retry({**base, **state}, budget, "rate_limit").stop_reason
            == reason
        )


def test_real_synthetic_pilot_derives_report_and_passes_full_verifier() -> None:
    from agent.retry_ledger import run_synthetic_pilot

    result = run_synthetic_pilot()
    report, verifier = result["report"], result["verifier"]
    assert report["attempt_count"] == 6
    assert set(report["scenarios"]) == {
        "rate_limit_retry_to_pass",
        "validation_stop",
        "duplicate_stop",
        "input_budget_stop",
    }
    assert (
        report["scenarios"]["rate_limit_retry_to_pass"]["terminal_stop_reason"]
        == "terminal_pass"
    )
    assert (
        report["scenarios"]["validation_stop"]["terminal_stop_reason"]
        == "validation_schema"
    )
    assert (
        report["scenarios"]["duplicate_stop"]["terminal_stop_reason"]
        == "duplicate_second"
    )
    assert (
        report["scenarios"]["input_budget_stop"]["terminal_stop_reason"]
        == "input_tokens_cap"
    )
    assert verifier["ok"] is True
    assert all(verifier["checks"].values())
