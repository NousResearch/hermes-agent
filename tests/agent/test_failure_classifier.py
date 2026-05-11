"""Tests for failure taxonomy, types, and deterministic classifier."""

from __future__ import annotations

import pytest

from agent.failure_analysis.types import (
    FailureType, FailureSubtype, Severity, NormalizedFailure,
)
from agent.failure_analysis.taxonomy import (
    VALID_PAIRS, is_valid_pair, all_types, subtypes_for,
)
from agent.failure_analysis.classifier import (
    classify_eval_case, classify_raw_error, _compute_fingerprint,
)


# ---------------------------------------------------------------------------
# Taxonomy tests
# ---------------------------------------------------------------------------

class TestTaxonomy:
    def test_valid_pairs_is_nonempty(self):
        assert len(VALID_PAIRS) >= 9

    def test_all_types_includes_minimum_set(self):
        types = all_types()
        for t in ("eval", "tool", "policy", "infra", "model", "unknown"):
            assert t in types, f"Missing type: {t}"

    def test_subtypes_for_eval(self):
        subs = subtypes_for("eval")
        assert "regression" in subs
        assert "failed_check" in subs

    def test_subtypes_for_tool(self):
        subs = subtypes_for("tool")
        assert "timeout" in subs
        assert "execution" in subs

    def test_is_valid_pair_true(self):
        assert is_valid_pair("eval", "regression")
        assert is_valid_pair("tool", "timeout")
        assert is_valid_pair("unknown", "unknown")

    def test_is_valid_pair_false(self):
        assert not is_valid_pair("eval", "timeout")
        assert not is_valid_pair("bogus", "bogus")

    def test_every_pair_validates(self):
        for t, s in VALID_PAIRS:
            assert is_valid_pair(t, s)


# ---------------------------------------------------------------------------
# Type tests
# ---------------------------------------------------------------------------

class TestTypes:
    def test_failure_type_values(self):
        assert FailureType.EVAL.value == "eval"
        assert FailureType.UNKNOWN.value == "unknown"

    def test_severity_values(self):
        assert Severity.LOW.value == "low"
        assert Severity.CRITICAL.value == "critical"

    def test_normalized_failure_defaults(self):
        nf = NormalizedFailure(id="test-001")
        assert nf.failure_type == "unknown"
        assert nf.failure_subtype == "unknown"
        assert nf.severity == "medium"
        assert nf.source_surface == ""
        assert nf.created_at > 0


# ---------------------------------------------------------------------------
# Classifier tests — eval cases
# ---------------------------------------------------------------------------

class TestClassifyEvalCase:
    def test_failed_check_default(self):
        cr = {"case_id": "c1", "status": "failed", "failure_summary": "check failed"}
        nf = classify_eval_case(cr, "run-001")
        assert nf.failure_type == "eval"
        assert nf.failure_subtype == "failed_check"
        assert nf.source_surface == "eval"
        assert nf.eval_run_id == "run-001"

    def test_regression_detected(self):
        cr = {"case_id": "c1", "status": "failed", "failure_summary": "score dropped"}
        nf = classify_eval_case(cr, "run-002", prior_status="passed")
        assert nf.failure_type == "eval"
        assert nf.failure_subtype == "regression"
        assert nf.severity == "high"

    def test_timeout_status(self):
        cr = {"case_id": "c1", "status": "timeout", "failure_summary": ""}
        nf = classify_eval_case(cr, "run-003")
        assert nf.failure_type == "tool"
        assert nf.failure_subtype == "timeout"

    def test_timeout_in_summary(self):
        cr = {"case_id": "c1", "status": "error", "failure_summary": "Command timed out after 60s"}
        nf = classify_eval_case(cr, "run-004")
        assert nf.failure_type == "tool"
        assert nf.failure_subtype == "timeout"

    def test_auth_failure(self):
        cr = {"case_id": "c1", "status": "error",
              "failure_summary": "401 Unauthorized: invalid api key"}
        nf = classify_eval_case(cr, "run-005")
        assert nf.failure_type == "infra"
        assert nf.failure_subtype == "auth"

    def test_env_failure(self):
        cr = {"case_id": "c1", "status": "error",
              "failure_summary": "ModuleNotFoundError: No module named 'foo'"}
        nf = classify_eval_case(cr, "run-006")
        assert nf.failure_type == "infra"
        assert nf.failure_subtype == "environment"

    def test_malformed_output(self):
        cr = {"case_id": "c1", "status": "error",
              "failure_summary": "JSON parse error on model response"}
        nf = classify_eval_case(cr, "run-007")
        assert nf.failure_type == "model"
        assert nf.failure_subtype == "output_malformed"

    def test_approval_blocked(self):
        cr = {"case_id": "c1", "status": "error",
              "failure_summary": "approval blocked: unsafe action refused"}
        nf = classify_eval_case(cr, "run-008")
        assert nf.failure_type == "policy"
        assert nf.failure_subtype == "approval_blocked"

    def test_tool_execution_error(self):
        cr = {"case_id": "c1", "status": "error",
              "failure_summary": "tool error: exit code 1"}
        nf = classify_eval_case(cr, "run-009")
        assert nf.failure_type == "tool"
        assert nf.failure_subtype == "execution"

    def test_fingerprint_is_set(self):
        cr = {"case_id": "c1", "status": "failed", "failure_summary": "check failed"}
        nf = classify_eval_case(cr, "run-010")
        assert len(nf.fingerprint) == 16

    def test_evidence_json_is_valid(self):
        import json
        cr = {"case_id": "c1", "status": "failed", "failure_summary": "oops",
              "deterministic_score": 0.5, "category": "reliability"}
        nf = classify_eval_case(cr, "run-011")
        ev = json.loads(nf.evidence_json)
        assert ev["status"] == "failed"
        assert ev["score"] == 0.5


# ---------------------------------------------------------------------------
# Classifier tests — raw errors
# ---------------------------------------------------------------------------

class TestClassifyRawError:
    def test_timeout(self):
        nf = classify_raw_error("Request timed out after 30s")
        assert nf.failure_type == "tool"
        assert nf.failure_subtype == "timeout"

    def test_auth(self):
        nf = classify_raw_error("authentication failed: invalid token",
                                tool_name="api_call", provider="openai")
        assert nf.failure_type == "infra"
        assert nf.failure_subtype == "auth"
        assert nf.tool_name == "api_call"
        assert nf.provider == "openai"

    def test_env(self):
        nf = classify_raw_error("ImportError: missing dependency 'numpy'")
        assert nf.failure_type == "infra"
        assert nf.failure_subtype == "environment"

    def test_unknown_fallback(self):
        nf = classify_raw_error("something weird happened")
        assert nf.failure_type == "unknown"
        assert nf.failure_subtype == "unknown"

    def test_summary_truncated(self):
        long_error = "x" * 1000
        nf = classify_raw_error(long_error)
        assert len(nf.summary) <= 500

    def test_session_and_task_propagated(self):
        nf = classify_raw_error("timeout", session_id="s1", task_id="t1")
        assert nf.session_id == "s1"
        assert nf.task_id == "t1"


# ---------------------------------------------------------------------------
# Fingerprint tests
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_same_input_same_fingerprint(self):
        fp1 = _compute_fingerprint("eval", "regression", None, None, "check failed")
        fp2 = _compute_fingerprint("eval", "regression", None, None, "check failed")
        assert fp1 == fp2

    def test_different_type_different_fingerprint(self):
        fp1 = _compute_fingerprint("eval", "regression", None, None, "check failed")
        fp2 = _compute_fingerprint("tool", "timeout", None, None, "check failed")
        assert fp1 != fp2

    def test_ids_normalized_away(self):
        fp1 = _compute_fingerprint("eval", "regression", None, None,
                                   "case abc12345 failed")
        fp2 = _compute_fingerprint("eval", "regression", None, None,
                                   "case def67890 failed")
        assert fp1 == fp2

    def test_fingerprint_length(self):
        fp = _compute_fingerprint("eval", "regression", None, None, "test")
        assert len(fp) == 16
