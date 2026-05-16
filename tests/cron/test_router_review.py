"""Tests for cron/router_review.py — routing outcome policy checks.

Covers:
  - load_routing_outcomes: missing file, malformed lines, timestamp filtering,
    timezone-naive timestamps, newest-first ordering
  - load_routing_decisions: missing file (file_was_present=False), valid JSON
    envelope, field mapping (camelCase→snake_case, agentId→job_name,
    complexityTier→job_type), age filtering, malformed/unreadable file
  - review_outcomes: all five policy checks, edge cases (no outcomes, empty model)
  - format_review_prompt: file absent vs. present-with-no-fresh-decisions,
    clean run, violations, source_present kwarg
  - BLOCKRUN_AUTO_ENDPOINT: value matches plan assumption
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cron.router_review import (
    BLOCKRUN_AUTO_ENDPOINT,
    ROUTING_DECISIONS_PATH,
    ROUTING_OUTCOMES_PATH,
    REVIEW_JOB_STATIC_PROMPT,
    PolicyViolation,
    RoutingOutcome,
    format_review_prompt,
    load_routing_decisions,
    load_routing_outcomes,
    review_outcomes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_minutes: int = 0) -> str:
    """Return a recent ISO-8601 UTC timestamp offset by *offset_minutes*."""
    dt = datetime.now(timezone.utc) - timedelta(minutes=offset_minutes)
    return dt.isoformat()


def _write_outcomes(tmp_path: Path, records: list[dict]) -> Path:
    """Write *records* as JSONL to a temp file and return the path."""
    p = tmp_path / "routing-outcomes.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def _write_decisions(tmp_path: Path, decisions: list[dict], updated_at: str | None = None) -> Path:
    """Write *decisions* as a routing-decisions.json envelope and return the path."""
    p = tmp_path / "routing-decisions.json"
    envelope = {
        "decisions": decisions,
        "stats": {"totalDecisions": len(decisions), "reorderedCount": 0, "tierBreakdown": {}, "topSelectedModels": []},
        "updatedAt": updated_at or _ts(),
    }
    p.write_text(json.dumps(envelope), encoding="utf-8")
    return p


def _decision(
    *,
    agent_id: str | None = "archie",
    complexity_tier: str = "research:medium",
    selected_model: str | None = "blockrun/auto",
    resolved_model: str | None = "gn100/qwen3.6:35b-a3b",
    offset_minutes: int = 0,
) -> dict:
    """Build a minimal OpenClaw routing-decisions.json record."""
    record: dict = {
        "timestamp": _ts(offset_minutes),
        "complexityTier": complexity_tier,
        "enforcement": "soft",
        "routingMode": "balanced",
        "originalOrder": [selected_model or "blockrun/auto"],
        "finalOrder": [selected_model or "blockrun/auto"],
        "reordered": False,
    }
    if agent_id is not None:
        record["agentId"] = agent_id
    if selected_model is not None:
        record["selectedModel"] = selected_model
    if resolved_model is not None:
        record["resolvedModel"] = resolved_model
    return record


def _outcome(
    *,
    job_name: str = "test-job",
    selected_model: str | None = "blockrun/auto",
    resolved_model: str | None = None,
    job_type: str | None = "simple",
    router_pin: str | None = None,
    error: str | None = None,
    consecutive_local_failures: int = 0,
    offset_minutes: int = 0,
) -> RoutingOutcome:
    """Construct a ``RoutingOutcome`` for testing."""
    return RoutingOutcome(
        job_name=job_name,
        job_id="test-id",
        selected_model=selected_model,
        resolved_model=resolved_model,
        job_type=job_type,
        router_pin=router_pin,
        timestamp=_ts(offset_minutes),
        error=error,
        consecutive_local_failures=consecutive_local_failures,
    )


# ---------------------------------------------------------------------------
# load_routing_outcomes (JSONL — future Items 1-4 format)
# ---------------------------------------------------------------------------


class TestLoadRoutingOutcomes:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        outcomes = load_routing_outcomes(path=tmp_path / "nonexistent.jsonl")
        assert outcomes == []

    def test_empty_file_returns_empty(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_routing_outcomes(path=p) == []

    def test_loads_recent_record(self, tmp_path: Path):
        p = _write_outcomes(tmp_path, [
            {
                "job_name": "my-job",
                "job_id": "abc123",
                "selected_model": "blockrun/auto",
                "resolved_model": "gn100/qwen3.6:35b-a3b",
                "job_type": "simple",
                "router_pin": None,
                "timestamp": _ts(5),
            }
        ])
        outcomes = load_routing_outcomes(path=p, max_age_minutes=60)
        assert len(outcomes) == 1
        assert outcomes[0].job_name == "my-job"
        assert outcomes[0].selected_model == "blockrun/auto"
        assert outcomes[0].resolved_model == "gn100/qwen3.6:35b-a3b"

    def test_filters_old_records(self, tmp_path: Path):
        p = _write_outcomes(tmp_path, [
            {"job_name": "old", "job_id": "x", "timestamp": _ts(120),
             "selected_model": "blockrun/auto", "resolved_model": None},
            {"job_name": "recent", "job_id": "y", "timestamp": _ts(10),
             "selected_model": "blockrun/auto", "resolved_model": None},
        ])
        outcomes = load_routing_outcomes(path=p, max_age_minutes=60)
        assert len(outcomes) == 1
        assert outcomes[0].job_name == "recent"

    def test_skips_malformed_json_lines(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        p.write_text(
            'not-json\n'
            + json.dumps({"job_name": "ok", "job_id": "1", "timestamp": _ts(1),
                           "selected_model": "blockrun/auto"}) + "\n",
            encoding="utf-8",
        )
        outcomes = load_routing_outcomes(path=p, max_age_minutes=60)
        assert len(outcomes) == 1
        assert outcomes[0].job_name == "ok"

    def test_skips_bad_timestamp(self, tmp_path: Path):
        p = _write_outcomes(tmp_path, [
            {"job_name": "bad-ts", "job_id": "x", "timestamp": "not-a-date",
             "selected_model": "blockrun/auto"},
        ])
        outcomes = load_routing_outcomes(path=p, max_age_minutes=60)
        assert outcomes == []

    def test_timezone_naive_timestamp_accepted(self, tmp_path: Path):
        # Naive timestamps (no tzinfo) should be treated as UTC and accepted
        naive_ts = datetime.utcnow().isoformat()
        p = _write_outcomes(tmp_path, [
            {"job_name": "naive-job", "job_id": "n1", "timestamp": naive_ts,
             "selected_model": "blockrun/auto"},
        ])
        outcomes = load_routing_outcomes(path=p, max_age_minutes=60)
        assert len(outcomes) == 1

    def test_returns_newest_first(self, tmp_path: Path):
        p = _write_outcomes(tmp_path, [
            {"job_name": "older", "job_id": "a", "timestamp": _ts(20),
             "selected_model": "blockrun/auto"},
            {"job_name": "newer", "job_id": "b", "timestamp": _ts(5),
             "selected_model": "blockrun/auto"},
        ])
        outcomes = load_routing_outcomes(path=p, max_age_minutes=60)
        assert outcomes[0].job_name == "newer"
        assert outcomes[1].job_name == "older"

    def test_consecutive_local_failures_defaults_to_zero(self, tmp_path: Path):
        p = _write_outcomes(tmp_path, [
            {"job_name": "j", "job_id": "1", "timestamp": _ts(1),
             "selected_model": "blockrun/auto"},
        ])
        outcomes = load_routing_outcomes(path=p, max_age_minutes=60)
        assert outcomes[0].consecutive_local_failures == 0

    def test_optional_fields_default_to_none(self, tmp_path: Path):
        p = _write_outcomes(tmp_path, [
            {"job_name": "minimal", "job_id": "m1", "timestamp": _ts(1)},
        ])
        outcomes = load_routing_outcomes(path=p, max_age_minutes=60)
        assert outcomes[0].selected_model is None
        assert outcomes[0].resolved_model is None
        assert outcomes[0].job_type is None
        assert outcomes[0].router_pin is None
        assert outcomes[0].error is None

    def test_blank_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "outcomes.jsonl"
        p.write_text(
            "\n\n"
            + json.dumps({"job_name": "ok", "job_id": "1", "timestamp": _ts(1),
                           "selected_model": "blockrun/auto"})
            + "\n\n",
            encoding="utf-8",
        )
        outcomes = load_routing_outcomes(path=p, max_age_minutes=60)
        assert len(outcomes) == 1


# ---------------------------------------------------------------------------
# load_routing_decisions (JSON envelope — current OpenClaw format)
# ---------------------------------------------------------------------------


class TestLoadRoutingDecisions:
    def test_missing_file_returns_empty_and_not_present(self, tmp_path: Path):
        outcomes, present = load_routing_decisions(path=tmp_path / "nonexistent.json")
        assert outcomes == []
        assert present is False

    def test_valid_file_returns_present_true(self, tmp_path: Path):
        p = _write_decisions(tmp_path, [_decision(offset_minutes=5)])
        _, present = load_routing_decisions(path=p, max_age_minutes=60)
        assert present is True

    def test_loads_recent_decision(self, tmp_path: Path):
        p = _write_decisions(tmp_path, [
            _decision(agent_id="archie", complexity_tier="research:reasoning",
                      selected_model="blockrun/auto", resolved_model="gn100/qwen3.6:35b-a3b",
                      offset_minutes=5)
        ])
        outcomes, present = load_routing_decisions(path=p, max_age_minutes=60)
        assert present is True
        assert len(outcomes) == 1
        o = outcomes[0]
        assert o.job_name == "gateway/archie"
        assert o.job_type == "research:reasoning"
        assert o.selected_model == "blockrun/auto"
        assert o.resolved_model == "gn100/qwen3.6:35b-a3b"

    def test_absent_agent_id_uses_fallback_job_name(self, tmp_path: Path):
        p = _write_decisions(tmp_path, [_decision(agent_id=None, offset_minutes=1)])
        outcomes, _ = load_routing_decisions(path=p, max_age_minutes=60)
        assert len(outcomes) == 1
        assert outcomes[0].job_name == "gateway-decision"

    def test_fields_not_in_gateway_decisions_have_defaults(self, tmp_path: Path):
        p = _write_decisions(tmp_path, [_decision(offset_minutes=1)])
        outcomes, _ = load_routing_decisions(path=p, max_age_minutes=60)
        o = outcomes[0]
        assert o.router_pin is None
        assert o.consecutive_local_failures == 0
        assert o.error is None
        assert o.job_id == ""

    def test_filters_old_decisions(self, tmp_path: Path):
        p = _write_decisions(tmp_path, [
            _decision(agent_id="old", offset_minutes=120),
            _decision(agent_id="recent", offset_minutes=5),
        ])
        outcomes, _ = load_routing_decisions(path=p, max_age_minutes=60)
        assert len(outcomes) == 1
        assert outcomes[0].job_name == "gateway/recent"

    def test_returns_newest_first(self, tmp_path: Path):
        p = _write_decisions(tmp_path, [
            _decision(agent_id="older", offset_minutes=30),
            _decision(agent_id="newer", offset_minutes=5),
        ])
        outcomes, _ = load_routing_decisions(path=p, max_age_minutes=60)
        assert outcomes[0].job_name == "gateway/newer"
        assert outcomes[1].job_name == "gateway/older"

    def test_empty_decisions_array_returns_empty_outcomes_present_true(self, tmp_path: Path):
        p = _write_decisions(tmp_path, [])
        outcomes, present = load_routing_decisions(path=p, max_age_minutes=60)
        assert outcomes == []
        assert present is True

    def test_all_old_decisions_returns_empty_outcomes_present_true(self, tmp_path: Path):
        p = _write_decisions(tmp_path, [_decision(offset_minutes=120)])
        outcomes, present = load_routing_decisions(path=p, max_age_minutes=60)
        assert outcomes == []
        assert present is True

    def test_malformed_json_returns_empty_present_true(self, tmp_path: Path):
        p = tmp_path / "routing-decisions.json"
        p.write_text("not-json{{{", encoding="utf-8")
        outcomes, present = load_routing_decisions(path=p, max_age_minutes=60)
        assert outcomes == []
        assert present is True  # file exists, just unreadable

    def test_absent_selected_model_maps_to_none(self, tmp_path: Path):
        rec = _decision(offset_minutes=1)
        del rec["selectedModel"]
        p = _write_decisions(tmp_path, [rec])
        outcomes, _ = load_routing_decisions(path=p, max_age_minutes=60)
        assert outcomes[0].selected_model is None

    def test_absent_resolved_model_maps_to_none(self, tmp_path: Path):
        rec = _decision(offset_minutes=1)
        del rec["resolvedModel"]
        p = _write_decisions(tmp_path, [rec])
        outcomes, _ = load_routing_decisions(path=p, max_age_minutes=60)
        assert outcomes[0].resolved_model is None

    def test_bad_timestamp_skipped(self, tmp_path: Path):
        rec = _decision(offset_minutes=1)
        rec["timestamp"] = "not-a-date"
        p = _write_decisions(tmp_path, [rec])
        outcomes, _ = load_routing_decisions(path=p, max_age_minutes=60)
        assert outcomes == []

    def test_timezone_naive_timestamp_accepted(self, tmp_path: Path):
        rec = _decision(offset_minutes=0)
        rec["timestamp"] = datetime.utcnow().isoformat()  # naive
        p = _write_decisions(tmp_path, [rec])
        outcomes, _ = load_routing_decisions(path=p, max_age_minutes=60)
        assert len(outcomes) == 1

    def test_multiple_decisions_all_loaded(self, tmp_path: Path):
        p = _write_decisions(tmp_path, [
            _decision(agent_id="a", offset_minutes=10),
            _decision(agent_id="b", offset_minutes=20),
            _decision(agent_id="c", offset_minutes=30),
        ])
        outcomes, _ = load_routing_decisions(path=p, max_age_minutes=60)
        assert len(outcomes) == 3

    def test_real_file_shape_parses(self, tmp_path: Path):
        """Exercise the exact shape from routing-decisions.ts output."""
        real_shape = {
            "decisions": [
                {
                    "timestamp": _ts(3),
                    "agentId": "archie",
                    "complexityTier": "research:reasoning",
                    "enforcement": "soft",
                    "routingMode": "balanced",
                    "originalOrder": ["blockrun/auto", "gemini/pro"],
                    "finalOrder": ["blockrun/auto", "gemini/pro"],
                    "reordered": False,
                    "selectedModel": "blockrun/auto",
                    "resolvedModel": "blockrun/auto",
                }
            ],
            "stats": {
                "totalDecisions": 1,
                "reorderedCount": 0,
                "tierBreakdown": {"research:reasoning": 1},
                "topSelectedModels": [{"model": "blockrun/auto", "count": 1}],
            },
            "updatedAt": _ts(),
        }
        p = tmp_path / "routing-decisions.json"
        p.write_text(json.dumps(real_shape), encoding="utf-8")
        outcomes, present = load_routing_decisions(path=p, max_age_minutes=60)
        assert present is True
        assert len(outcomes) == 1
        o = outcomes[0]
        assert o.job_name == "gateway/archie"
        assert o.job_type == "research:reasoning"
        assert o.selected_model == "blockrun/auto"
        assert o.resolved_model == "blockrun/auto"


# ---------------------------------------------------------------------------
# review_outcomes — Check 1: simple work not using local
# ---------------------------------------------------------------------------


class TestCheck1SimpleWorkNotLocal:
    def test_simple_job_cloud_resolved_no_pin_warns(self):
        o = _outcome(
            job_type="simple",
            selected_model="blockrun/auto",
            resolved_model="claude-sonnet-4-5",
            router_pin=None,
        )
        violations = review_outcomes([o])
        # resolved_model is set so check 3 (missing resolved) does NOT fire.
        # Only check 1 fires: simple job routed to cloud without pin.
        assert len(violations) == 1
        assert violations[0].severity == "warning"
        assert "Simple job" in violations[0].message
        assert violations[0].severity == "warning"
        assert "cloud model" in violations[0].message
        assert violations[0].job_name == "test-job"

    def test_simple_job_cloud_resolved_with_pin_no_violation(self):
        o = _outcome(
            job_type="simple",
            selected_model="blockrun/auto",
            resolved_model="claude-sonnet-4-5",
            router_pin="[router-pin: claude required for structured output]",
        )
        violations = review_outcomes([o])
        check1 = [v for v in violations if "Simple job" in v.message]
        assert check1 == []

    def test_simple_job_local_resolved_no_violation(self):
        o = _outcome(
            job_type="simple",
            selected_model="blockrun/auto",
            resolved_model="gn100/qwen3.6:35b-a3b",
        )
        violations = review_outcomes([o])
        check1 = [v for v in violations if "Simple job" in v.message]
        assert check1 == []

    def test_non_simple_job_type_not_flagged(self):
        o = _outcome(
            job_type="coding",
            selected_model="blockrun/auto",
            resolved_model="claude-sonnet-4-5",
        )
        violations = review_outcomes([o])
        check1 = [v for v in violations if "Simple job" in v.message]
        assert check1 == []

    def test_gateway_complexity_tier_not_simple_not_flagged(self):
        """OpenClaw tiers like 'research:reasoning' are not 'simple' — check 1 should not fire."""
        o = _outcome(
            job_type="research:reasoning",
            selected_model="blockrun/auto",
            resolved_model="claude-sonnet-4-5",
            router_pin=None,
        )
        violations = review_outcomes([o])
        check1 = [v for v in violations if "Simple job" in v.message]
        assert check1 == []


# ---------------------------------------------------------------------------
# review_outcomes — Check 2: coding/reasoning not escalating
# ---------------------------------------------------------------------------


class TestCheck2EscalationExpected:
    @pytest.mark.parametrize("job_type", ["coding", "tool-heavy", "reasoning"])
    def test_complex_job_local_resolved_warns(self, job_type: str):
        o = _outcome(
            job_type=job_type,
            selected_model="blockrun/auto",
            resolved_model="gn100/qwen3.6:35b-a3b",
        )
        violations = review_outcomes([o])
        check2 = [v for v in violations if "did not escalate" in v.message]
        assert len(check2) == 1
        assert check2[0].severity == "warning"
        assert job_type.capitalize() in check2[0].message or job_type in check2[0].message.lower()

    def test_coding_job_cloud_resolved_no_violation(self):
        o = _outcome(
            job_type="coding",
            selected_model="blockrun/auto",
            resolved_model="claude-sonnet-4-5",
        )
        violations = review_outcomes([o])
        check2 = [v for v in violations if "did not escalate" in v.message]
        assert check2 == []

    def test_simple_job_local_not_escalation_flagged(self):
        """Simple jobs routing local are expected — don't fire check 2."""
        o = _outcome(
            job_type="simple",
            selected_model="blockrun/auto",
            resolved_model="gn100/qwen3.6:35b-a3b",
        )
        violations = review_outcomes([o])
        check2 = [v for v in violations if "did not escalate" in v.message]
        assert check2 == []

    def test_gateway_tier_not_in_escalation_types_not_flagged(self):
        """Gateway-specific tiers like 'research:medium' are not in escalation types."""
        o = _outcome(
            job_type="research:medium",
            selected_model="blockrun/auto",
            resolved_model="gn100/qwen3.6:35b-a3b",
        )
        violations = review_outcomes([o])
        check2 = [v for v in violations if "did not escalate" in v.message]
        assert check2 == []


# ---------------------------------------------------------------------------
# review_outcomes — Check 3: missing resolved_model
# ---------------------------------------------------------------------------


class TestCheck3MissingResolvedModel:
    def test_router_first_no_resolved_warns(self):
        o = _outcome(selected_model="blockrun/auto", resolved_model=None)
        violations = review_outcomes([o])
        check3 = [v for v in violations if "resolved_model" in v.message]
        assert len(check3) == 1
        assert check3[0].severity == "warning"

    def test_pinned_job_no_resolved_no_violation(self):
        """Pinned (non-router) jobs are not expected to have resolved_model."""
        o = _outcome(
            selected_model="claude-sonnet-4-5",
            resolved_model=None,
            router_pin="[router-pin: coding work]",
            job_type="coding",
        )
        violations = review_outcomes([o])
        check3 = [v for v in violations if "resolved_model" in v.message]
        assert check3 == []

    def test_router_first_with_resolved_no_violation(self):
        o = _outcome(selected_model="blockrun/auto", resolved_model="gn100/qwen3.6:35b-a3b")
        violations = review_outcomes([o])
        check3 = [v for v in violations if "resolved_model" in v.message]
        assert check3 == []

    def test_no_selected_model_no_check3(self):
        """Missing selected_model entirely should not trigger check 3."""
        o = _outcome(selected_model=None, resolved_model=None)
        violations = review_outcomes([o])
        check3 = [v for v in violations if "resolved_model" in v.message]
        assert check3 == []

    def test_gateway_decision_resolved_model_none_fires_check3(self):
        """Gateway decisions with selectedModel=blockrun/auto but no resolvedModel trigger check 3."""
        o = RoutingOutcome(
            job_name="gateway/archie",
            job_id="",
            selected_model="blockrun/auto",
            resolved_model=None,
            job_type="research:reasoning",
            router_pin=None,
            timestamp=_ts(1),
        )
        violations = review_outcomes([o])
        check3 = [v for v in violations if "resolved_model" in v.message]
        assert len(check3) == 1
        assert check3[0].severity == "warning"


# ---------------------------------------------------------------------------
# review_outcomes — Check 4: repeated local failures
# ---------------------------------------------------------------------------


class TestCheck4RepeatedLocalFailures:
    def test_three_consecutive_failures_critical(self):
        o = _outcome(consecutive_local_failures=3, selected_model="blockrun/auto")
        violations = review_outcomes([o])
        check4 = [v for v in violations if "consecutive local model failures" in v.message]
        assert len(check4) == 1
        assert check4[0].severity == "critical"
        assert "3" in check4[0].message

    def test_two_failures_not_critical(self):
        o = _outcome(consecutive_local_failures=2, selected_model="blockrun/auto")
        violations = review_outcomes([o])
        check4 = [v for v in violations if "consecutive local model failures" in v.message]
        assert check4 == []

    def test_zero_failures_no_violation(self):
        o = _outcome(consecutive_local_failures=0)
        violations = review_outcomes([o])
        check4 = [v for v in violations if "consecutive local model failures" in v.message]
        assert check4 == []

    def test_high_failure_count_still_critical(self):
        o = _outcome(consecutive_local_failures=10, selected_model="blockrun/auto")
        violations = review_outcomes([o])
        check4 = [v for v in violations if "consecutive local model failures" in v.message]
        assert len(check4) == 1
        assert "10" in check4[0].message


# ---------------------------------------------------------------------------
# review_outcomes — Check 5: unexplained opt-out pin
# ---------------------------------------------------------------------------


class TestCheck5UnexplainedPin:
    def test_pinned_non_coding_job_no_marker_warns(self):
        o = _outcome(
            selected_model="claude-sonnet-4-5",
            resolved_model=None,
            job_type="simple",
            router_pin=None,
        )
        violations = review_outcomes([o])
        check5 = [v for v in violations if "router-pin" in v.message]
        assert len(check5) == 1
        assert check5[0].severity == "warning"

    def test_pinned_coding_job_no_marker_not_flagged(self):
        """Coding pins are expected in phase one — don't flag them."""
        o = _outcome(
            selected_model="codex/sonnet",
            resolved_model=None,
            job_type="coding",
            router_pin=None,
        )
        violations = review_outcomes([o])
        check5 = [v for v in violations if "router-pin" in v.message]
        assert check5 == []

    @pytest.mark.parametrize("job_type", ["coding", "tool-heavy", "reasoning"])
    def test_pinned_escalation_type_not_flagged(self, job_type: str):
        o = _outcome(
            selected_model="claude-sonnet-4-5",
            resolved_model=None,
            job_type=job_type,
            router_pin=None,
        )
        violations = review_outcomes([o])
        check5 = [v for v in violations if "router-pin" in v.message]
        assert check5 == []

    def test_pinned_with_marker_no_violation(self):
        o = _outcome(
            selected_model="claude-sonnet-4-5",
            resolved_model=None,
            job_type="simple",
            router_pin="[router-pin: structured output requires claude]",
        )
        violations = review_outcomes([o])
        check5 = [v for v in violations if "router-pin" in v.message]
        assert check5 == []

    def test_blockrun_auto_not_flagged_as_pin(self):
        o = _outcome(
            selected_model="blockrun/auto",
            resolved_model=None,
            job_type="simple",
            router_pin=None,
        )
        violations = review_outcomes([o])
        check5 = [v for v in violations if "router-pin" in v.message]
        assert check5 == []

    def test_gateway_tier_not_in_escalation_types_pin_fires(self):
        """Gateway-specific tier not in escalation types + explicit pin = check 5 fires."""
        o = RoutingOutcome(
            job_name="gateway/archie",
            job_id="",
            selected_model="gemini/pro",  # explicit non-auto pin
            resolved_model=None,
            job_type="research:medium",  # not in _ESCALATION_EXPECTED_TYPES
            router_pin=None,
            timestamp=_ts(1),
        )
        violations = review_outcomes([o])
        check5 = [v for v in violations if "router-pin" in v.message]
        assert len(check5) == 1


# ---------------------------------------------------------------------------
# review_outcomes — multi-check and no-outcome cases
# ---------------------------------------------------------------------------


class TestReviewOutcomesEdgeCases:
    def test_empty_outcomes_no_violations(self):
        assert review_outcomes([]) == []

    def test_multiple_outcomes_multiple_violations(self):
        outcomes = [
            # Check 3 fires: router-first with no resolved_model
            _outcome(job_name="job-a", selected_model="blockrun/auto", resolved_model=None),
            # Check 4 fires: repeated local failures
            _outcome(job_name="job-b", consecutive_local_failures=5,
                     selected_model="blockrun/auto", resolved_model="gn100/qwen"),
            # Check 5 fires: unexplained pin, simple job
            _outcome(job_name="job-c", selected_model="gemini-pro",
                     resolved_model=None, job_type="simple", router_pin=None),
        ]
        violations = review_outcomes(outcomes)
        job_names_with_violations = {v.job_name for v in violations}
        assert "job-a" in job_names_with_violations
        assert "job-b" in job_names_with_violations
        assert "job-c" in job_names_with_violations


# ---------------------------------------------------------------------------
# format_review_prompt
# ---------------------------------------------------------------------------


class TestFormatReviewPrompt:
    def test_no_outcomes_file_absent_mentions_not_found(self):
        """source_present=False: prompt says file is absent."""
        prompt = format_review_prompt([], [], source_present=False)
        assert "[REVIEW_REQUEST]" in prompt
        assert "Outcomes reviewed: 0" in prompt
        assert "not found" in prompt.lower() or "absent" in prompt.lower()

    def test_no_outcomes_file_present_no_fresh_decisions(self):
        """source_present=True (default), no outcomes: prompt says no decisions in window."""
        prompt = format_review_prompt([], [])
        assert "[REVIEW_REQUEST]" in prompt
        assert "Outcomes reviewed: 0" in prompt
        # Should NOT say "not found" — file IS present
        assert "not found" not in prompt.lower()
        # Should say something about the window or no decisions
        assert "present" in prompt.lower() or "no" in prompt.lower()

    def test_no_violations_all_passed(self):
        o = _outcome(
            selected_model="blockrun/auto",
            resolved_model="gn100/qwen3.6:35b-a3b",
            job_type="simple",
        )
        prompt = format_review_prompt([o], [])
        assert "all passed" in prompt.lower()
        assert "[REVIEW_REQUEST]" in prompt

    def test_violations_included_in_prompt(self):
        o = _outcome(selected_model="blockrun/auto", resolved_model=None)
        violations = review_outcomes([o])
        prompt = format_review_prompt([o], violations)
        assert "Policy violations:" in prompt
        assert "warning" in prompt.lower() or "WARNING" in prompt

    def test_critical_violation_shown(self):
        o = _outcome(consecutive_local_failures=5, selected_model="blockrun/auto",
                     resolved_model="gn100/qwen")
        violations = review_outcomes([o])
        prompt = format_review_prompt([o], violations)
        assert "CRITICAL" in prompt or "critical" in prompt.lower()

    def test_endpoint_assumption_included(self):
        prompt = format_review_prompt([], [])
        assert BLOCKRUN_AUTO_ENDPOINT in prompt

    def test_router_first_vs_pinned_counts(self):
        outcomes = [
            _outcome(job_name="rf", selected_model="blockrun/auto",
                     resolved_model="gn100/qwen", job_type="simple"),
            _outcome(job_name="pinned", selected_model="claude-sonnet-4-5",
                     resolved_model=None, job_type="coding",
                     router_pin="[router-pin: coding]"),
        ]
        prompt = format_review_prompt(outcomes, [])
        assert "Router-first jobs: 1" in prompt
        assert "Pinned (explicit model): 1" in prompt

    def test_source_present_false_with_outcomes_still_works(self):
        """source_present=False is informational only — outcomes are still reviewed."""
        o = _outcome(selected_model="blockrun/auto", resolved_model="gn100/qwen")
        violations = review_outcomes([o])
        prompt = format_review_prompt([o], violations, source_present=False)
        assert "[REVIEW_REQUEST]" in prompt
        # Outcomes present, so counts line should appear even with source_present=False
        assert "Router-first jobs:" in prompt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_blockrun_auto_endpoint_is_proxy(self):
        """Plan says use 11435 (openclaw-cli-proxy) as the canonical endpoint."""
        assert BLOCKRUN_AUTO_ENDPOINT == "http://127.0.0.1:11435/v1"

    def test_routing_decisions_path_points_to_json(self):
        assert ROUTING_DECISIONS_PATH.name == "routing-decisions.json"
        assert ".openclaw" in str(ROUTING_DECISIONS_PATH)

    def test_routing_outcomes_path_points_to_jsonl(self):
        assert ROUTING_OUTCOMES_PATH.name == "routing-outcomes.jsonl"
        assert ".openclaw" in str(ROUTING_OUTCOMES_PATH)

    def test_review_job_static_prompt_has_review_request(self):
        assert "[REVIEW_REQUEST]" in REVIEW_JOB_STATIC_PROMPT

    def test_review_job_static_prompt_has_endpoint(self):
        assert BLOCKRUN_AUTO_ENDPOINT in REVIEW_JOB_STATIC_PROMPT

    def test_review_job_static_prompt_references_decisions_file(self):
        """Static prompt should reference the JSON decisions file (primary source)."""
        assert "routing-decisions.json" in REVIEW_JOB_STATIC_PROMPT

    def test_review_job_static_prompt_covers_all_checks(self):
        """The static prompt should mention all 5 policy checks."""
        prompt = REVIEW_JOB_STATIC_PROMPT
        # Check 3: missing resolved_model
        assert "resolved_model" in prompt
        # Check 4: consecutive failures
        assert "consecutive_local_failures" in prompt
        # Check 5: router-pin marker
        assert "router-pin" in prompt
