import time

from gateway.dev_control.reliability import (
    DevReliabilityStore,
    ReliabilityConfig,
    classify_trust_tier,
    compose_task_outcome,
    normalize_outcome,
    measure_category_improvement,
    scorecard,
    weakest_categories,
)


def _config() -> ReliabilityConfig:
    return ReliabilityConfig(
        window_days=30,
        escape_window_days=14,
        observed_min_samples=10,
        observed_success_rate=0.90,
        trusted_min_samples=30,
        trusted_success_rate=0.95,
        trusted_min_window_days=30,
    )


def _outcome(index: int, *, category: str = "workspace.implement/high", success: bool = True, escaped: bool = False, completed_at: float = 1_000_000.0):
    return {
        "plan_id": f"plan-{index}",
        "task_id": f"task-{index}",
        "category": category,
        "profile_id": category.split("/", 1)[0],
        "risk_level": category.split("/", 1)[1],
        "terminal_status": "completed" if success else "failed",
        "merged": success,
        "verification_verdict": "verified" if success else "failed",
        "ci_state": "success" if success else "failure",
        "code_review_verdict": "approved" if success else "changes_requested",
        "output_contract_score": 0.92 if success else 0.44,
        "rework_count": 0 if success else 1,
        "escaped": escaped,
        "escape_refs": [{"type": "incident", "incident_id": "devinc-1"}] if escaped else [],
        "completed_at": completed_at + index,
        "updated_at": completed_at + index,
    }


def test_task_outcome_composes_gate_signals_and_escape_join():
    merged_at = 1_000_000.0
    outcome = compose_task_outcome(
        plan={"plan_id": "plan-1", "status": "completed", "updated_at": merged_at},
        task={
            "task_id": "task-1",
            "profile_id": "workspace.implement",
            "status": "completed",
            "updated_at": merged_at,
            "payload": {"risk_level": "high", "output_contract_score": 0.9},
        },
        verification={"verdict": "verified"},
        pr_state={
            "ci_state": "success",
            "merge_state": "merged",
            "head_sha": "abc123",
            "raw": {"merged": True},
            "merged_at": merged_at,
        },
        code_review={"verdict": "approved"},
        incidents=[{
            "incident_id": "devinc-1",
            "detected_at": merged_at + 3600,
            "correlated_release": {"commit_sha": "abc123456"},
        }],
        config=_config(),
        now=merged_at + 7200,
    )

    assert outcome["category"] == "workspace.implement/high"
    assert outcome["verification_verdict"] == "verified"
    assert outcome["ci_state"] == "success"
    assert outcome["code_review_verdict"] == "approved"
    assert outcome["escaped"] is True
    assert outcome["success"] is False
    assert outcome["escape_refs"][0]["incident_id"] == "devinc-1"


def test_scorecard_rates_trend_and_tier_gating():
    now = 2_000_000.0
    current = [_outcome(i, completed_at=now - 1000) for i in range(30)]
    previous = [
        _outcome(100 + i, success=i < 20, completed_at=now - (31 * 86400))
        for i in range(30)
    ]

    card = scorecard(current + previous, now=now, config=_config())
    category = card["categories"][0]

    assert category["sample_count"] == 30
    assert category["success_rate"] == 1.0
    assert category["escape_rate"] == 0.0
    assert category["trend"] == "improving"
    assert category["tier"] == "trusted"


def test_tier_gating_unproven_for_insufficient_samples_and_escape_demotes():
    insufficient = {
        "sample_count": 9,
        "success_rate": 1.0,
        "escape_count": 0,
        "window": {"days": 30},
    }
    escaped = {
        "sample_count": 30,
        "success_rate": 1.0,
        "escape_count": 1,
        "window": {"days": 30},
    }

    assert classify_trust_tier(insufficient, config=_config())[0] == "unproven"
    tier, reasons = classify_trust_tier(escaped, config=_config())
    assert tier == "unproven"
    assert "escape" in reasons[0]


def test_draft_pr_only_success_excludes_unmeasured_ci_and_review():
    outcome = normalize_outcome({
        "plan_id": "lab-plan",
        "task_id": "lab-task",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "completed",
        "merged": False,
        "verification_verdict": "verified",
        "ci_state": "unknown",
        "code_review_verdict": "unknown",
        "source_refs": {
            "source": "dogfood_lab_loop",
            "draft_pr_only": True,
            "draft_pr_ready": True,
            "gates": {"ci": "not_measured", "review": "not_measured"},
        },
    })

    assert outcome["success"] is True


def test_draft_pr_only_success_fails_when_measured_gate_is_red():
    ci_failed = normalize_outcome({
        "plan_id": "lab-plan-ci",
        "task_id": "lab-task-ci",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "completed",
        "merged": False,
        "verification_verdict": "verified",
        "ci_state": "failure",
        "code_review_verdict": "unknown",
        "source_refs": {
            "source": "dogfood_lab_loop",
            "draft_pr_only": True,
            "draft_pr_ready": True,
            "gates": {"ci": "failure", "review": "not_measured"},
        },
    })
    review_failed = normalize_outcome({
        "plan_id": "lab-plan-review",
        "task_id": "lab-task-review",
        "profile_id": "platform.implement",
        "risk_level": "low",
        "terminal_status": "completed",
        "merged": False,
        "verification_verdict": "verified",
        "ci_state": "unknown",
        "code_review_verdict": "changes_requested",
        "source_refs": {
            "source": "dogfood_lab_loop",
            "draft_pr_only": True,
            "draft_pr_ready": True,
            "gates": {"ci": "not_measured", "review": "changes_requested"},
        },
    })

    assert ci_failed["success"] is False
    assert review_failed["success"] is False


def test_weakest_categories_and_before_after_measurement(tmp_path):
    store = DevReliabilityStore(tmp_path / "state.db")
    before_start = 1_000_000.0
    before_end = before_start + 7 * 86400
    after_start = before_end
    after_end = after_start + 7 * 86400
    for index in range(10):
        store.upsert_outcome(_outcome(index, success=index < 6, completed_at=before_start + index))
    for index in range(10, 20):
        store.upsert_outcome(_outcome(index, success=True, completed_at=after_start + index))
    weak = weakest_categories(scorecard(store.list_outcomes(limit=100), now=after_end, config=_config())["categories"])

    measurement = measure_category_improvement(
        store=store,
        category="workspace.implement/high",
        before_start=before_start,
        before_end=before_end,
        after_start=after_start,
        after_end=after_end,
        proposal_id="proposal-1",
        plan_id="plan-fix",
        config=_config(),
    )

    assert weak[0]["category"] == "workspace.implement/high"
    assert measurement["before_score"] == 0.6
    assert measurement["after_score"] == 1.0
    assert measurement["before_sample_count"] == 10
    assert measurement["after_sample_count"] == 10
    assert store.list_improvement_measurements(category="workspace.implement/high")[0]["proposal_id"] == "proposal-1"
