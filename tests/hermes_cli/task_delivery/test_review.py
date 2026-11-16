"""Managed reviews retain issue budgets and refuse ambiguous or stale verdicts."""
import copy

import pytest

from hermes_cli.task_delivery.review import ReviewError, allow_repair, apply_review, validate_review


def _report(request="review-1", candidate="candidate-1", *, issue="S1", blocking=True, reviewer="reviewer"):
    return {
        "request_id": request,
        "reviewer": reviewer,
        "candidate_digest": candidate,
        "passed": not blocking,
        "findings": [{
            "issue_id": issue, "category": "release_blocker", "blocking": blocking,
            "evidence_ref": "verification/repro.json", "message": "The old request still changes account state.",
        }],
    }


def test_review_budget_survives_candidates_replays_and_new_findings():
    report = _report()
    untouched = copy.deepcopy(report)
    state = apply_review(None, report, candidate_digest="candidate-1")
    assert report == untouched
    assert state["status"] == "changes_requested" and allow_repair(state)
    initial = copy.deepcopy(state)
    assert apply_review(state, report, candidate_digest="newer-workspace") == state
    assert state == initial
    with pytest.raises(ReviewError, match="request_id"):
        apply_review(state, _report(reviewer="other"), candidate_digest="candidate-1")

    # A second opinion on unchanged source is not another automatic repair.
    state = apply_review(state, _report("second-opinion", reviewer="other"), candidate_digest="candidate-1")
    assert state["issues"]["S1"]["candidate_digests"] == ["candidate-1"]
    for attempt in (2, 3):
        candidate = f"candidate-{attempt}"
        state = apply_review(state, _report(f"review-{attempt}", candidate), candidate_digest=candidate)
    assert state["status"] == "needs_decision" and not allow_repair(state)
    assert state["issues"]["S1"]["candidate_digests"] == ["candidate-1", "candidate-2", "candidate-3"]

    # A new candidate, renamed finding, or optimistic verdict cannot reset exhaustion.
    for report in (_report("fourth", "candidate-4", issue="S2"),
                   _report("optimistic", "candidate-4", blocking=False)):
        with pytest.raises(ReviewError, match="needs_decision"):
            apply_review(state, report, candidate_digest="candidate-4")
    assert state["status"] == "needs_decision", "rejected transitions must not mutate snapshots"
    assert apply_review(state, untouched, candidate_digest="candidate-4") == state


def test_renaming_a_finding_cannot_reset_budget_and_stale_candidates_cannot_close_blockers():
    state = None
    for attempt in (1, 2):
        candidate = f"candidate-{attempt}"
        state = apply_review(state, _report(f"review-{attempt}", candidate), candidate_digest=candidate)
    with pytest.raises(ReviewError, match="candidate"):
        apply_review(state, _report("stale-pass", blocking=False), candidate_digest="candidate-2")
    next_report = _report("new-issue", "candidate-3", issue="S2")
    state = apply_review(state, next_report, candidate_digest="candidate-3")
    assert state["status"] == "needs_decision" and not allow_repair(state)
    assert state["issues"]["S1"]["resolved"] is True
    assert state["issues"]["S1"]["candidate_digests"] == ["candidate-1", "candidate-2"]
    assert state["issues"]["S2"]["candidate_digests"] == ["candidate-3"]
    assert state["candidate_digests"] == ["candidate-1", "candidate-2", "candidate-3"]
    assert not allow_repair(None)


def test_complete_current_snapshot_can_resolve_findings_without_erasing_history():
    first = apply_review(None, _report(), candidate_digest="candidate-1")
    clean = _report("all-closed", "candidate-2", blocking=False)
    clean["findings"] = []
    passed = apply_review(first, clean, candidate_digest="candidate-2")
    assert passed["status"] == "passed" and not allow_repair(passed)
    assert passed["issues"]["S1"]["resolved"] is True
    assert passed["issues"]["S1"]["candidate_digests"] == ["candidate-1"]
    assert first["issues"]["S1"]["resolved"] is False
    assert len(passed["history"]) == 2


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(passed=True),
    lambda r: r.update(passed=1),
    lambda r: r.update(extra="not part of the contract"),
    lambda r: r.pop("reviewer"),
    lambda r: r.update(request_id="  "),
    lambda r: r.update(candidate_digest=None),
    lambda r: r.update(findings={}),
    lambda r: r["findings"].append(copy.deepcopy(r["findings"][0])),
    lambda r: r["findings"][0].update(blocking="false"),
    lambda r: r["findings"][0].update(category="looks_fine"),
    lambda r: r["findings"][0].pop("evidence_ref"),
    lambda r: r["findings"][0].update(message=""),
    lambda r: r["findings"][0].update(resolved=True),
    lambda r: r.update(findings=[]),
])
def test_invalid_review_cannot_turn_into_an_accepted_verdict(mutation):
    report = _report()
    mutation(report)
    with pytest.raises(ReviewError):
        validate_review(report)


def test_report_normalization_is_detached_and_existing_blockers_are_not_downgraded_by_category():
    report = _report()
    report["request_id"] = " review-1 "
    report["findings"][0]["category"] = "preexisting"
    normalized = validate_review(report)
    assert normalized["request_id"] == "review-1"
    normalized["findings"][0]["message"] = "changed copy"
    assert report["findings"][0]["message"] != "changed copy"
    state = apply_review(None, report, candidate_digest="candidate-1")
    assert state["status"] == "changes_requested"
    forged = copy.deepcopy(state)
    forged["status"] = "passed"
    with pytest.raises(ReviewError, match="state"):
        apply_review(forged, _report("next", "candidate-2"), candidate_digest="candidate-2")
    assert not allow_repair(forged)


def test_repeated_same_candidate_reviews_have_bounded_history_but_exact_replays_still_work():
    state = None
    for index in range(128):
        state = apply_review(state, _report(f"request-{index}"), candidate_digest="candidate-1")
    assert state["candidate_digests"] == ["candidate-1"] and allow_repair(state)
    assert apply_review(state, _report("request-0"), candidate_digest="candidate-2") == state
    with pytest.raises(ReviewError, match="history"):
        apply_review(state, _report("overflow"), candidate_digest="candidate-1")
