"""A skipped/neutral required check satisfies acceptance only when its own
check suite concluded success/skipped/neutral (#140025, review narrowing).

A job skipped by design (path filter, ``if:`` false) sits in a suite that
concluded ``success`` (or ``skipped`` when every job in it skipped). A job
skipped because a job it ``needs`` failed also reports ``skipped``, but its
suite concludes ``failure`` — accepting that unconditionally would mark a card
done on a head whose build failed. The suite is read only when a required run
actually skipped, so success/failure runs pay no extra API call.
"""
import pytest

from hermes_cli import kanban_pr_acceptance as kp

SHA = "a" * 40


def _collect(monkeypatch, run_conclusion, suite_conclusion=None, suite_status="completed"):
    calls = []

    def fake_api(endpoint, *, query=None, paginate=False, profile_home=None):
        calls.append(endpoint.split("?")[0])
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": SHA, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                    {"context": "required", "app": {"databaseId": 1}}]}}}}}}
        if "/rules/branches/" in endpoint:
            return [[]]
        if "/check-suites" in endpoint:
            return [{"total_count": 1, "check_suites": [
                {"id": 7, "status": suite_status, "conclusion": suite_conclusion}]}]
        if "/check-runs" in endpoint:
            run = {"id": 42, "name": "required", "head_sha": SHA, "app": {"id": 1},
                   "status": "completed", "conclusion": run_conclusion,
                   "check_suite": {"id": 7}}
            return [{"total_count": 1, "check_runs": [run]}]
        if "/statuses" in endpoint:
            return [[]]
        if "/pulls/" in endpoint:
            return {"head": {"sha": SHA}, "base": {"ref": "main"}, "state": "open"}
        raise AssertionError(f"unexpected endpoint {endpoint}")

    monkeypatch.setattr(kp, "_api", fake_api)
    return kp.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7"), calls


@pytest.mark.parametrize("run_conclusion,suite_conclusion,suite_status,expected,reads_suites", [
    # A run skipped by design sits in a suite that passed.
    ("skipped", "success", "completed", "success", True),
    ("skipped", "skipped", "completed", "success", True),
    ("neutral", "success", "completed", "success", True),
    ("neutral", "skipped", "completed", "success", True),
    # A run skipped because a job it `needs` failed sits in a failing suite.
    ("skipped", "failure", "completed", "infra", True),
    ("skipped", "cancelled", "completed", "infra", True),
    # An unfinished suite read stays infra, fail-closed like any other partial evidence.
    ("skipped", "success", "in_progress", "infra", True),
    # Other run outcomes never pay the extra check-suite read.
    ("success", None, None, "success", False),
    ("failure", None, None, "failure", False),
    ("cancelled", None, None, "infra", False),
])
def test_skipped_or_neutral_requires_a_passing_check_suite(
        monkeypatch, run_conclusion, suite_conclusion, suite_status, expected, reads_suites):
    receipt, calls = _collect(monkeypatch, run_conclusion, suite_conclusion, suite_status)
    assert receipt["classification"] == expected
    assert receipt["ok"] is (expected == "success")
    # The receipt keeps the raw conclusion either way.
    assert receipt["checks"][0]["conclusion"] == run_conclusion
    assert (f"repos/acme/repo/commits/{SHA}/check-suites" in calls) is reads_suites
