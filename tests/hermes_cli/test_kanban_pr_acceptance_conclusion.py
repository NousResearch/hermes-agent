"""A check run with a conclusion and a finish time is final, even if GitHub's status lags."""

import pytest

from hermes_cli import kanban_pr_acceptance as acc

SHA = "a" * 40
PR = "https://github.com/acme/repo/pull/7"
FINISHED = "2026-11-24T00:49:35Z"


def _collect(monkeypatch, status, conclusion, completed_at):
    run = {"id": 42, "name": "required", "head_sha": SHA, "app": {"id": 1}, "status": status,
           "conclusion": conclusion, "completed_at": completed_at}

    def fake_api(endpoint, *, query=None, paginate=False, profile_home=None):
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": SHA, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                    {"context": "required", "app": {"databaseId": 1}}]}}}}}}
        if "/check-runs" in endpoint:
            return [{"total_count": 1, "check_runs": [run]}]
        if "/rules/branches/" in endpoint or "/statuses" in endpoint:
            return [[]]
        return {"head": {"sha": SHA}, "base": {"ref": "main"}, "state": "open"}

    monkeypatch.setattr(acc, "_api", fake_api)
    return acc.collect_acceptance(PR, None)["classification"]


@pytest.mark.parametrize("status, conclusion, completed_at, expected", [
    ("in_progress", "success", FINISHED, "success"),
    ("in_progress", "failure", FINISHED, "failure"),
    ("in_progress", None, None, "pending"),
    ("queued", None, None, "pending"),
    ("in_progress", "success", None, "pending"),
    ("completed", "success", FINISHED, "success"),
])
def test_run_state_decides_acceptance(monkeypatch, status, conclusion, completed_at, expected):
    assert _collect(monkeypatch, status, conclusion, completed_at) == expected
