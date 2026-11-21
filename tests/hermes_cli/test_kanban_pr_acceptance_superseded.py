"""PR acceptance judges the newest same-name check run per app (#122688).

A workflow re-triggered on the same head (draft -> ready under
``cancel-in-progress``) leaves a cancelled run from the superseded suite next
to the replacement; ``filter=latest`` keeps both.
"""

import pytest

from hermes_cli import kanban_pr_acceptance as acc

SHA = "a" * 40
PR = "https://github.com/acme/repo/pull/7"


def _run(run_id, conclusion, started_at, app_id=1, status="completed"):
    return {"id": run_id, "name": "required", "head_sha": SHA, "app": {"id": app_id},
            "status": status, "conclusion": conclusion, "started_at": started_at}


def _collect(monkeypatch, runs, required_app=1):
    def fake_api(endpoint, *, query=None, paginate=False, profile_home=None):
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": SHA, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                    {"context": "required", "app": {"databaseId": required_app}}]}}}}}}
        if "/check-runs" in endpoint:
            return [{"total_count": len(runs), "check_runs": runs}]
        if "/rules/branches/" in endpoint or "/statuses" in endpoint:
            return [[]]
        return {"head": {"sha": SHA}, "base": {"ref": "main"}, "state": "open"}

    monkeypatch.setattr(acc, "_api", fake_api)
    return acc.collect_acceptance(PR, None)


@pytest.mark.parametrize("reverse", [False, True])
def test_newer_success_supersedes_cancelled_run(monkeypatch, reverse):
    runs = [_run(10, "cancelled", "2026-09-25T14:00:00Z"), _run(11, "success", "2026-09-25T14:01:00Z")]
    receipt = _collect(monkeypatch, runs[::-1] if reverse else runs)
    assert receipt["classification"] == "success" and receipt["ok"]
    assert [c["id"] for c in receipt["checks"]] == [11]


def test_newer_failure_still_decides(monkeypatch):
    runs = [_run(10, "success", "2026-09-25T14:00:00Z"), _run(11, "failure", "2026-09-25T14:01:00Z")]
    assert _collect(monkeypatch, runs)["classification"] == "failure"


def test_queued_replacement_without_start_time_is_newest(monkeypatch):
    runs = [_run(10, "success", "2026-09-25T14:00:00Z"), _run(11, None, None, status="queued")]
    assert _collect(monkeypatch, runs)["classification"] == "pending"


def test_unpinned_context_needs_every_apps_newest_run(monkeypatch):
    runs = [_run(10, "failure", "2026-09-25T14:00:00Z", app_id=1),
            _run(11, "success", "2026-09-25T14:01:00Z", app_id=2)]
    assert _collect(monkeypatch, runs, required_app=None)["classification"] == "failure"
