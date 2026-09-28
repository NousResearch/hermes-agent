from hermes_cli import kanban_pr_acceptance as acceptance


PR = "https://github.com/acme/repo/pull/7"
SHA = "a" * 40


def _receipt(ci_classification="not_required", check_run_ids=None):
    return {
        "head_sha": SHA,
        "result": "passed",
        "steps": [
            {"name": "fmt", "result": "passed"},
            {"name": "lint", "result": "passed"},
            {"name": "tests", "result": "passed"},
        ],
        "private_database": True,
        "independent_review": {
            "status": "approved", "reviewer": "reviewer", "task_id": "t_deadbeef",
            "head_sha": SHA,
        },
        "approved_by": "meetitcoordinator",
        "ci": {"classification": ci_classification, "check_run_ids": check_run_ids or []},
    }


def _api_stub(monkeypatch, *, checks=None, statuses=None, jobs=None):
    checks = checks or []
    statuses = statuses or []
    jobs = jobs or []
    requested = []

    def fake_api(endpoint, *, query=None, paginate=False):
        requested.append(endpoint)
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": SHA, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": {"requiredStatusChecks": []}},
            }}}}
        if "/rules/branches/" in endpoint:
            return [[]]
        if "/check-runs" in endpoint:
            return [{"total_count": len(checks), "check_runs": checks}]
        if "/statuses" in endpoint:
            return statuses
        if "/actions/runs/" in endpoint and "/jobs" in endpoint:
            return [{"total_count": len(jobs), "jobs": jobs}]
        if "/pulls/" in endpoint:
            return {"head": {"sha": SHA}, "base": {"ref": "main"}, "state": "open"}
        raise AssertionError(endpoint)

    monkeypatch.setattr(acceptance, "_api", fake_api)
    return requested


def test_local_gate_can_complete_when_hosted_checks_are_not_configured(monkeypatch):
    requested = _api_stub(monkeypatch)
    receipt = acceptance.collect_acceptance(
        "acme/repo", PR, local_gate=_receipt(), coordinator="meetitcoordinator",
    )
    assert receipt["ok"] is True
    assert receipt["classification"] == "local_gate"
    assert receipt["head_sha"] == SHA
    assert receipt["local_gate"]["reviewed_head_sha"] == SHA
    assert "/check-runs" in " ".join(requested)


def test_local_gate_is_bound_to_exact_head_and_independent_coordinator_review(monkeypatch):
    _api_stub(monkeypatch)
    wrong_head = _receipt()
    wrong_head["head_sha"] = "b" * 40
    receipt = acceptance.collect_acceptance(
        "acme/repo", PR, local_gate=wrong_head, coordinator="meetitcoordinator",
    )
    assert receipt["ok"] is False

    wrong_review_head = _receipt()
    wrong_review_head["independent_review"]["head_sha"] = "b" * 40
    receipt = acceptance.collect_acceptance(
        "acme/repo", PR, local_gate=wrong_review_head, coordinator="meetitcoordinator",
    )
    assert receipt["ok"] is False

    same_reviewer = _receipt()
    same_reviewer["independent_review"]["reviewer"] = "meetitcoordinator"
    receipt = acceptance.collect_acceptance(
        "acme/repo", PR, local_gate=same_reviewer, coordinator="meetitcoordinator",
    )
    assert receipt["ok"] is False

    receipt = acceptance.collect_acceptance(
        "acme/repo", PR, local_gate=_receipt(), coordinator="another-profile",
    )
    assert receipt["ok"] is False


def test_local_gate_accepts_only_verified_pre_runner_infrastructure(monkeypatch):
    failed = {
        "id": 42, "name": "CI", "head_sha": SHA, "app": {"id": 1},
        "status": "completed", "conclusion": "failure",
        "details_url": "https://github.com/acme/repo/actions/runs/35983742702",
    }
    requested = _api_stub(monkeypatch, checks=[failed], jobs=[{
        "id": 8, "conclusion": "failure", "runner_id": 0, "steps": [],
    }])
    local_gate = _receipt("pre_runner_infrastructure", [42])
    receipt = acceptance.collect_acceptance(
        "acme/repo", PR, local_gate=local_gate, coordinator="meetitcoordinator",
    )
    assert receipt["ok"] is True
    assert receipt["classification"] == "local_gate"
    assert any("actions/runs/35983742702/jobs" in path for path in requested)

    _api_stub(monkeypatch, checks=[failed], jobs=[{
        "id": 8, "conclusion": "failure", "runner_id": 987, "steps": [
            {"name": "tests", "conclusion": "failure"},
        ],
    }])
    receipt = acceptance.collect_acceptance(
        "acme/repo", PR, local_gate=local_gate, coordinator="meetitcoordinator",
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "failure"


def test_local_gate_does_not_override_pending_or_unrelated_failed_checks(monkeypatch):
    pending = {
        "id": 42, "name": "CI", "head_sha": SHA, "app": {"id": 1},
        "status": "in_progress", "conclusion": None,
        "details_url": "https://github.com/acme/repo/actions/runs/35983742702",
    }
    _api_stub(monkeypatch, checks=[pending])
    receipt = acceptance.collect_acceptance(
        "acme/repo", PR, local_gate=_receipt("pre_runner_infrastructure", [42]),
        coordinator="meetitcoordinator",
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "pending"

    failed = {**pending, "id": 43, "status": "completed", "conclusion": "failure"}
    _api_stub(monkeypatch, checks=[failed], jobs=[{
        "id": 8, "conclusion": "failure", "runner_id": 0, "steps": [],
    }])
    receipt = acceptance.collect_acceptance(
        "acme/repo", PR, local_gate=_receipt("pre_runner_infrastructure", [42]),
        coordinator="meetitcoordinator",
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "failure"
