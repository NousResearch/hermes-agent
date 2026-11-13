"""Plan-limited rules discovery must not turn into auth advice or a CI bypass."""
import json
import subprocess

import pytest

from hermes_cli import kanban_pr_acceptance as acceptance


PLAN_MESSAGE = "Upgrade to GitHub Pro or make this repository public to enable this feature."
SHA = "a" * 40


@pytest.mark.parametrize("required,conclusion,expected", [
    (True, "success", "success"),
    (True, "failure", "failure"),
    (True, "pending", "pending"),
    (True, "stale", "stale"),
    (True, "missing", "missing"),
    (True, "head_change", "stale"),
    (False, "success", "missing"),
    (False, "missing", "missing"),
])
@pytest.mark.parametrize("slurp", [False, True])
def test_plan_unavailable_rules_preserve_required_check_gate(monkeypatch, required, conclusion, expected, slurp):
    """Exercise the real API error parser and collector at the gh transport seam."""
    def run(command, **kwargs):
        endpoint = command[2]
        if endpoint == "graphql":
            checks = [{"context": "CI", "app": {"databaseId": 1}}] if required else []
            value = {"data": {"repository": {"pullRequest": {
                "headRefOid": SHA, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": {"requiredStatusChecks": checks}},
            }}}}
        elif "/rules/branches/" in endpoint:
            body = {"message": PLAN_MESSAGE, "status": "403"}
            raise subprocess.CalledProcessError(1, command,
                output=json.dumps([body] if slurp else body),
                stderr=f"gh: {PLAN_MESSAGE} (HTTP 403)")
        elif "/check-runs" in endpoint:
            runs = [] if conclusion == "missing" else [{
                "id": 1, "name": "CI", "app": {"id": 1},
                "head_sha": "b" * 40 if conclusion == "stale" else SHA,
                "status": "in_progress" if conclusion == "pending" else "completed",
                "conclusion": "success" if conclusion in {"head_change", "stale"} else conclusion,
            }]
            value = [{"total_count": len(runs), "check_runs": runs}]
        elif "/statuses" in endpoint:
            value = [[]]
        elif "/pulls/" in endpoint:
            value = {"head": {"sha": "b" * 40 if conclusion == "head_change" else SHA},
                     "base": {"ref": "main"}, "state": "open"}
        else:
            raise AssertionError(endpoint)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(value))

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    receipt = acceptance.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["classification"] == expected
    assert receipt["ok"] is (expected == "success")
    assert receipt["rules_availability"] == "unavailable_plan"
    if not required:
        assert receipt["required"] == []
        assert "No repository-required checks" in receipt["detail"]
        assert "credentials" not in receipt["detail"]


@pytest.mark.parametrize("endpoint,status,body", [
    ("repos/acme/repo/rules/branches/main", 403, {"message": "Resource not accessible by integration"}),
    ("repos/acme/repo/rules/branches/main", 403, {"message": "API rate limit exceeded"}),
    ("repos/acme/repo/rules/branches/main", 403, {"message": PLAN_MESSAGE + " Unexpected"}),
    ("repos/acme/repo/rules/branches/main", 403, "malformed"),
    ("repos/acme/repo/rules/branches/main", 401, {"message": PLAN_MESSAGE}),
    ("repos/acme/repo/rules/branches/main", 404, {"message": PLAN_MESSAGE}),
    ("repos/acme/repo/commits/abc/check-runs", 403, {"message": PLAN_MESSAGE}),
])
def test_other_denials_never_become_empty_rules(monkeypatch, endpoint, status, body):
    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command,
            output=json.dumps(body), stderr=f"gh: {PLAN_MESSAGE} (HTTP {status})")

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    with pytest.raises(acceptance._GateAuthError):
        acceptance._api(endpoint)
