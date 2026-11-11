"""Plan-limited rule reads require nonempty green exact-head checks."""
import subprocess

import pytest

from hermes_cli import kanban_pr_acceptance as acc

SHA = "a" * 40
PR = "https://github.com/example/repo/pull/47"
PLAN_403 = ("gh: Upgrade to GitHub Pro or make this repository public to enable this feature. (HTTP 403)\n")


def _fake_github(monkeypatch, *, runs, statuses=(), rules_error=PLAN_403, protection=None, other_403=None):
    def run(command, **kwargs):
        endpoint = command[2]
        def fail(stderr):
            raise subprocess.CalledProcessError(1, command, output="", stderr=stderr)
        if other_403 and other_403 in endpoint:
            fail("gh: Resource not accessible by integration (HTTP 403)")
        if endpoint == "graphql":
            body = {"data": {"repository": {"pullRequest": {
                "headRefOid": SHA, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": protection}}}}}
        elif "/rules/branches/" in endpoint:
            if rules_error:
                fail(rules_error)
            body = [[]]
        elif "/check-runs" in endpoint:
            body = [{"total_count": len(runs), "check_runs": list(runs)}]
        elif "/statuses" in endpoint:
            body = [list(statuses)]
        elif "/pulls/" in endpoint:
            body = {"head": {"sha": SHA}, "base": {"ref": "main"}, "state": "open"}
        else:
            raise AssertionError(endpoint)
        import json
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(body), stderr="")
    monkeypatch.setattr(acc.subprocess, "run", run)
    monkeypatch.setattr(acc, "_gh_env", lambda home: None)


def _run(name, conclusion="success", status="completed", sha=SHA, app=15368):
    return {"id": hash((name, sha)) & 0xFFFF, "name": name, "head_sha": sha, "app": {"id": app},
            "status": status, "conclusion": conclusion, "html_url": f"https://x/{name}"}


def test_plan_limited_repo_accepts_when_every_head_check_is_green(monkeypatch):
    _fake_github(monkeypatch, runs=[_run("test"), _run("lint"), _run("docs", conclusion="skipped")])
    receipt = acc.collect_acceptance("example/repo", PR)
    assert receipt["ok"] and receipt["classification"] == "success"
    assert "plan" in receipt["required_source"]
    assert {c["context"] for c in receipt["required"]} == {"test", "lint"}


@pytest.mark.parametrize("bad,expected", [
    (_run("lint", conclusion="failure"), "failure"),
    (_run("lint", conclusion=None, status="in_progress"), "pending"),
    (_run("lint", sha="b" * 40), "stale"),
    (_run("lint", conclusion="cancelled"), "infra"),
])
def test_plan_limited_repo_still_blocks_on_any_non_green_head_check(monkeypatch, bad, expected):
    _fake_github(monkeypatch, runs=[_run("test"), bad])
    receipt = acc.collect_acceptance("example/repo", PR)
    assert not receipt["ok"] and receipt["classification"] == expected


def test_plan_limited_repo_with_no_checks_is_not_accepted(monkeypatch):
    _fake_github(monkeypatch, runs=[])
    receipt = acc.collect_acceptance("example/repo", PR)
    assert not receipt["ok"] and receipt["classification"] == "missing"
    assert "no checks" in receipt["detail"]


def test_legacy_commit_statuses_count_as_head_checks(monkeypatch):
    _fake_github(monkeypatch, runs=[_run("test")],
                 statuses=[{"id": 9, "context": "ci/legacy", "state": "failure", "target_url": "u"}])
    receipt = acc.collect_acceptance("example/repo", PR)
    assert not receipt["ok"] and receipt["classification"] == "failure"


def test_other_403s_remain_auth_failures(monkeypatch):
    _fake_github(monkeypatch, runs=[_run("test")], rules_error="gh: Resource not accessible by integration (HTTP 403)")
    assert acc.collect_acceptance("example/repo", PR)["classification"] == "auth"
    _fake_github(monkeypatch, runs=[_run("test")], other_403="/check-runs")
    assert acc.collect_acceptance("example/repo", PR)["classification"] == "auth"


def test_plan_limit_text_on_a_non_rules_read_is_not_tolerated(monkeypatch):
    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output="", stderr=PLAN_403)
    monkeypatch.setattr(acc.subprocess, "run", run)
    monkeypatch.setattr(acc, "_gh_env", lambda home: None)
    with pytest.raises(acc._GatePlanRestricted):
        acc._api("graphql", query="{}")
    assert acc.collect_acceptance("example/repo", PR)["classification"] == "auth"


def test_readable_rules_keep_the_original_contract(monkeypatch):
    # Rules readable and empty, no branch protection: unchanged "no required checks" refusal.
    _fake_github(monkeypatch, runs=[_run("test")], rules_error=None)
    receipt = acc.collect_acceptance("example/repo", PR)
    assert not receipt["ok"] and "No repository-required checks" in receipt["detail"]
    assert "required_source" not in receipt


def test_plan_limited_but_protection_declares_checks_uses_them(monkeypatch):
    protection = {"requiredStatusChecks": [{"context": "test", "app": {"databaseId": 15368}}]}
    _fake_github(monkeypatch, runs=[_run("test"), _run("flaky", conclusion="failure")], protection=protection)
    receipt = acc.collect_acceptance("example/repo", PR)
    assert receipt["ok"] and "required_source" not in receipt
