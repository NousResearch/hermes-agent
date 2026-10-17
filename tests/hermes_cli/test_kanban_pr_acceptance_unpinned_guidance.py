"""Ruleset-required checks: unpinned ones are refused with the fix, pinned ones are proven."""
from hermes_cli.kanban_pr_acceptance import collect_acceptance
from tests.hermes_cli.test_kanban_pr_acceptance_actions_fallback import (
    ACTIONS_APP, PR, SHA, _base_routes, _calls, _install_gh, _job, _page, _pr, _replace, _run,
)

CONTEXTS = ["Lint", "Classify changes", "Targeted tests",
            "Full suite (shard 1)", "Full suite (shard 2)", "Full suite (shard 3)"]


def _ruleset_routes(app_id):
    """No classic protection; one ruleset requires every check, as GitHub returns it."""
    pr = _pr()
    pr["data"]["repository"]["pullRequest"]["baseRef"]["branchProtectionRule"] = None
    checks = [{"context": c, **({"integration_id": app_id} if app_id is not None else {})} for c in CONTEXTS]
    rules = [[{"type": "required_status_checks", "parameters": {
        "strict_required_status_checks_policy": False, "required_status_checks": checks}}]]
    jobs = [_job(attempt=1, name=c, job_id=100 + i) for i, c in enumerate(CONTEXTS)]
    routes = _replace(_base_routes(), "graphql", body=pr)
    routes = _replace(routes, "/rules/branches/", body=rules)
    routes = _replace(routes, "/actions/runs?", body=[_page("workflow_runs", [_run(attempt=1)])])
    return _replace(routes, "/jobs", body=[_page("jobs", jobs)])


def test_unpinned_ruleset_refusal_names_the_configuration_fix(tmp_path, monkeypatch):
    log = _install_gh(tmp_path, monkeypatch, _ruleset_routes(None))
    receipt = collect_acceptance(PR, PR)
    assert receipt["ok"] is not True
    assert all(context in receipt["detail"] for context in sorted(CONTEXTS, key=str)[:3])
    assert "and 3 more" in receipt["detail"]
    assert receipt["detail"].index(str(ACTIONS_APP)) < receipt["detail"].index("Checks:")
    assert "unpinned" in receipt["detail"]
    assert str(ACTIONS_APP) in receipt["detail"]
    assert "ruleset" in receipt["detail"]
    assert "retry" not in receipt["recovery"].lower()
    assert not any("/actions/runs" in call for call in _calls(log))


def test_foreign_app_requirement_is_named_as_another_app(tmp_path, monkeypatch):
    routes = _ruleset_routes(ACTIONS_APP)
    rules = routes[[r["match"] for r in routes].index("/rules/branches/")]["body"]
    rules[0][0]["parameters"]["required_status_checks"].append({"context": "codecov", "integration_id": 222})
    log = _install_gh(tmp_path, monkeypatch, routes)
    receipt = collect_acceptance(PR, PR)
    assert receipt["ok"] is not True
    assert "codecov (app 222)" in receipt["detail"]
    assert "Lint" not in receipt["detail"]
    assert not any("/actions/runs" in call for call in _calls(log))


def test_pinning_the_ruleset_to_actions_proves_the_exact_head(tmp_path, monkeypatch):
    _install_gh(tmp_path, monkeypatch, _ruleset_routes(ACTIONS_APP))
    receipt = collect_acceptance(PR, PR)
    assert receipt["ok"] is True, receipt.get("detail")
    assert receipt["head_sha"] == SHA
    assert sorted(check["name"] for check in receipt["checks"]) == sorted(CONTEXTS)
    assert {check["classification"] for check in receipt["checks"]} == {"success"}
