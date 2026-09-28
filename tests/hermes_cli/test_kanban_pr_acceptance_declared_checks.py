"""PR acceptance when GitHub cannot supply the required-check set.

Private repositories on the free GitHub plan answer HTTP 403 ("Upgrade to GitHub
Pro or make this repository public") for branch protection and rulesets, while
exact-head check runs stay readable. The gate must (a) not report that 403 as an
infra/auth failure and (b) accept an operator-declared required set from
``kanban.pr_required_checks`` as the fallback source, never as an override.

``gh`` is replaced by a fake ``subprocess.run`` so the tests run on every host; the
declaration is read through the real config loader from a temp ``HERMES_HOME``.
"""
from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from hermes_cli import kanban_pr_acceptance as acc

_SHA = "a" * 40
_PR = "https://github.com/acme/widgets/pull/7"
_PLAN_403 = ('gh: Upgrade to GitHub Pro or make this repository public to enable this feature. '
             '(HTTP 403)')


def _run(name, conclusion="success", app_id=15368, run_id=42):
    return {"id": run_id, "name": name, "head_sha": _SHA, "app": {"id": app_id},
            "status": "completed", "conclusion": conclusion,
            "html_url": f"https://github.com/acme/widgets/actions/runs/{run_id}"}


def _routes(*, protection=(), rules_stderr=None, rules=(), check_runs=()):
    checks = [{"context": c, "app": {"databaseId": 15368}} for c in protection]
    graphql = {"data": {"repository": {"pullRequest": {
        "headRefOid": _SHA, "baseRefName": "main", "state": "OPEN",
        "baseRef": {"branchProtectionRule": {"requiredStatusChecks": checks} if checks else None}}}}}
    return {
        "graphql": {"stdout": graphql},
        "/rules/branches/": {"stderr": rules_stderr} if rules_stderr else {"stdout": [list(rules)]},
        "/check-runs": {"stdout": [{"total_count": len(check_runs), "check_runs": list(check_runs)}]},
        "/statuses": {"stdout": [[]]},
        "/pulls/": {"stdout": {"head": {"sha": _SHA}, "base": {"ref": "main"}, "state": "open"}},
    }


@pytest.fixture
def gh(monkeypatch):
    def install(routes):
        calls = []

        def fake_run(command, **kwargs):
            endpoint = command[2]
            calls.append(endpoint)
            for key, spec in routes.items():
                if key in endpoint:
                    if "stderr" in spec:
                        raise subprocess.CalledProcessError(1, command, output="", stderr=spec["stderr"])
                    return subprocess.CompletedProcess(command, 0, stdout=json.dumps(spec["stdout"]), stderr="")
            raise AssertionError(f"unexpected gh endpoint {endpoint!r}")

        monkeypatch.setattr(acc.subprocess, "run", fake_run)
        return calls
    return install


@pytest.fixture
def declare(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def write(mapping):
        (home / "config.yaml").write_text(yaml.safe_dump({"kanban": {"pr_required_checks": mapping}}))
    write({})
    return write


def test_plan_gated_repo_uses_declared_checks_at_exact_head(gh, declare):
    declare({"acme/widgets": ["Go Backend", {"context": "E2E Tests", "app_id": 15368}]})
    gh(_routes(rules_stderr=_PLAN_403, check_runs=[_run("Go Backend"), _run("E2E Tests", run_id=43)]))
    receipt = acc.collect_acceptance(_PR, None)
    assert receipt["ok"] and receipt["classification"] == "success"
    assert receipt["required_source"] == "declared"
    assert {r["context"] for r in receipt["required"]} == {"Go Backend", "E2E Tests"}
    assert {c["id"] for c in receipt["checks"]} == {42, 43}


def test_declared_checks_still_require_every_context_green(gh, declare):
    declare({"acme/widgets": ["Go Backend", "E2E Tests"]})
    gh(_routes(rules_stderr=_PLAN_403, check_runs=[_run("Go Backend")]))
    assert acc.collect_acceptance(_PR, None)["classification"] == "missing"
    gh(_routes(rules_stderr=_PLAN_403, check_runs=[_run("Go Backend"), _run("E2E Tests", "failure", run_id=43)]))
    assert acc.collect_acceptance(_PR, None)["classification"] == "failure"


def test_declared_app_pin_rejects_same_name_from_other_app(gh, declare):
    declare({"acme/widgets": [{"context": "Go Backend", "app_id": 15368}]})
    gh(_routes(rules_stderr=_PLAN_403, check_runs=[_run("Go Backend", app_id=99)]))
    receipt = acc.collect_acceptance(_PR, None)
    assert not receipt["ok"] and receipt["classification"] == "missing"


def test_plan_gate_without_declaration_is_not_infra(gh, declare):
    gh(_routes(rules_stderr=_PLAN_403, check_runs=[_run("Go Backend")]))
    receipt = acc.collect_acceptance(_PR, None)
    assert not receipt["ok"]
    assert receipt["classification"] == "plan_gated"
    assert "pr_required_checks" in receipt["detail"]


def test_github_required_set_wins_over_declaration(gh, declare):
    declare({"acme/widgets": ["Declared Only"]})
    gh(_routes(protection=["Go Backend"], check_runs=[_run("Go Backend")]))
    receipt = acc.collect_acceptance(_PR, None)
    assert receipt["ok"] and receipt["required_source"] == "github"
    assert [r["context"] for r in receipt["required"]] == ["Go Backend"]


def test_declaration_is_per_repository(gh, declare):
    declare({"acme/other": ["Go Backend"]})
    gh(_routes(rules_stderr=_PLAN_403, check_runs=[_run("Go Backend")]))
    assert acc.collect_acceptance(_PR, None)["classification"] == "plan_gated"


def test_non_plan_403_on_rulesets_remains_infra(gh, declare):
    declare({"acme/widgets": ["Go Backend"]})
    gh(_routes(rules_stderr="gh: Resource not accessible by integration (HTTP 403)",
               check_runs=[_run("Go Backend")]))
    assert acc.collect_acceptance(_PR, None)["classification"] == "infra"
