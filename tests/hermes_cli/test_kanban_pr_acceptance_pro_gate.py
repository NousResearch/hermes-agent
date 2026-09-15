"""Portable unit tests: a GitHub-plan 403 on the branch-rulesets endpoint must not
be misclassified as an infra failure.

Private repositories on the free plan cannot inspect branch rulesets: the REST
endpoint returns HTTP 403 ("Upgrade to GitHub Pro or make this repository public").
`gh api` surfaces that as a non-zero exit, which the broad except-clause in
`collect_acceptance` previously turned into `classification="infra"`, blocking the
terminal write of every private-repo PR-contract release card even though GraphQL
branch protection and the check-run/status evidence were readable.

These tests drive `collect_acceptance` with a fake `subprocess.run` so they run on
any platform (the existing acceptance suite is linux_only via a bash shim).
"""
from __future__ import annotations

import json
import subprocess

import pytest

from hermes_cli import kanban_pr_acceptance as acc


_SHA = "a" * 40


class _FakeRun:
    """Route `gh api <endpoint>` to canned responses keyed by endpoint substring."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        endpoint = command[2] if len(command) > 2 else ""
        for key, spec in self.responses.items():
            if key in endpoint:
                rc = spec.get("returncode", 0)
                if rc != 0:
                    # Mirror real subprocess.run(check=True): a non-zero gh exit raises.
                    raise subprocess.CalledProcessError(
                        rc, command, output="", stderr=spec.get("stderr", ""))
                return subprocess.CompletedProcess(
                    command, 0, stdout=json.dumps(spec["stdout"]), stderr="")
        raise AssertionError(f"unexpected gh api endpoint: {endpoint!r}")


def _graphql(protection_required):
    checks = [{"context": c, "app": {"databaseId": 1}} for c in protection_required]
    return {"stdout": {"data": {"repository": {"pullRequest": {
        "headRefOid": _SHA, "baseRefName": "main", "state": "OPEN",
        "baseRef": {"branchProtectionRule": {"requiredStatusChecks": checks}}}}}}}


def _check_runs(runs):
    return {"stdout": [{"total_count": len(runs), "check_runs": runs}]}


def _required_run(name="verify", conclusion="success"):
    return {"id": 42, "name": name, "head_sha": _SHA, "app": {"id": 1},
            "status": "completed", "conclusion": conclusion,
            "html_url": "https://github.com/acme/repo/actions/runs/42"}


def _pulls_reread():
    return {"stdout": {"head": {"sha": _SHA}, "base": {"ref": "main"}, "state": "open"}}


@pytest.fixture
def fake_run(monkeypatch):
    def install(responses):
        fr = _FakeRun(responses)
        monkeypatch.setattr(acc.subprocess, "run", fr)
        return fr
    return install


def test_rulesets_pro_gate_403_degrades_not_infra(fake_run):
    """protection=null + rulesets 403 (Pro-gate) -> honest 'use local-only', never infra."""
    fake_run({
        "graphql": _graphql([]),
        "/rules/branches/": {"returncode": 1,
                             "stderr": "gh: Upgrade to GitHub Pro or make this repository "
                                       "public to enable this feature. (HTTP 403)"},
    })
    receipt = acc.collect_acceptance("https://github.com/acme/repo/pull/7", None)
    assert receipt["classification"] != "infra", receipt
    assert receipt["ok"] is False
    assert "local-only" in receipt["detail"]


def test_rulesets_pro_gate_403_keeps_branch_protection_requirements(fake_run):
    """A Pro-gate 403 on rulesets must not drop real GraphQL branch-protection checks."""
    fake_run({
        "graphql": _graphql(["verify"]),
        "/rules/branches/": {"returncode": 1, "stderr": "(HTTP 403) Upgrade to GitHub Pro"},
        "/check-runs": _check_runs([_required_run("verify", "success")]),
        "/statuses": {"stdout": [[]]},
        "/pulls/": _pulls_reread(),
    })
    receipt = acc.collect_acceptance("https://github.com/acme/repo/pull/7", None)
    assert receipt["classification"] == "success", receipt
    assert receipt["ok"] is True
    assert {c["name"] for c in receipt["checks"]} == {"verify"}


def test_genuine_network_error_on_rulesets_still_infra(fake_run):
    """A real transport failure (not a plan gate) must still surface as infra."""
    fake_run({
        "graphql": _graphql([]),
        "/rules/branches/": {"returncode": 1,
                             "stderr": "dial tcp: lookup api.github.com: no such host"},
    })
    receipt = acc.collect_acceptance("https://github.com/acme/repo/pull/7", None)
    assert receipt["classification"] == "infra", receipt
    assert receipt["ok"] is False
