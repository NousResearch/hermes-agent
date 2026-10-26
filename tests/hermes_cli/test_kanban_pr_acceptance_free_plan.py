"""GitHub Free private repos return 403 on rulesets. That must not jam PR acceptance.

Classic branch protection (the GraphQL read) still governs. A 401/404 on the
rules endpoint, a 403 on any other endpoint, and 5xx/network failures stay
failures. Only the rulesets 403 falls back to check-runs and statuses on the
head SHA. Skipped and neutral checks are not votes.
"""
import json
import os
import sys

import pytest

from hermes_cli.kanban_pr_acceptance import collect_acceptance

PR = "https://github.com/acme/repo/pull/7"
SHA = "a" * 40


def _graphql(protection=None, state="MERGED"):
    return {"data": {"repository": {"pullRequest": {
        "headRefOid": SHA,
        "baseRefName": "main",
        "state": state,
        "baseRef": {"branchProtectionRule": protection},
    }}}}


def _run(name, conclusion, check_id, app_id=1):
    return {
        "id": check_id,
        "name": name,
        "head_sha": SHA,
        "app": {"id": app_id},
        "status": "completed",
        "conclusion": conclusion,
        "html_url": f"https://example/{name}",
    }


def _pages(*runs):
    return [{"total_count": len(runs), "check_runs": list(runs)}]


_MERGED = {"head": {"sha": SHA}, "base": {"ref": "main"}, "state": "closed", "merged": True}


def _drive(tmp_path, monkeypatch, scenario):
    scenario_path = tmp_path / "scenario.json"
    log = tmp_path / "log"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    gh = shim / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"scenario = json.loads(open({str(scenario_path)!r}, encoding='utf-8').read())\n"
        f"open({str(log)!r}, 'a', encoding='utf-8').write(sys.argv[2].split('?')[0] + '\\n')\n"
        "endpoint = sys.argv[2]\n"
        "def deny(status):\n"
        "    sys.stderr.write(f'gh: refused (HTTP {status})\\n')\n"
        "    raise SystemExit(1)\n"
        "def emit(item):\n"
        "    if isinstance(item, int):\n"
        "        deny(item)\n"
        "    print(json.dumps(item))\n"
        "    raise SystemExit(0)\n"
        "if endpoint == 'graphql':\n"
        "    emit(scenario['graphql'])\n"
        "if '/rules/branches/' in endpoint:\n"
        "    emit(scenario['rules'])\n"
        "if '/check-runs' in endpoint:\n"
        "    emit(scenario['checks'])\n"
        "if '/statuses' in endpoint:\n"
        "    emit(scenario.get('statuses', [[]]))\n"
        "if '/pulls/' in endpoint:\n"
        "    emit(scenario['pulls'])\n"
        "sys.stderr.write('unexpected endpoint\\n')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ.get("PATH", ""))
    receipt = collect_acceptance(PR, PR)
    requests = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return receipt, requests


@pytest.mark.platforms("posix")
def test_rulesets_403_resolves_from_head_checks_and_ignores_skipped(tmp_path, monkeypatch):
    receipt, requests = _drive(tmp_path, monkeypatch, {
        "graphql": _graphql(),
        "rules": 403,
        "checks": _pages(
            _run("test", "success", 1),
            _run("build", "success", 2),
            _run("deploy-dev", "skipped", 3),
            _run("deploy-prod", "skipped", 4),
        ),
        "pulls": _MERGED,
    })
    assert receipt["ok"] is True
    assert receipt["classification"] == "success"
    assert receipt["head_sha"] == SHA
    assert "HTTP 403" in receipt["detail"]
    assert "check-runs" in receipt["detail"]
    by_name = {item["name"]: item["classification"] for item in receipt["checks"]}
    assert by_name["test"] == "success"
    assert by_name["deploy-dev"] == "skipped"
    assert any("/check-runs" in path for path in requests)


@pytest.mark.platforms("posix")
def test_rulesets_403_failing_check_is_failure_not_infra(tmp_path, monkeypatch):
    receipt, _ = _drive(tmp_path, monkeypatch, {
        "graphql": _graphql(),
        "rules": 403,
        "checks": _pages(_run("test", "failure", 1), _run("build", "success", 2)),
        "pulls": _MERGED,
    })
    assert receipt["ok"] is False
    assert receipt["classification"] == "failure"


@pytest.mark.platforms("posix")
def test_rulesets_403_still_honors_classic_branch_protection(tmp_path, monkeypatch):
    receipt, _ = _drive(tmp_path, monkeypatch, {
        "graphql": _graphql(protection={
            "requiredStatusChecks": [{"context": "test", "app": {"databaseId": 1}}],
        }),
        "rules": 403,
        "checks": _pages(_run("test", "failure", 1), _run("build", "success", 2)),
        "pulls": _MERGED,
    })
    assert receipt["ok"] is False
    assert receipt["classification"] == "failure"
    assert not (receipt.get("detail") or "").startswith("Rulesets API")
    assert [item["name"] for item in receipt["checks"]] == ["test"]


@pytest.mark.platforms("posix")
def test_successful_rulesets_read_with_no_required_checks_does_not_fall_back(tmp_path, monkeypatch):
    receipt, requests = _drive(tmp_path, monkeypatch, {
        "graphql": _graphql(protection=None),
        "rules": [[]],
        "checks": _pages(_run("test", "success", 1)),
        "pulls": _MERGED,
    })
    assert receipt["ok"] is False
    assert receipt["classification"] == "missing"
    assert "local-only" in receipt["detail"]
    assert not any("/check-runs" in path for path in requests)


@pytest.mark.platforms("posix")
def test_rulesets_401_stays_auth_and_500_stays_infra(tmp_path, monkeypatch):
    auth, _ = _drive(tmp_path, monkeypatch, {
        "graphql": _graphql(),
        "rules": 401,
        "checks": _pages(_run("test", "success", 1)),
        "pulls": _MERGED,
    })
    assert auth["classification"] == "auth"
    assert "HTTP 401" in auth["detail"]
    infra, _ = _drive(tmp_path, monkeypatch, {
        "graphql": _graphql(),
        "rules": 500,
        "checks": _pages(_run("test", "success", 1)),
        "pulls": _MERGED,
    })
    assert infra["ok"] is False
    assert infra["classification"] == "infra"


@pytest.mark.platforms("posix")
def test_graphql_403_is_not_the_rulesets_fallback(tmp_path, monkeypatch):
    receipt, requests = _drive(tmp_path, monkeypatch, {
        "graphql": 403,
        "rules": 403,
        "checks": _pages(_run("test", "success", 1)),
        "pulls": _MERGED,
    })
    assert receipt["classification"] == "auth"
    assert "graphql" in receipt["detail"]
    assert not any("/check-runs" in path for path in requests)


@pytest.mark.platforms("posix")
def test_rulesets_403_with_only_skipped_checks_is_missing(tmp_path, monkeypatch):
    receipt, _ = _drive(tmp_path, monkeypatch, {
        "graphql": _graphql(),
        "rules": 403,
        "checks": _pages(_run("deploy-dev", "skipped", 3)),
        "pulls": _MERGED,
    })
    assert receipt["ok"] is False
    assert receipt["classification"] == "missing"
