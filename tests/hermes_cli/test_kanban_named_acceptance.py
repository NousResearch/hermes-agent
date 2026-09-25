"""Scoped policy must bind jobs, workflow, app, attempt and tested tree."""
import copy
import json
import subprocess

import pytest

from hermes_cli import kanban_pr_acceptance as gate

REPO = "acme/private"
URL = f"https://github.com/{REPO}/pull/3"
HEAD, MERGE, TREE, BLOB = (x * 40 for x in "abcd")
POLICY = {"workflow_id": 12, "workflow_path": ".github/workflows/ci.yml",
          "workflow_blob_sha": BLOB, "app_id": 15368,
          "required_checks": ["tests", "ci ok"], "checkout_checks": ["tests"]}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({"kanban": {"pr_acceptance_policies": {REPO: POLICY}}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    pr = {"number": 3, "state": "open", "merged": False, "merge_commit_sha": MERGE,
          "head": {"sha": HEAD, "repo": {"full_name": REPO}},
          "base": {"sha": "e" * 40, "ref": "main", "repo": {"full_name": REPO}}}
    run = {"id": 42, "run_attempt": 1, "head_sha": HEAD, "event": "pull_request",
           "workflow_id": 12, "path": POLICY["workflow_path"], "check_suite_id": 91,
           "repository": {"full_name": REPO}, "head_repository": {"full_name": REPO},
           "status": "completed", "conclusion": "success",
           "pull_requests": [{"number": 3, "url": f"https://api.github.com/repos/{REPO}/pulls/3",
                              "head": {"sha": HEAD}, "base": {"sha": "e" * 40}}]}
    jobs = [{"id": i, "name": name, "run_id": 42, "run_attempt": 1,
             "head_sha": HEAD, "status": "completed", "conclusion": "success"}
            for i, name in enumerate(POLICY["required_checks"], 100)]
    state = {"pr": pr, "run": run, "jobs": jobs, "tree": TREE, "blob": BLOB,
             "app": 15368, "log": f"2026-09-25T19:00:00Z [command]/usr/bin/git log -1 --format=%H\n2026-09-25T19:00:01Z {MERGE}\n",
             "requests": []}

    def api(endpoint, **kwargs):
        state["requests"].append(endpoint)
        if state.get("error"):
            raise subprocess.CalledProcessError(1, "gh")
        if endpoint.endswith("/pulls/3"):
            value = copy.deepcopy(state["pr"])
            if state.get("race") and state["requests"].count(endpoint) > 1:
                value["head"]["sha"] = "f" * 40
            return value
        if "/contents/" in endpoint:
            return {"sha": evidence_blob if (evidence_blob := state.get("merge_blob")) and endpoint.endswith(MERGE) else state["blob"]}
        if "/workflows/12/runs?" in endpoint:
            runs = [state["run"]]
            if state.get("newer") or (state.get("newer_during_read") and state["requests"].count(endpoint) > 1):
                runs.append({**state["run"], "id": 43, "conclusion": "failure"})
            return [{"total_count": len(runs), "workflow_runs": copy.deepcopy(runs)}]
        if "/attempts/1/jobs?" in endpoint:
            jobs = copy.deepcopy(state["jobs"])
            if state.get("jobs_during_read") and state["requests"].count(endpoint) > 1:
                jobs[0]["conclusion"] = "failure"
            return [{"total_count": len(jobs) + int(bool(state.get("pagination"))), "jobs": jobs}]
        if "/check-runs/" in endpoint:
            job = next(j for j in state["jobs"] if endpoint.endswith(str(j["id"])))
            return {**job, "app": {"id": state["app"]}, "check_suite": {"id": 91}}
        if "/git/commits/" in endpoint:
            sha = endpoint.rsplit("/", 1)[-1]
            return {"sha": sha, "tree": {"sha": state["tree"] if sha == MERGE else TREE},
                    "parents": [{"sha": HEAD}, {"sha": "e" * 40}]}
        if endpoint.endswith("/actions/runs/42"):
            return copy.deepcopy(state["run"])
        raise AssertionError(endpoint)

    monkeypatch.setattr(gate, "_api", api)
    # Deliberately late-bound: the baseline should fail acceptance, not collection.
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, state["log"], ""))
    return state


def test_named_policy_accepts_exact_tree_and_records_provenance(evidence):
    result = gate.collect_acceptance(REPO, URL)
    assert result["ok"], result
    assert result["policy_source"] == "explicit_named_checks"
    assert result["tested_sha"] == MERGE
    assert result["head_sha"] == HEAD
    assert result["tree_sha"] == TREE
    assert result["workflow_run_id"] == 42
    assert result["run_attempt"] == 1
    assert len(result["checks"]) == len(POLICY["required_checks"])
    assert not any("rules/branches" in r or r == "graphql" for r in evidence["requests"])


@pytest.mark.parametrize("fault", ["missing", "pending", "failure", "cancelled", "skipped", "neutral", "stale", "tree", "blob", "app", "log", "race", "error", "newer", "attempt", "repo", "workflow", "newer_during_read", "jobs_during_read", "pagination"])
def test_named_policy_rejects_untrusted_or_noncurrent_evidence(evidence, fault):
    if fault == "missing":
        evidence["jobs"].pop()
    elif fault in {"pending", "failure", "cancelled", "skipped", "neutral"}:
        evidence["jobs"][0]["conclusion"] = fault
    elif fault == "stale":
        evidence["jobs"][0]["head_sha"] = "f" * 40
    elif fault in {"tree", "blob"}:
        evidence[fault] = "f" * 40
    elif fault == "app":
        evidence["app"] = 7
    elif fault == "log":
        evidence["log"] = "No checkout evidence"
    elif fault == "attempt":
        evidence["jobs"][0]["run_attempt"] = 2
    elif fault == "repo":
        evidence["pr"]["head"]["repo"]["full_name"] = "other/repo"
    elif fault == "workflow":
        evidence["run"]["workflow_id"] = 99
    else:
        evidence[fault] = True
    assert not gate.collect_acceptance(REPO, URL)["ok"]


def test_direct_head_checkout_is_accepted(evidence):
    evidence["log"] = evidence["log"].replace(MERGE, HEAD)
    assert gate.collect_acceptance(REPO, URL)["ok"]


def test_direct_head_cannot_use_unapproved_merge_workflow(evidence):
    evidence["log"] = evidence["log"].replace(MERGE, HEAD)
    evidence["merge_blob"] = "f" * 40
    assert not gate.collect_acceptance(REPO, URL)["ok"]


def test_duplicate_checkout_evidence_is_rejected(evidence):
    evidence["log"] *= 2
    assert not gate.collect_acceptance(REPO, URL)["ok"]


def test_unconfigured_repository_still_uses_protected_checks(evidence, monkeypatch):
    seen = []
    def unavailable(endpoint, **kwargs):
        seen.append(endpoint)
        raise subprocess.CalledProcessError(1, "gh")
    monkeypatch.setattr(gate, "_api", unavailable)
    result = gate.collect_acceptance("other/repo", "https://github.com/other/repo/pull/3")
    assert not result["ok"]
    assert seen == ["graphql"]
    assert "policy_source" not in result


@pytest.mark.parametrize("value", [[], [""], ["tests", "tests"], "tests", None])
def test_invalid_required_checks_fail_closed(evidence, monkeypatch, value):
    from hermes_cli import config_effective
    policy = {**POLICY, "required_checks": value}
    monkeypatch.setattr(config_effective, "load_user_config_effective", lambda **kw: {"kanban": {"pr_acceptance_policies": {REPO: policy}}})
    assert not gate.collect_acceptance(REPO, URL)["ok"]


def test_policy_receipt_and_terminal_state_are_persisted(evidence):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    kb.init_db()
    with connect() as conn:
        for success in (False, True):
            evidence["jobs"][0]["conclusion"] = "success" if success else "failure"
            task_id = kb.create_task(conn, title="scoped evidence", completion_contract=REPO)
            assert kb.complete_task(conn, task_id, summary="test", metadata={"published_pr": URL}) is success
            assert (kb.get_task(conn, task_id).status == "done") is success
            payload = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (task_id,)).fetchone()[0])
            assert payload["policy_source"] == "explicit_named_checks"
            assert payload["workflow_run_id"] == 42
            assert payload["ok"] is success


def test_empty_policy_fails_closed(evidence, monkeypatch):
    from hermes_cli import config_effective
    monkeypatch.setattr(config_effective, "load_user_config_effective", lambda **kw: {"kanban": {"pr_acceptance_policies": {REPO: {}}}})
    assert not gate.collect_acceptance(REPO, URL)["ok"]
