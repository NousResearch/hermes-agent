"""Commit statuses cannot prove the producer of an app-pinned requirement.

Exercise the native completion boundary with real SQLite and synthetic GitHub
responses. A matching creator name, URL or isRequired flag is not app attribution.
"""
from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_acceptance as acceptance
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def status_api(monkeypatch):
    head = "a" * 40
    state = {"head": head, "status_state": "success", "runs": []}

    def api(endpoint, *, query=None, paginate=False, profile_home=None):
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": head, "baseRefName": "main", "state": "MERGED",
                "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                    {"context": "required", "app": {"databaseId": 17}},
                ]}},
            }}}}
        if "/rules/branches/" in endpoint:
            return [[]]
        if "/check-runs" in endpoint:
            return [{"total_count": len(state["runs"]), "check_runs": state["runs"]}]
        if "/statuses" in endpoint:
            return [[{
                "id": 42, "context": "required", "state": state["status_state"],
                "creator": {"login": "trusted-app[bot]", "type": "Bot", "id": 1700},
                "target_url": "https://github.com/acme/repo/actions/runs/42",
                "isRequired": True,
            }]]
        if "/pulls/" in endpoint:
            return {"head": {"sha": state["head"]}, "base": {"ref": "main"},
                    "state": "closed", "merged": True}
        raise AssertionError(f"Unexpected GitHub endpoint: {endpoint}")

    monkeypatch.setattr(acceptance, "_api", api)
    return state


@pytest.mark.parametrize("status_state", ["success", "pending", "failure", "error"])
def test_pinned_status_retains_evidence_without_claiming_app_attribution(status_api, status_state):
    status_api["status_state"] = status_state
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        owner = kb.claim_task(conn, tid)
        assert owner is not None
        assert not kb.complete_task(
            conn, tid, result="merged", expected_run_id=owner.current_run_id,
            metadata={"published_pr": "https://github.com/acme/repo/pull/7"},
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == owner.current_run_id
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,),
        ).fetchone()[0])
        assert receipt["classification"] == "unverifiable_app"
        assert receipt["ok"] is False
        assert receipt["checks"][0] == {
            "name": "required", "id": 42,
            "url": "https://github.com/acme/repo/actions/runs/42",
            "head_sha": "a" * 40, "classification": "unverifiable_app",
            "conclusion": status_state, "required_app_id": 17,
        }
        assert "originating GitHub App ID" in receipt["detail"]
        assert task.last_failure_error is not None
        assert "unverifiable_app" in task.last_failure_error
        assert "source pin" in receipt["recovery"]


@pytest.mark.parametrize("app_id", [17, 18])
def test_only_matching_check_run_proves_pinned_app(status_api, app_id):
    status_api["runs"] = [{
        "id": 99, "name": "required", "head_sha": "a" * 40,
        "app": {"id": app_id}, "status": "completed", "conclusion": "success",
        "html_url": "https://github.com/acme/repo/actions/runs/99",
    }]
    receipt = acceptance.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"] is (app_id == 17)
    assert receipt["classification"] == ("success" if app_id == 17 else "unverifiable_app")
    status_api["head"] = "b" * 40
    raced = acceptance.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert raced["ok"] is False
    assert raced["classification"] == "stale"
