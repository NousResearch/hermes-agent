"""Current required evidence must not include superseded check suites."""
import json
import subprocess

import pytest

from hermes_cli import kanban_pr_acceptance as acceptance

SHA = "a" * 40


def run(run_id, conclusion, *, app=1, status="completed", sha=SHA):
    return {"id": run_id, "name": "required", "head_sha": sha,
            "app": {"id": app}, "status": status, "conclusion": conclusion}


@pytest.fixture
def github(monkeypatch):
    state = {"runs": [], "app": 1, "statuses": [], "head": SHA}

    def transport(command, **kwargs):
        endpoint = command[2]
        if endpoint == "graphql":
            value = {"data": {"repository": {"pullRequest": {
                "headRefOid": SHA, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                    {"context": "required", "app": {"databaseId": state["app"]}}]}}}}}}
        elif "/rules/branches/" in endpoint:
            value = [[]]
        elif "/check-runs?" in endpoint:
            value = [{"total_count": len(state["runs"]), "check_runs": [r]}
                     for r in state["runs"]] or [{"total_count": 0, "check_runs": []}]
        elif "/statuses?" in endpoint:
            value = [state["statuses"]]
        elif "/pulls/" in endpoint:
            value = {"head": {"sha": state["head"]}, "base": {"ref": "main"}, "state": "open"}
        else:
            raise AssertionError(endpoint)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(value))

    monkeypatch.setattr(acceptance.subprocess, "run", transport)
    return state


def collect():
    return acceptance.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")


@pytest.mark.parametrize("reverse", [False, True])
def test_new_success_supersedes_cancelled_suite(github, reverse):
    github["runs"] = [run(10, "cancelled"), run(20, "success")][:: -1 if reverse else 1]
    receipt = collect()
    assert receipt["ok"], receipt
    assert [c["id"] for c in receipt["checks"]] == [20]


@pytest.mark.parametrize("conclusion,status,expected", [
    ("failure", "completed", "failure"), ("cancelled", "completed", "infra"),
    (None, "queued", "pending"), (None, "in_progress", "pending"),
    ("success", "completed", "success"),
])
def test_new_run_not_old_success_decides(github, conclusion, status, expected):
    github["runs"] = [run(20, conclusion, status=status), run(10, "success")]
    assert collect()["classification"] == expected


@pytest.mark.parametrize("app", [None, -1, 1])
def test_different_apps_do_not_supersede_each_other(github, app):
    github.update(app=app, runs=[run(10, "failure"), run(20, "success", app=2)])
    assert collect()["classification"] == "failure"


def test_wrong_app_cannot_satisfy_pinned_context(github):
    github["runs"] = [run(20, "success", app=2)]
    assert collect()["classification"] == "missing"


@pytest.mark.parametrize("fault,expected", [("empty", "missing"), ("stale", "stale"), ("head", "stale")])
def test_missing_and_stale_evidence_still_reject(github, fault, expected):
    github["runs"] = [] if fault == "empty" else [run(20, "success", sha="b" * 40 if fault == "stale" else SHA)]
    if fault == "head":
        github["head"] = "b" * 40
    assert collect()["classification"] == expected


def test_legacy_status_remains_required_alongside_latest_run(github):
    github.update(app=None, runs=[run(10, "cancelled"), run(20, "success")], statuses=[
        {"id": 1, "context": "required", "state": "success"},
        {"id": 2, "context": "required", "state": "failure"}])
    assert collect()["classification"] == "failure"


def test_completion_persists_only_current_check_receipt(github):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    github["runs"] = [run(10, "cancelled"), run(20, "success")]
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="Rerun CI", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, result="done", metadata={
            "published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status == "done"
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,),
        ).fetchone()[0])
        assert receipt["ok"]
        assert [c["id"] for c in receipt["checks"]] == [20]


def test_transport_exception_fails_closed(github, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("gh", 30)
    monkeypatch.setattr(acceptance.subprocess, "run", fail)
    assert collect()["classification"] == "infra"
