"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from hermes_cli import kanban_pr_acceptance as acceptance


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = [{"type": "required_status_checks", "parameters": {
                    "required_status_checks": [{"context": "required", "integration_id": 1}]}}]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                value = {"total_count": 100 + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)] + runs}
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = []
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "print(urllib.request.urlopen(u).read().decode())\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.linux_only
def test_pr_completion_requires_current_required_evidence(github):
    with connect() as conn:
        for conclusion in ("failure", "pending", "cancelled", "timed_out", "action_required", "neutral", "skipped", None, "success"):
            github.update(conclusion=conclusion, head="a" * 40)
            tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
            ok = kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert ok is (conclusion == "success")
            task = kb.get_task(conn, tid)
            assert (task.status == "done") is ok
            receipts = [json.loads(r[0]) for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
            assert receipts and receipts[-1]["head_sha"] == "a" * 40
            if not ok:
                assert task.status in {"running", "ready", "blocked", "review"}
                assert "retry" in receipts[-1]["recovery"]
                assert receipts[-1]["checks"][0]["id"] == 42
        for fault in ("missing", "stale", "head_change"):
            github.update(conclusion="success", head="a" * 40)
            github[fault] = True
            tid = kb.create_task(conn, title=fault, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).status != "done"
            github.pop(fault)
        # Omission and a sibling repository cannot downgrade the stored declaration.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, summary="local green")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/other/repo/pull/7"})
        before = len(github["requests"])
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="https://github.com/acme/repo/pull/7 is background context")
        assert len(github["requests"]) == before


@pytest.mark.linux_only
def test_acceptance_receipts_and_terminal_write_share_run_ownership(github):
    with connect() as conn:
        for conclusion in ("success", "failure"):
            tid = kb.create_task(conn, title="race", completion_contract="acme/repo")
            owner = kb.claim_task(conn, tid)
            run_id = owner.current_run_id
            def reclaim():
                with connect() as rival:
                    assert kb.block_task(rival, tid, reason="Reassigned during acceptance")
                    assert kb.unblock_task(rival, tid)
                    github["replacement"] = kb.claim_task(rival, tid).current_run_id
            github.update(conclusion=conclusion, race=reclaim)
            assert not kb.complete_task(conn, tid, expected_run_id=run_id,
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


def test_paginated_api_does_not_require_slurp(monkeypatch):
    calls = []

    class Result:
        stdout = '{"page": 1}\n{"page": 2}\n'

    def run(command, **kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    assert acceptance._api("repos/acme/repo/rules", paginate=True) == [{"page": 1}, {"page": 2}]
    assert "--paginate" in calls[0]
    assert "--slurp" not in calls[0]


def test_paginated_api_preserves_single_page_documents(monkeypatch):
    payloads = [
        "[{\"type\": \"required_status_checks\"}]\n",
        "{\"total_count\": 1, \"check_runs\": []}\n",
        "[{\"context\": \"required\"}]\n",
    ]

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def run(command, **kwargs):
        return Result(payloads.pop(0))

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    assert acceptance._api("rules", paginate=True) == [[{"type": "required_status_checks"}]]
    assert acceptance._api("check-runs", paginate=True) == [{"total_count": 1, "check_runs": []}]
    assert acceptance._api("statuses", paginate=True) == [[{"context": "required"}]]


def test_single_page_rest_payloads_are_classified_correctly(monkeypatch):
    sha = "a" * 40
    responses = iter([
        {"data": {"repository": {"pullRequest": {
            "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
            "baseRef": {"branchProtectionRule": {"requiredStatusChecks": []}},
        }}}},
        [[{"type": "required_status_checks", "parameters": {
            "required_status_checks": [{"context": "required", "integration_id": 1}]}}]],
        [{"total_count": 1, "check_runs": [{
            "id": 42, "name": "required", "head_sha": sha, "app": {"id": 1},
            "status": "completed", "conclusion": "success"}]}],
        [[{"context": "required", "id": 7, "sha": sha, "state": "success"}]],
        {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"},
    ])
    monkeypatch.setattr(acceptance, "_api", lambda *args, **kwargs: next(responses))

    receipt = acceptance.collect_acceptance(
        "acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"] is True
    assert receipt["checks"][0]["classification"] == "success"


@pytest.mark.parametrize("merge_sha", [None, "not-a-sha", "b" * 39, "B" * 40])
def test_no_required_checks_requires_immutable_merge_sha(monkeypatch, merge_sha):
    sha = "a" * 40
    responses = iter([
        {"data": {"repository": {"pullRequest": {
            "headRefOid": sha, "baseRefName": "main", "state": "MERGED",
            "baseRef": {"branchProtectionRule": {"requiredStatusChecks": []}},
        }}}},
        [],
        {"head": {"sha": sha}, "base": {"ref": "main"},
         "state": "closed", "merged": True, "merge_commit_sha": merge_sha},
    ])
    monkeypatch.setattr(acceptance, "_api", lambda *args, **kwargs: next(responses))
    receipt = acceptance.collect_acceptance(
        "acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"] is (merge_sha == "b" * 40)
    assert receipt["classification"] == "no_required_checks"
    assert receipt["merge_sha"] == (merge_sha if merge_sha == "b" * 40 else None)


def test_no_required_checks_is_distinct_from_api_failure(monkeypatch):
    sha = "a" * 40

    def denied(*args, **kwargs):
        if "rules/branches" in args[0]:
            raise acceptance.subprocess.CalledProcessError(
                1, "gh", stderr="HTTP 403: Resource not accessible")
        return {"data": {"repository": {"pullRequest": {
            "headRefOid": sha, "baseRefName": "main", "state": "MERGED",
            "baseRef": {"branchProtectionRule": {"requiredStatusChecks": []}},
        }}}}

    monkeypatch.setattr(acceptance, "_api", denied)
    denied_receipt = acceptance.collect_acceptance(
        "acme/repo", "https://github.com/acme/repo/pull/7")
    assert denied_receipt["ok"] is False
    assert denied_receipt["classification"] == "infra"


def test_plan_gated_rules_endpoint_falls_back_to_no_required_checks(monkeypatch):
    sha = "a" * 40
    responses = iter([
        {"data": {"repository": {"pullRequest": {
            "headRefOid": sha, "baseRefName": "main", "state": "MERGED",
            "baseRef": {"branchProtectionRule": {"requiredStatusChecks": []}},
        }}}},
        {"head": {"sha": sha}, "base": {"ref": "main"},
         "state": "closed", "merged": True, "merge_commit_sha": "b" * 40},
    ])

    def plan_gated(*args, **kwargs):
        if "rules/branches" in args[0]:
            raise acceptance.subprocess.CalledProcessError(
                1, "gh", stderr="HTTP 403: Upgrade to GitHub Pro to enable this feature")
        return next(responses)

    monkeypatch.setattr(acceptance, "_api", plan_gated)
    receipt = acceptance.collect_acceptance(
        "acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"] is True
    assert receipt["classification"] == "no_required_checks"
    assert receipt["merge_sha"] == "b" * 40


@pytest.mark.parametrize(
    ("message", "is_no_rules"),
    [("HTTP 404: Branch not protected", True),
     ("HTTP 404: Not Found", False),
     ("HTTP 404: Repository not found", False)],
)
def test_rules_404_requires_explicit_unprotected_branch_response(message, is_no_rules):
    error = acceptance.subprocess.CalledProcessError(1, "gh", stderr=message)
    assert acceptance._rules_endpoint_reports_unprotected_branch(error) is is_no_rules


def test_inaccessible_rules_404_cannot_false_success(monkeypatch):
    sha = "a" * 40
    responses = iter([
        {"data": {"repository": {"pullRequest": {
            "headRefOid": sha, "baseRefName": "main", "state": "MERGED",
            "baseRef": {"branchProtectionRule": {"requiredStatusChecks": []}},
        }}}},
        {"head": {"sha": sha}, "base": {"ref": "main"},
         "state": "closed", "merged": True, "merge_commit_sha": "b" * 40},
    ])

    def inaccessible(*args, **kwargs):
        if "rules/branches" in args[0]:
            raise acceptance.subprocess.CalledProcessError(
                1, "gh", stderr="HTTP 404: Repository not found")
        return next(responses)

    monkeypatch.setattr(acceptance, "_api", inaccessible)
    receipt = acceptance.collect_acceptance(
        "acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"] is False
    assert receipt["classification"] == "infra"


@pytest.mark.parametrize("mismatch", ["head", "base"])
def test_no_required_checks_rejects_head_or_base_race(monkeypatch, mismatch):
    sha = "a" * 40
    current = {"head": {"sha": sha}, "base": {"ref": "main"},
               "state": "closed", "merged": True,
               "merge_commit_sha": "b" * 40}
    current[mismatch] = {"sha": "c" * 40} if mismatch == "head" else {"ref": "release"}
    responses = iter([
        {"data": {"repository": {"pullRequest": {
            "headRefOid": sha, "baseRefName": "main", "state": "MERGED",
            "baseRef": {"branchProtectionRule": {"requiredStatusChecks": []}},
        }}}},
        [],
        current,
    ])
    monkeypatch.setattr(acceptance, "_api", lambda *args, **kwargs: next(responses))

    receipt = acceptance.collect_acceptance(
        "acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"] is False
    assert receipt["classification"] == "stale"
