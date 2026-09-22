"""Lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


PLAN_ERROR = {
    "message": "Upgrade to GitHub Pro or make this repository public to enable this feature.",
    "documentation_url": "https://docs.github.com/rest/repos/rules#get-rules-for-a-branch",
    "status": "403",
}


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            status = 200
            if self.path == "/graphql":
                value = {"data": {"repository": {"isPrivate": state.get("private", False), "pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
                if "protection" in state:
                    value["data"]["repository"]["pullRequest"]["baseRef"]["branchProtectionRule"] = state["protection"]
                if state.get("base_missing"):
                    value["data"]["repository"]["pullRequest"]["baseRef"] = None
                if state.get("graphql_errors"):
                    value["errors"] = [{"message": "Protection unavailable"}]
            elif "/rules/branches/" in self.path:
                value = state.get("rules_pages", [[]])
                if "rules_error" in state:
                    status, value = state["rules_error"]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 2 if state.get("wrong_app") else 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                count = state.get("optional_count", 100)
                value = [{"total_count": count + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "status": "completed",
                     "conclusion": state.get("optional_conclusion", "skipped")}
                    for i in range(count)]}, {"total_count": count + len(runs), "check_runs": runs}]
                if state.get("incomplete"):
                    value.pop()
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = state.get("statuses", [[]])
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "other" if state.get("base_change") else "main"},
                         "state": "closed" if state.get("closed") else "open"}
            else:
                self.send_error(404)
                return
            self.send_response(status)
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
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request,urllib.error\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "try:\n"
                  "    print(urllib.request.urlopen(u).read().decode())\n"
                  "except urllib.error.HTTPError as exc:\n"
                  "    body = exc.read().decode()\n"
                  "    print('[' + body + ']' if '--slurp' in sys.argv else body)\n"
                  "    print(f'gh: request failed (HTTP {exc.code})', file=sys.stderr)\n"
                  "    sys.exit(1)\n")
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
@pytest.mark.parametrize("policy_source", ["graphql", "ruleset"])
def test_pr_completion_requires_current_required_evidence(github, policy_source):
    if policy_source == "ruleset":
        github.update(protection=None, rules_pages=[[], [{"type": "required_status_checks",
            "parameters": {"required_status_checks": [{"context": "required", "integration_id": 1}]}}]])
    with connect() as conn:
        for conclusion in ("failure", "pending", "cancelled", "timed_out", "action_required", "neutral", "skipped", None, "success"):
            github.update(conclusion=conclusion, head="a" * 40)
            tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
            ok = kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
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
        for fault in ("missing", "stale", "head_change", "wrong_app"):
            github.update(conclusion="success", head="a" * 40)
            github[fault] = True
            tid = kb.create_task(conn, title=fault, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).status != "done"
            github.pop(fault)
        # Omission and a sibling repository cannot downgrade the stored declaration.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, summary="local green")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/other/repo/pull/7"})
        before = len(github["requests"])
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="https://github.com/acme/repo/pull/7 is background context")
        assert len(github["requests"]) == before


@pytest.mark.linux_only
@pytest.mark.parametrize("plan_limited", [False, True])
def test_acceptance_receipts_and_terminal_write_share_run_ownership(github, plan_limited):
    if plan_limited:
        github.update(private=True, protection=None, rules_error=(403, PLAN_ERROR), optional_conclusion="success")
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
            assert not kb.complete_task(conn, tid, result="done", expected_run_id=run_id,
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


@pytest.mark.linux_only
@pytest.mark.parametrize("private,error,conclusion,classification", [
    (True, PLAN_ERROR, "success", "success"),
    (True, PLAN_ERROR, "failure", "failure"),
    (True, {**PLAN_ERROR, "message": "Resource not accessible by integration"}, "success", "infra"),
    (True, {**PLAN_ERROR, "message": "API rate limit exceeded"}, "success", "infra"),
    (True, {**PLAN_ERROR, "status": "401"}, "success", "infra"),
    (True, {**PLAN_ERROR, "status": "429"}, "success", "infra"),
    (True, {**PLAN_ERROR, "status": "500"}, "success", "infra"),
    (False, PLAN_ERROR, "success", "infra"),
    (None, PLAN_ERROR, "success", "infra"),
])
def test_only_private_plan_denial_preserves_graphql_policy(github, private, error, conclusion, classification):
    github.update(private=private, rules_error=(int(error["status"]), error), conclusion=conclusion)
    with connect() as conn:
        tid = kb.create_task(conn, title="Plan-limited rules", completion_contract="acme/repo")
        ok = kb.complete_task(conn, tid, summary="Published", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
        assert receipt["classification"] == classification
        assert ok is (classification == "success")
        assert (kb.get_task(conn, tid).status == "done") is ok
        assert kb.get_task(conn, tid).completion_contract == "https://github.com/acme/repo/pull/7"
        if classification != "infra":
            assert receipt["rules_status"] == "plan_unavailable"
            assert receipt["policy_source"] == "required_checks"
            assert receipt["required"] == [{"context": "required", "app_id": 1}]
            assert {c["id"] for c in receipt["checks"]} == {42}
            assert github["requests"][-1] == "/repos/acme/repo/pulls/7"


@pytest.mark.linux_only
@pytest.mark.parametrize("changes,classification", [
    ({}, "success"),
    *[({"conclusion": outcome}, classification) for outcome, classification in (
        ("failure", "failure"), ("pending", "pending"), ("cancelled", "infra"),
        ("timed_out", "infra"), ("action_required", "infra"), ("skipped", "infra"),
        ("neutral", "infra"), (None, "infra"))],
    ({"optional_conclusion": "skipped"}, "infra"),
    ({"optional_conclusion": "failure"}, "failure"),
    ({"optional_count": 0, "missing": True}, "missing"),
    ({"stale": True}, "stale"),
    ({"head_change": True}, "stale"),
    ({"base_change": True}, "stale"),
    ({"closed": True}, "stale"),
    ({"incomplete": True}, "infra"),
    ({"graphql_errors": True}, "infra"),
    ({"base_missing": True}, "infra"),
    ({"protection": {}}, "infra"),
    ({"rules_error": None}, "missing"),
    ({"statuses": [[{"id": 12, "context": "legacy", "state": "failure"}]]}, "failure"),
    ({"statuses": [[{"id": 12, "context": "legacy", "state": "pending"}]]}, "pending"),
    ({"optional_count": 0, "missing": True, "statuses": [
        [{"id": 12, "context": "legacy", "state": "success"}],
        [{"id": 11, "context": "legacy", "state": "failure"}]]}, "success"),
    pytest.param({"rules_error": (500, PLAN_ERROR)}, "infra", id="http-status-mismatch"),
    pytest.param({"rules_error": None, "rules_pages": None}, "infra", id="null-rules-response"),
])
def test_plan_limited_completion_requires_all_observed_head_evidence(github, changes, classification):
    github.update(private=True, protection=None, rules_error=(403, PLAN_ERROR), optional_conclusion="success")
    github.update(changes)
    if github["rules_error"] is None:
        del github["rules_error"]
    with connect() as conn:
        tid = kb.create_task(conn, title="Observed CI", completion_contract="https://github.com/acme/repo/pull/7")
        ok = kb.complete_task(conn, tid, summary="Published")
        task = kb.get_task(conn, tid)
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
        assert receipt["classification"] == classification
        assert ok is (classification == "success")
        assert (task.status == "done") is ok
        assert task.completion_contract == "https://github.com/acme/repo/pull/7"
        if ok:
            assert receipt["rules_status"] == "plan_unavailable"
            assert receipt["policy_source"] == "observed_checks"
            assert receipt["required"] == []
            expected_ids = {12} if changes.get("statuses") else {42, *range(1000, 1100)}
            assert {c["id"] for c in receipt["checks"]} == expected_ids
            assert {c["head_sha"] for c in receipt["checks"]} == {"a" * 40}
            assert github["requests"][-1] == "/repos/acme/repo/pulls/7"
