"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            failures = state.get("failures")
            for fragment, failure in (failures if isinstance(failures, dict) else {}).items():
                code, message = failure
                if fragment in self.path:
                    self.send_response(code)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(message.encode())
                    return
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = [[]]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                value = [{"total_count": 100 + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)]}, {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = [[]]
            elif "/compare/" in self.path:
                value = {"status": state.get("compare", "ahead"),
                         "merge_base_commit": {"sha": "b" * 40}}
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            elif "/commits/" in self.path:
                if state.get("missing_commit") and self.path.endswith(f"/commits/{sha}"):
                    self.send_error(404, "Not Found")
                    return
                value = {"sha": sha}
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
    gh.write_text(
        f"#!{sys.executable}\n"
        "import sys, urllib.request, urllib.error\n"
        f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
        "try:\n"
        "    print(urllib.request.urlopen(u).read().decode())\n"
        "except urllib.error.HTTPError as e:\n"
        "    print('gh: ' + e.read().decode().strip() + ' (HTTP %d)' % e.code, file=sys.stderr)\n"
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


PLAN_GATE = "Upgrade to GitHub Pro or make this repository public to enable this feature."
PR = "https://github.com/acme/repo/pull/7"


def _receipt(conn, tid):
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' "
                       "ORDER BY id DESC", (tid,)).fetchone()
    assert row is not None
    return json.loads(row[0])


def _status(conn, tid):
    task = kb.get_task(conn, tid)
    assert task is not None
    return task.status


@pytest.mark.linux_only
def test_capability_403_degrades_only_with_validated_alternative_evidence(github):
    with connect() as conn:
        github["failures"] = {"/rules/branches/": (403, PLAN_GATE)}
        tid = kb.create_task(conn, title="Private free-plan repository", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR})
        assert _status(conn, tid) == "done"
        receipt = _receipt(conn, tid)
        assert receipt["ok"] and receipt["classification"] == "success"
        assert receipt["verification_mode"] == "local-only"
        assert receipt["degraded_reason"] == "provider_capability"
        assert receipt["degraded_check"] == "repos/acme/repo/rules/branches/main"
        assert receipt["degraded_status"] == 403
        assert receipt["degraded_message"] == PLAN_GATE
        evidence = {item["check"]: item for item in receipt["alternative_evidence"]}
        assert evidence["repos/acme/repo/pulls/7"]["ok"]
        assert evidence["repos/acme/repo/commits/" + "a" * 40]["ok"]
        assert evidence["repos/acme/repo/compare/main..." + "a" * 40]["ok"]
        assert all(item["ok"] for item in receipt["alternative_evidence"])
        assert receipt["head_sha"] == "a" * 40


@pytest.mark.linux_only
@pytest.mark.parametrize("code,message", [
    (401, "Bad credentials"),
    (403, "Resource not accessible by personal access token"),
    (403, "You do not have permission to view branch protection rules"),
    (404, "Not Found"),
    (429, "You have exceeded a secondary rate limit"),
    (500, "Server Error"),
])
def test_non_capability_failures_never_fall_back(github, code, message):
    with connect() as conn:
        github["failures"] = {"/rules/branches/": (code, message)}
        tid = kb.create_task(conn, title="rules failure", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": PR})
        assert _status(conn, tid) != "done"
        receipt = _receipt(conn, tid)
        assert receipt["ok"] is False and receipt["classification"] != "success"
        assert receipt["verification_mode"] == "github-api"
        assert "degraded_reason" not in receipt


@pytest.mark.linux_only
@pytest.mark.parametrize("conclusion", ["failure", "pending", "cancelled", None])
def test_capability_403_with_unproven_ci_does_not_pass(github, conclusion):
    with connect() as conn:
        github["failures"] = {"/rules/branches/": (403, PLAN_GATE)}
        github["conclusion"] = conclusion
        tid = kb.create_task(conn, title="unproven checks", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": PR})
        assert _status(conn, tid) != "done"
        receipt = _receipt(conn, tid)
        assert receipt["ok"] is False
        assert receipt["classification"] in {"failure", "pending", "infra", "missing"}
        assert receipt["verification_mode"] == "local-only"
        assert any(not item["ok"] for item in receipt["alternative_evidence"])


@pytest.mark.linux_only
def test_capability_403_without_remote_commit_does_not_pass(github):
    with connect() as conn:
        github["failures"] = {"/rules/branches/": (403, PLAN_GATE)}
        github["missing_commit"] = True
        tid = kb.create_task(conn, title="absent commit", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": PR})
        assert _status(conn, tid) != "done"
        receipt = _receipt(conn, tid)
        assert receipt["ok"] is False and receipt["classification"] == "missing"
        assert "commit" in receipt["detail"]
        assert any(item["check"].endswith("commits/" + "a" * 40) and not item["ok"]
                   for item in receipt["alternative_evidence"])


@pytest.mark.linux_only
def test_capability_403_with_absent_pr_does_not_pass(github):
    with connect() as conn:
        github["failures"] = {"/rules/branches/": (403, PLAN_GATE), "/pulls/": (404, "Not Found")}
        tid = kb.create_task(conn, title="absent PR", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": PR})
        assert _status(conn, tid) != "done"
        receipt = _receipt(conn, tid)
        assert receipt["ok"] is False and receipt["classification"] != "success"


@pytest.mark.linux_only
def test_capability_403_with_diverged_head_does_not_pass(github):
    with connect() as conn:
        github["failures"] = {"/rules/branches/": (403, PLAN_GATE)}
        github["compare"] = "diverged"
        tid = kb.create_task(conn, title="merge conflict", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": PR})
        assert _status(conn, tid) != "done"
        receipt = _receipt(conn, tid)
        assert receipt["ok"] is False and receipt["classification"] == "missing"
        assert "fast-forward" in receipt["detail"]


@pytest.mark.linux_only
def test_full_api_path_verdict_is_unchanged(github):
    with connect() as conn:
        tid = kb.create_task(conn, title="Normal publication", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, metadata={"published_pr": PR})
        receipt = _receipt(conn, tid)
        assert receipt["ok"] and receipt["classification"] == "success"
        assert receipt["verification_mode"] == "github-api"
        assert "degraded_reason" not in receipt and "alternative_evidence" not in receipt
        assert "rules/branches" in " ".join(github["requests"])


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
            assert _status(conn, tid) != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")
