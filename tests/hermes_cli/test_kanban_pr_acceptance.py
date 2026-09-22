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
        for fault in ("missing", "stale", "head_change"):
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
            assert not kb.complete_task(conn, tid, result="done", expected_run_id=run_id,
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


PR7 = "https://github.com/acme/repo/pull/7"


def _last_receipt(conn, tid):
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchall()
    assert rows
    return json.loads(rows[-1][0])


def _last_run_metadata(conn, tid):
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


@pytest.mark.linux_only
def test_exact_pr_contract_auto_populates_published_pr(github):
    with connect() as conn:
        github.update(conclusion="success", head="a" * 40)
        # Omitted: the exact-PR-URL contract resolves itself and the closing
        # run's metadata carries published_pr.
        tid = kb.create_task(conn, title="omitted", completion_contract=PR7)
        assert kb.complete_task(conn, tid, summary="done")
        assert kb.get_task(conn, tid).status == "done"
        assert _last_receipt(conn, tid)["pr_url"] == PR7
        assert _last_run_metadata(conn, tid)["published_pr"] == PR7
        # Matching explicit value stays accepted unchanged.
        tid = kb.create_task(conn, title="matching", completion_contract=PR7)
        assert kb.complete_task(conn, tid, summary="done", metadata={"published_pr": PR7})
        assert kb.get_task(conn, tid).status == "done"
        assert _last_run_metadata(conn, tid)["published_pr"] == PR7
        # Conflicting explicit value is rejected, classified, and not done.
        tid = kb.create_task(conn, title="conflict", completion_contract=PR7)
        assert not kb.complete_task(conn, tid, summary="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/8"})
        assert kb.get_task(conn, tid).status != "done"
        assert _last_receipt(conn, tid)["classification"] == "conflict"


@pytest.mark.linux_only
def test_exact_pr_contract_keeps_exact_head_and_non_pr_explicit(github):
    with connect() as conn:
        github.update(conclusion="success", head="a" * 40)
        # Exact-head acceptance still applies to the auto-populated contract.
        tid = kb.create_task(conn, title="head moved", completion_contract=PR7)
        github["head_change"] = True
        assert not kb.complete_task(conn, tid, summary="done")
        assert kb.get_task(conn, tid).status != "done"
        assert _last_receipt(conn, tid)["classification"] == "stale"
        github.pop("head_change")
        # OWNER/REPO and local-only contracts are never guessed.
        owner = kb.create_task(conn, title="owner", completion_contract="acme/repo")
        assert not kb.complete_task(conn, owner, summary="local green")
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="local green")


@pytest.mark.linux_only
def test_request_review_auto_populates_published_pr_from_contract(github):
    with connect() as conn:
        # Omitted: review handoff metadata carries the contract URL.
        tid = kb.create_task(conn, title="review omitted", completion_contract=PR7)
        assert kb.request_review(conn, tid, summary="ready for review")
        assert kb.get_task(conn, tid).status == "review"
        assert _last_run_metadata(conn, tid)["published_pr"] == PR7
        # Matching explicit value passes through.
        tid = kb.create_task(conn, title="review matching", completion_contract=PR7)
        assert kb.request_review(conn, tid, summary="ready", metadata={"published_pr": PR7})
        assert kb.get_task(conn, tid).status == "review"
        assert _last_run_metadata(conn, tid)["published_pr"] == PR7
        # Conflicting value rejects the handoff; the task stays untouched.
        tid = kb.create_task(conn, title="review conflict", completion_contract=PR7)
        ok, reason = kb.request_review(conn, tid, summary="ready", with_reason=True,
                                       metadata={"published_pr": "https://github.com/acme/repo/pull/8"})
        assert not ok
        assert "conflicts" in reason
        assert kb.get_task(conn, tid).status != "review"
        # Non-PR contracts: nothing is auto-populated.
        tid = kb.create_task(conn, title="review owner", completion_contract="acme/repo")
        assert kb.request_review(conn, tid, summary="ready")
        assert "published_pr" not in _last_run_metadata(conn, tid)


@pytest.mark.linux_only
def test_worker_context_exposes_exact_pr_contract(github):
    with connect() as conn:
        tid = kb.create_task(conn, title="ctx", completion_contract=PR7)
        context = kb.build_worker_context(conn, tid)
        assert f"PR contract: {PR7}" in context
        owner = kb.create_task(conn, title="ctx owner", completion_contract="acme/repo")
        assert "PR contract:" not in kb.build_worker_context(conn, owner)
        local = kb.create_task(conn, title="ctx local", completion_contract="local-only")
        assert "PR contract:" not in kb.build_worker_context(conn, local)
