"""Lifecycle invariants using real SQLite and a local GitHub HTTP contract.

Parked-card acceptance regressions cover #116170.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


@pytest.mark.parametrize("status", ["triage", "blocked"])
def test_parked_completion_keeps_acceptance_gates_and_audit(github, status):
    from hermes_cli import kanban as cli
    from hermes_cli.profiles import get_active_profile_name

    actor = get_active_profile_name()
    with connect() as conn:
        parent = kb.create_task(conn, title="Required dependency")
        tid = kb.create_task(
            conn, title="Accepted work", assignee="builder", parents=[parent],
            triage=status == "triage", initial_status="blocked" if status == "blocked" else "running",
            completion_contract="https://github.com/acme/repo/pull/7",
        )
        command = f'complete {tid} --result "Human accepted the work"'
        assert "unsatisfied parent dependencies" in cli.run_slash(command)
        assert kb.get_task(conn, tid).status == status
        assert kb.complete_task(conn, parent, result="Dependency accepted")

        github["conclusion"] = "failure"
        assert "Completed" not in cli.run_slash(command)
        assert kb.get_task(conn, tid).status == status
        receipts = [e for e in kb.list_events(conn, tid) if e.kind == "pr_acceptance"]
        assert receipts and receipts[-1].payload["ok"] is False

        github["conclusion"] = "success"
        assert f"Completed {tid}" in cli.run_slash(command)
        task = kb.get_task(conn, tid)
        assert task.status == "done" and task.completed_at is not None
        events = kb.list_events(conn, tid)
        completed = [e for e in events if e.kind == "completed"]
        assert len(completed) == 1
        assert completed[0].payload["source_status"] == status
        assert completed[0].payload["accepted_by"] == actor
        assert completed[0].payload["summary"] == task.result
        assert not any(e.kind in {"specified", "promoted", "claimed"} for e in events)
        receipt = [e for e in events if e.kind == "pr_acceptance"][-1]
        assert receipt.payload["ok"] is True
        run = kb.latest_run(conn, tid)
        assert run.outcome == "completed" and run.summary == task.result


@pytest.mark.parametrize("status", ["triage", "blocked"])
def test_parked_completion_requires_independent_profile(github, tmp_path, monkeypatch, status):
    from hermes_cli import kanban as cli
    from tools import kanban_tools as kt

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))
    homes = {name: tmp_path / ".hermes" / "profiles" / name for name in ("builder", "reviewer")}
    for home in homes.values():
        home.mkdir(parents=True)

    with connect() as conn:
        tid = kb.create_task(conn, title="Parked work", assignee="builder",
                             triage=status == "triage", initial_status="blocked" if status == "blocked" else "running")
        monkeypatch.setenv("HERMES_HOME", str(homes["builder"]))
        before = kb.list_events(conn, tid)
        message = "acceptance requires a profile other than the implementer"
        assert message in cli.run_slash(f"complete {tid} --force --result accepted")
        assert message in kt._handle_complete({"task_id": tid, "result": "accepted"})
        assert kb.get_task(conn, tid).status == status
        assert kb.list_events(conn, tid) == before

        monkeypatch.setenv("HERMES_HOME", str(homes["reviewer"]))
        assert "error" not in json.loads(kt._handle_complete({"task_id": tid, "result": "accepted"}))
        assert kb.get_task(conn, tid).status == "done"
        assert kb.list_events(conn, tid)[-1].payload["accepted_by"] == "reviewer"

        # Review handoffs reassign the card, but must not erase its implementer.
        reviewed = kb.create_task(conn, title="Review handoff", assignee="builder")
        assert kb.request_review(conn, reviewed, summary="Ready", reviewer="reviewer")
        assert kb.claim_review_task(conn, reviewed) is not None
        assert kb.block_task(conn, reviewed, reason="Waiting for acceptance", kind="needs_input")
        monkeypatch.setenv("HERMES_HOME", str(homes["builder"]))
        assert message in cli.run_slash(f"complete {reviewed} --result accepted")
        assert kb.get_task(conn, reviewed).status == "blocked"
        monkeypatch.setenv("HERMES_HOME", str(homes["reviewer"]))
        assert f"Completed {reviewed}" in cli.run_slash(f"complete {reviewed} --result accepted")
