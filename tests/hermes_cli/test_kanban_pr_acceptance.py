"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import subprocess
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
                if state.get("plan_limited"):
                    value["data"]["repository"]["pullRequest"]["baseRef"]["branchProtectionRule"] = None
                if state.get("merged"):
                    value["data"]["repository"]["pullRequest"]["state"] = "MERGED"
            elif "/rules/branches/" in self.path:
                value = [[]]
            elif "/protection/required_status_checks" in self.path:
                if state.get("plan_limited"):
                    self.send_response(state.get("protection_error_status", 403))
                    self.end_headers()
                    self.wfile.write(b'{"message":"Upgrade to GitHub Pro or make this repository public to enable this feature."}')
                    return
                value = {"contexts": ["required"]}
            elif "/actions/jobs/" in self.path:
                skipped = self.path.endswith("/3")
                value = {"check_run_url": "https://api.github.com/repos/acme/repo/check-runs/56" if skipped else
                         "https://api.github.com/repos/acme/repo/check-runs/55",
                         "conclusion": "skipped" if skipped else "failure", "status": "completed",
                         "head_sha": sha, "run_id": 8,
                         "runner_name": "runner" if state.get("executed") else None,
                         "steps": [{"conclusion": "failure"}] if state.get("executed") else []}
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                if state.get("plan_limited"):
                    runs = [{"id": 55, "name": "action", "head_sha": sha, "status": "completed",
                             "conclusion": "failure", "app": {"slug": "github-actions", "id": 2},
                             "url": "https://api.github.com/repos/acme/repo/check-runs/55",
                             "details_url": "https://github.com/acme/repo/actions/runs/8/job/2"},
                            {"id": 56, "name": "dependent", "head_sha": sha, "status": "completed",
                             "conclusion": "skipped", "app": {"slug": "github-actions", "id": 2},
                             "url": "https://api.github.com/repos/acme/repo/check-runs/56",
                             "details_url": "https://github.com/acme/repo/actions/runs/8/job/3"}]
                optional_count = 0 if state.get("plan_limited") else 100
                value = [{"total_count": optional_count + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(optional_count)]}, {"total_count": optional_count + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = [[{"id": 3, "context": "continuous-integration/jenkins/pr-merge",
                           "state": state.get("jenkins", "success"),
                           "sha": state.get("status_sha", sha),
                           "target_url": "https://jenkins.example/job/PR-7/5/"}]] if state.get("plan_limited") else [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": state.get("rest_head", sha)}, "base": {"ref": "main"},
                         "state": "closed" if state.get("merged") else "open",
                         "merged": bool(state.get("merged")),
                         "merge_commit_sha": state.get("merge_sha", "b" * 40)}
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
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request,json\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "try:\n"
                  " data=json.loads(urllib.request.urlopen(u).read().decode())\n"
                  " if \"--paginate\" in sys.argv: print(\"\\n\".join(json.dumps(p) for p in data))\n"
                  " else: print(json.dumps(data))\n"
                  "except urllib.error.HTTPError as e:\n print(e.read().decode()+f' (HTTP {e.code})',file=sys.stderr); sys.exit(1)\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    state["home"] = tmp_path / "home"
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


@pytest.mark.linux_only
def test_plan_limited_fallback_requires_exact_authority_and_nonexecuted_actions(github, monkeypatch):
    from hermes_cli import kanban_pr_acceptance as acceptance

    Path(github["home"]).mkdir(exist_ok=True)
    (github["home"] / "config.yaml").write_text(
        "kanban:\n  pr_acceptance_authorities:\n    acme/repo:\n"
        "      - continuous-integration/jenkins/pr-merge\n")
    github.update(plan_limited=True, merged=True)
    url = "https://github.com/acme/repo/pull/7"

    def receipt():
        return acceptance.collect_acceptance(url, url)

    passed = receipt()
    assert passed["ok"] and passed["authority_source"] == "configured-plan-limited"
    assert {c["classification"] for c in passed["checks"]} == {
        "success", "pre-runner-infra", "skipped-after-pre-runner"}
    for changes, expected in [
        ({"jenkins": "pending"}, "pending"), ({"jenkins": "failure"}, "failure"),
        ({"status_sha": "b" * 40}, "stale"), ({"executed": True}, "failure"),
        ({"rest_head": "b" * 40}, "stale"), ({"merge_sha": None}, "stale"),
    ]:
        github.update(plan_limited=True, merged=True)
        github.update(changes)
        result = receipt()
        assert not result["ok"] and result["classification"] == expected, (changes, result)
        for key in changes:
            github.pop(key)

    github["protection_error_status"] = 401
    assert receipt()["classification"] == "infra"  # A matching message without HTTP 403 is not a plan limit.
    github.pop("protection_error_status")

    (github["home"] / "config.yaml").write_text("kanban:\n  pr_acceptance_authorities: {}\n")
    assert not receipt()["ok"]
    github.pop("plan_limited")
    assert receipt()["ok"]  # Reachable required-check API never uses configured fallback.


@pytest.mark.linux_only
def test_archived_pr_restoration_preserves_contract_and_regates_children(github):
    with connect() as conn:
        parent = kb.create_task(conn, title="source", completion_contract="acme/repo")
        child = kb.create_task(conn, title="release", parents=[parent])
        assert kb.archive_task(conn, parent)
        event_id = conn.execute(
            "SELECT id FROM task_events WHERE task_id=? AND kind='archived'", (parent,),
        ).fetchone()[0]
        assert not kb.restore_archived_task(conn, parent, archive_event_id=event_id + 1,
                                            completion_contract="acme/repo", reason="mistake")
        assert not kb.restore_archived_task(conn, parent, archive_event_id=event_id,
                                            completion_contract="local-only", reason="mistake")
        assert kb.restore_archived_task(conn, parent, archive_event_id=event_id,
                                        completion_contract="acme/repo", reason="mistake")
        restored, gated = kb.get_task(conn, parent), kb.get_task(conn, child)
        assert restored is not None and restored.status == "blocked"
        assert gated is not None and gated.status == "todo"
        assert not kb.restore_archived_task(conn, parent, archive_event_id=event_id,
                                            completion_contract="acme/repo", reason="duplicate")
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='archive_restored'",
                            (parent,)).fetchone()[0] == 1


@pytest.mark.linux_only
def test_restoring_archived_ancestor_fences_done_intermediate_and_running_descendant(github):
    with connect() as conn:
        root = kb.create_task(conn, title="source", completion_contract="acme/repo")
        intermediate = kb.create_task(conn, title="review", parents=[root])
        unclaimed = kb.create_task(conn, title="release", parents=[intermediate])
        running = kb.create_task(conn, title="verification", parents=[intermediate])
        assert kb.archive_task(conn, root)
        assert kb.complete_task(conn, intermediate, summary="review completed while source archived")
        claimed = kb.claim_task(conn, running)
        assert claimed is not None
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            from hermes_cli import kanban_db_dispatch as dispatch
            dispatch._set_worker_pid(conn, running, proc.pid)
            event_id = conn.execute(
                "SELECT id FROM task_events WHERE task_id=? AND kind='archived'", (root,),
            ).fetchone()[0]
            assert kb.restore_archived_task(conn, root, archive_event_id=event_id,
                                            completion_contract="acme/repo", reason="acceptance missing")
            assert proc.wait(timeout=10) is not None
            assert kb.get_task(conn, unclaimed).status == "todo"
            fenced = kb.get_task(conn, running)
            assert fenced.status == "todo" and fenced.current_run_id is None
            assert fenced.claim_lock is None and fenced.worker_pid is None
            assert kb.latest_run(conn, running).outcome == "reclaimed"
            assert any(e.kind == "archive_restore_worker_termination" and e.payload["terminated"]
                       for e in kb.list_events(conn, running))
            assert kb.unsatisfied_parents(conn, unclaimed) == [(root, "blocked")]
            assert kb.claim_task(conn, unclaimed) is None
            assert not kb.complete_task(conn, running, summary="cannot bypass source",
                                        expected_run_id=claimed.current_run_id)
            kb.recompute_ready(conn)
            assert kb.get_task(conn, unclaimed).status == "todo"
            assert kb.get_task(conn, running).status == "todo"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.linux_only
def test_archive_restore_never_signals_recycled_worker_pid(github):
    with connect() as conn:
        root = kb.create_task(conn, title="source", completion_contract="acme/repo")
        child = kb.create_task(conn, title="worker", parents=[root])
        assert kb.archive_task(conn, root)
        claimed = kb.claim_task(conn, child)
        assert claimed is not None
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            from hermes_cli import kanban_db_dispatch as dispatch
            dispatch._set_worker_pid(conn, child, proc.pid)
            conn.execute("UPDATE tasks SET worker_started_at = worker_started_at - 1000000 WHERE id=?",
                         (child,))
            event_id = conn.execute("SELECT id FROM task_events WHERE task_id=? AND kind='archived'",
                                    (root,)).fetchone()[0]
            assert kb.restore_archived_task(conn, root, archive_event_id=event_id,
                                            completion_contract="acme/repo", reason="acceptance missing")
            assert proc.poll() is None
            assert kb.get_task(conn, child).status == "todo"
            assert kb.latest_run(conn, child).outcome == "reclaimed"
            assert any(e.kind == "archive_restore_worker_termination" and e.payload["pid_recycled"]
                       for e in kb.list_events(conn, child))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.linux_only
def test_restore_between_spawn_and_registration_stops_unregistered_worker(github, all_assignees_spawnable):
    from hermes_cli import kanban_db_dispatch as dispatch

    with connect() as conn:
        root = kb.create_task(conn, title="source", completion_contract="acme/repo")
        child = kb.create_task(conn, title="release", assignee="forge", parents=[root])
        assert kb.archive_task(conn, root)
        event_id = conn.execute(
            "SELECT id FROM task_events WHERE task_id=? AND kind='archived'", (root,),
        ).fetchone()[0]
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        try:
            def spawn(task, workspace, board=None):
                assert task.id == child and task.current_run_id is not None
                assert kb.restore_archived_task(conn, root, archive_event_id=event_id,
                                                completion_contract="acme/repo", reason="acceptance missing")
                return proc.pid

            result = dispatch.dispatch_once(conn, spawn_fn=spawn)
            assert not result.spawned
            assert proc.wait(timeout=10) is not None
            fenced = kb.get_task(conn, child)
            assert fenced.status == "todo" and fenced.worker_pid is None
            assert fenced.current_run_id is None and fenced.claim_lock is None
            assert kb.latest_run(conn, child).outcome == "reclaimed"
            assert kb.get_task(conn, root).status == "blocked"
            assert not any(e.kind == "spawned" for e in kb.list_events(conn, child))
            assert any(e.kind == "spawn_claim_lost" and e.payload["terminated"]
                       for e in kb.list_events(conn, child))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
