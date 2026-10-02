"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_acceptance as acceptance
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
    gh.write_text(f"#!{sys.executable}\nimport json,sys,urllib.request\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "value=json.load(urllib.request.urlopen(u))\n"
                  "print('\\n'.join(json.dumps(page) for page in value) if '--paginate' in sys.argv else json.dumps(value))\n")
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


@pytest.mark.platforms("linux")
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


@pytest.mark.platforms("linux")
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


def test_paginated_api_works_with_gh_without_slurp(monkeypatch):
    from subprocess import CompletedProcess

    def run(command, **kwargs):
        assert "--paginate" in command
        assert "--slurp" not in command
        return CompletedProcess(command, 0, '{"check_runs":[{"id":1}]}\n{"check_runs":[{"id":2}]}\n')

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    assert acceptance._api("repos/acme/repo/commits/abc/check-runs", paginate=True) == [
        {"check_runs": [{"id": 1}]}, {"check_runs": [{"id": 2}]}
    ]


def test_open_pr_accepts_exact_head_passing_checks_when_rules_api_is_paywalled(monkeypatch):
    from subprocess import CalledProcessError

    sha = "a" * 40

    def api(endpoint, *, query=None, paginate=False):
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": None}
            }}}}
        if "/rules/branches/" in endpoint:
            raise CalledProcessError(
                1, "gh api",
                stderr="Upgrade to GitHub Pro or make this repository public to enable this feature. (HTTP 403)",
            )
        if "/check-runs" in endpoint:
            return [{"total_count": 1, "check_runs": [{
                "id": 42, "name": "Lint, typecheck, migrations, tests", "head_sha": sha,
                "app": {"id": 1}, "status": "completed", "conclusion": "success",
            }]}]
        if "/statuses" in endpoint:
            return [[]]
        if "/pulls/" in endpoint:
            return {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open", "merged": False}
        raise AssertionError(endpoint)

    monkeypatch.setattr(acceptance, "_api", api)
    url = "https://github.com/acme/repo/pull/7"
    receipt = acceptance.collect_acceptance(url, url)
    assert receipt["ok"] is True
    assert receipt["classification"] == "success"
    assert receipt["head_sha"] == sha
    assert receipt["checks"] == [{
        "name": "Lint, typecheck, migrations, tests", "id": 42,
        "url": None, "head_sha": sha, "classification": "success", "conclusion": "success",
    }]


def test_open_pr_rejects_stale_head_success_when_rules_api_is_paywalled(monkeypatch):
    from subprocess import CalledProcessError

    current_sha = "a" * 40
    stale_sha = "b" * 40

    def api(endpoint, *, query=None, paginate=False):
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": current_sha, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": None}
            }}}}
        if "/rules/branches/" in endpoint:
            raise CalledProcessError(
                1, "gh api",
                stderr="Upgrade to GitHub Pro or make this repository public to enable this feature. (HTTP 403)",
            )
        if "/check-runs" in endpoint:
            return [{"total_count": 1, "check_runs": [{
                "id": 42, "name": "Lint, typecheck, migrations, tests", "head_sha": stale_sha,
                "app": {"id": 1}, "status": "completed", "conclusion": "success",
            }]}]
        if "/statuses" in endpoint:
            return [[]]
        if "/pulls/" in endpoint:
            return {"head": {"sha": current_sha}, "base": {"ref": "main"}, "state": "open", "merged": False}
        raise AssertionError(endpoint)

    monkeypatch.setattr(acceptance, "_api", api)
    url = "https://github.com/acme/repo/pull/7"
    receipt = acceptance.collect_acceptance(url, url)
    assert receipt["ok"] is False
    assert receipt["classification"] == "stale"
    assert receipt["checks"][0]["head_sha"] == stale_sha


def test_merged_pr_accepts_passing_checks_when_rules_api_is_paywalled(monkeypatch):
    from subprocess import CalledProcessError

    sha = "a" * 40
    def api(endpoint, *, query=None, paginate=False):
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": sha, "baseRefName": "main", "state": "MERGED",
                "baseRef": {"branchProtectionRule": None}
            }}}}
        if "/rules/branches/" in endpoint:
            raise CalledProcessError(1, "gh api", stderr="Upgrade to GitHub Pro or make this repository public to enable this feature. (HTTP 403)")
        if "/check-runs" in endpoint:
            return [{"total_count": 1, "check_runs": [{"id": 42, "name": "test", "head_sha": sha,
                     "app": {"id": 1}, "status": "completed", "conclusion": "success"}]}]
        if "/statuses" in endpoint:
            return [[]]
        if "/pulls/" in endpoint:
            return {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "closed", "merged": True}
        raise AssertionError(endpoint)

    monkeypatch.setattr(acceptance, "_api", api)
    url = "https://github.com/acme/repo/pull/7"
    receipt = acceptance.collect_acceptance(url, url)
    assert receipt["ok"] is True
    assert receipt["classification"] == "success"
    assert receipt["checks"][0]["name"] == "test"
