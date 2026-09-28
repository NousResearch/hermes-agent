"""GitLab completion contracts: exact-head pipeline evidence (and, for
``gitlab-merged``, the merge itself) gate ``complete_task``."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_pr_acceptance import validate_contract

MR = "https://gitlab.example.com/grp/sub/proj/-/merge_requests/7"
ENDPOINT = "projects/grp%2Fsub%2Fproj/merge_requests/7"


@pytest.fixture
def gitlab(tmp_path, monkeypatch):
    state = {"status": "success", "sha": "a" * 40, "mr_state": "opened", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            if self.path != "/" + ENDPOINT:
                self.send_error(404)
                return
            sha = state["sha"]
            value = {"sha": sha, "target_branch": "main", "state": state["mr_state"],
                     "head_pipeline": None if state.get("no_pipeline") else {
                         "id": 42, "status": state["status"], "web_url": "https://gitlab.example.com/p/42",
                         "sha": "b" * 40 if state.get("stale") else sha}}
            if state.get("head_change"):
                state["sha"] = "c" * 40
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
    glab = shim / "glab"
    # argv: glab api <endpoint> --hostname <host>
    glab.write_text(f"#!{sys.executable}\nimport sys,urllib.request\n"
                    "assert sys.argv[1] == 'api' and sys.argv[3:] == ['--hostname', 'gitlab.example.com'], sys.argv\n"
                    f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                    "print(urllib.request.urlopen(u).read().decode())\n")
    glab.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _complete(conn, contract, published=MR, title="Publish"):
    tid = kb.create_task(conn, title=title, completion_contract=contract)
    ok = kb.complete_task(conn, tid, result="done", metadata={"published_pr": published} if published else None)
    return tid, ok


def test_validate_contract_accepts_gitlab_forms():
    for good in ("gitlab:gitlab.example.com/grp/proj", "gitlab-merged:gitlab.example.com/grp/sub/proj",
                 MR, MR + "#merged"):
        assert validate_contract(good) == good
    for bad in ("gitlab:gitlab.example.com/proj", "gitlab:grp/proj", MR + "#open", "http://h/g/p/-/merge_requests/1"):
        with pytest.raises(ValueError):
            validate_contract(bad)


def test_pipeline_evidence_gates_completion(gitlab):
    with connect() as conn:
        for status in ("failed", "running", "canceled", "skipped", "manual", "success"):
            gitlab["status"] = status
            tid, ok = _complete(conn, "gitlab:gitlab.example.com/grp/sub/proj")
            assert ok is (status == "success"), status
            task = kb.get_task(conn, tid)
            assert (task.status == "done") is ok
            assert task.completion_contract == MR  # bound once, to the published MR
            receipt = json.loads(conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
                (tid,)).fetchone()[0])
            assert receipt["head_sha"] == "a" * 40 and receipt["checks"][0]["id"] == 42
        for fault in ("no_pipeline", "stale", "head_change"):
            gitlab.update(status="success", sha="a" * 40)
            gitlab[fault] = True
            tid, ok = _complete(conn, "gitlab:gitlab.example.com/grp/sub/proj", title=fault)
            assert not ok and kb.get_task(conn, tid).status != "done", fault
            gitlab.pop(fault)
        gitlab["sha"] = "a" * 40
        # A sibling project cannot satisfy the declaration; omission does not downgrade it.
        _, ok = _complete(conn, "gitlab:gitlab.example.com/grp/other", title="sibling")
        assert not ok
        _, ok = _complete(conn, "gitlab:gitlab.example.com/grp/sub/proj", published=None, title="omitted")
        assert not ok


def test_merged_contract_waits_for_the_merge(gitlab):
    with connect() as conn:
        tid, ok = _complete(conn, "gitlab-merged:gitlab.example.com/grp/sub/proj")
        assert not ok
        task = kb.get_task(conn, tid)
        assert task.completion_contract == MR + "#merged"
        assert "merged" in (task.last_failure_error or "")
        gitlab["mr_state"] = "merged"
        assert kb.complete_task(conn, tid, result="merged", metadata={"published_pr": MR})
        assert kb.get_task(conn, tid).status == "done"
