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


# --- #122689: acceptance must read the repo as the ASSIGNEE profile's gh login ---

@pytest.mark.platforms("posix")
def test_acceptance_runs_gh_as_the_assignee_profile(tmp_path, monkeypatch):
    """The gh child env carries the assignee's own GH credentials (its .env),
    never the ambient/launch residue, and an invisible repo is classified
    `auth` naming the repository — not a retryable `infra` failure."""
    from pathlib import Path

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GH_TOKEN=b-token\n", encoding="utf-8")
    # Ambient residue that must NOT decide the login.
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport json, os\n"
                  f"json.dump(dict(os.environ), open({str(env_dump)!r}, 'w'))\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        receipts = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
        assert receipts[-1]["classification"] == "auth"
        assert "acme/repo" in receipts[-1]["detail"]
    captured = json.loads(env_dump.read_text())
    assert captured["GH_TOKEN"] == "b-token"
    assert captured.get("GH_CONFIG_DIR") != "/nonexistent/launch/gh"
    assert "credentials" in (kb.get_task(conn, tid).last_failure_error or "")


@pytest.mark.platforms("posix")
def test_assignee_without_own_gh_login_never_falls_through_to_ambient_login(tmp_path, monkeypatch):
    """An assignee profile with no GH_TOKEN/GH_CONFIG_DIR of its own must not inherit the
    launch user's ~/.config/gh (HOME/XDG_CONFIG_HOME stay the launch process's): gh is pinned
    to a profile-owned config dir, its 'not logged in' exit is classified `auth` naming the profile."""
    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "launch-xdg"))

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Real gh: GH_CONFIG_DIR wins; a config dir without hosts.yml means "not logged in" (exit 4).
    gh.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
                  f"pathlib.Path({str(env_dump)!r}).write_text(json.dumps(dict(os.environ)), encoding='utf-8')\n"
                  "if 'GH_CONFIG_DIR' in os.environ and not os.path.exists(os.environ['GH_CONFIG_DIR']):\n"
                  "    sys.exit(4)\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'b'" in receipt["detail"] and "no login" in receipt["detail"]
    captured = json.loads(env_dump.read_text(encoding="utf-8-sig"))
    assert captured["GH_CONFIG_DIR"] == str(assignee_home / "gh")
    assert "GH_TOKEN" not in captured and "GITHUB_TOKEN" not in captured


def test_assigned_card_with_unresolvable_profile_is_auth_not_ambient(tmp_path, monkeypatch):
    """A card assigned to a profile that no longer exists must not run gh as the completing
    process's ambient login: classification `auth` naming the profile, gh never invoked."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # any gh spawn would fail as infra
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-ghost", completion_contract="acme/repo", assignee="ghost")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'ghost'" in receipt["detail"] and "cannot be resolved" in receipt["detail"]


@pytest.mark.parametrize("status, accepted", [
    ("success", True),
    ("failed", False),
    ("canceled", False),
    ("running", False),
    ("pending", False),
])
def test_gitlab_mr_completion_requires_current_successful_pipeline(tmp_path, monkeypatch, status, accepted):
    """An OWNER/REPO contract binds to an exact GitLab MR and gates on its
    current-head pipeline, with the same terminal-write path as GitHub."""
    from hermes_cli import kanban_pr_acceptance as acceptance

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    head = "a" * 40
    calls = []

    def fake_glab(endpoint, *, profile_home=None):
        calls.append(endpoint)
        if endpoint.endswith("/pipelines?per_page=100"):
            return [{"id": 42, "sha": head, "status": status,
                     "web_url": "https://gitlab.com/acme/repo/-/pipelines/42"}]
        return {"sha": head, "target_branch": "main", "state": "opened",
                "web_url": "https://gitlab.com/acme/repo/-/merge_requests/7"}

    monkeypatch.setattr(acceptance, "_glab_api", fake_glab)
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        ok = kb.complete_task(
            conn, tid, result="done",
            metadata={"published_pr": "https://gitlab.com/acme/repo/-/merge_requests/7"},
        )
        assert ok is accepted
        assert (kb.get_task(conn, tid).status == "done") is accepted
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])
    assert receipt["head_sha"] == head
    assert receipt["checks"][0]["classification"] == ("success" if accepted else ("pending" if status in {"running", "pending"} else "failure"))
    assert len(calls) == 3  # MR, pipelines, MR readback


def test_gitlab_exact_contract_and_repository_binding_are_fail_closed(tmp_path, monkeypatch):
    from hermes_cli import kanban_pr_acceptance as acceptance

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    exact = "https://gitlab.com/acme/repo/-/merge_requests/7"
    assert acceptance.validate_contract(exact) == exact
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert not kb.complete_task(
            conn, tid, result="done",
            metadata={"published_pr": "https://gitlab.com/other/repo/-/merge_requests/7"},
        )
        assert kb.get_task(conn, tid).status != "done"


@pytest.mark.platforms("posix")
def test_acceptance_runs_glab_as_the_assignee_profile(tmp_path, monkeypatch):
    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GITLAB_TOKEN=b-token\n", encoding="utf-8")
    monkeypatch.setenv("GITLAB_TOKEN", "launch-token")
    monkeypatch.setenv("GLAB_CONFIG_DIR", "/nonexistent/launch/glab")
    env_dump = tmp_path / "glab_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    glab = shim / "glab"
    glab.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"json.dump(dict(os.environ), open({str(env_dump)!r}, 'w'))\n"
        "endpoint=sys.argv[2]\nhead='a'*40\n"
        "print(json.dumps([{'id':42,'sha':head,'status':'success'}] if '/pipelines?' in endpoint "
        "else {'sha':head,'target_branch':'main','state':'opened'}))\n"
    )
    glab.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert kb.complete_task(conn, tid, result="done",
                                metadata={"published_pr": "https://gitlab.com/acme/repo/-/merge_requests/7"})
    captured = json.loads(env_dump.read_text())
    assert captured["GITLAB_TOKEN"] == "b-token"
    assert captured.get("GLAB_CONFIG_DIR") != "/nonexistent/launch/glab"


@pytest.mark.platforms("posix")
def test_assignee_without_glab_login_never_uses_ambient_token(tmp_path, monkeypatch):
    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("GITLAB_TOKEN", "launch-token")
    monkeypatch.setenv("GITLAB_CI_TOKEN", "launch-ci-token")
    monkeypatch.setenv("GLAB_CONFIG_DIR", "/nonexistent/launch/glab")
    env_dump = tmp_path / "glab_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    glab = shim / "glab"
    glab.write_text(
        f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
        f"pathlib.Path({str(env_dump)!r}).write_text(json.dumps(dict(os.environ)))\n"
        "print('not logged in', file=sys.stderr); sys.exit(1)\n"
    )
    glab.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://gitlab.com/acme/repo/-/merge_requests/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "GitLab" in receipt["detail"] and "'b'" in receipt["detail"]
    captured = json.loads(env_dump.read_text())
    assert captured["GLAB_CONFIG_DIR"] == str(assignee_home / "glab-cli")
    assert "GITLAB_TOKEN" not in captured and "GITLAB_CI_TOKEN" not in captured


def test_gitlab_missing_and_stale_evidence_fail_closed(monkeypatch):
    from hermes_cli import kanban_pr_acceptance as acceptance
    head = "a" * 40

    def missing(endpoint, *, profile_home=None):
        if "/pipelines?" in endpoint:
            return []
        return {"sha": head, "target_branch": "main", "state": "opened"}

    monkeypatch.setattr(acceptance, "_glab_api", missing)
    receipt = acceptance.collect_acceptance(
        "acme/repo", "https://gitlab.com/acme/repo/-/merge_requests/7")
    assert not receipt["ok"] and receipt["classification"] == "missing"

    mr_reads = 0
    def stale(endpoint, *, profile_home=None):
        nonlocal mr_reads
        if "/pipelines?" in endpoint:
            return [{"id": 42, "sha": head, "status": "success"}]
        mr_reads += 1
        return {"sha": head if mr_reads == 1 else "b" * 40,
                "target_branch": "main", "state": "opened"}

    monkeypatch.setattr(acceptance, "_glab_api", stale)
    receipt = acceptance.collect_acceptance(
        "acme/repo", "https://gitlab.com/acme/repo/-/merge_requests/7")
    assert not receipt["ok"] and receipt["classification"] == "stale"
