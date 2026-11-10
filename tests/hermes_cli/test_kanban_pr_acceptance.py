"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_pr_acceptance import collect_acceptance


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": state.get("pr_state", "OPEN"),
                    "mergeCommit": {"oid": state["merge_commit"]} if state.get("merge_commit") else None,
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = []
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
                value = []
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"},
                         "state": "closed" if state.get("merge_commit") else "open",
                         "merged": bool(state.get("merge_commit"))}
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
                  "v=json.loads(urllib.request.urlopen(u).read().decode())\n"
                  "# Real gh --paginate prints one JSON document per page.\n"
                  "if isinstance(v,list) and v and all(isinstance(x,dict) for x in v):\n"
                  "    for page in v: print(json.dumps(page))\n"
                  "else:\n"
                  "    print(json.dumps(v))\n")
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

@pytest.mark.platforms("linux")
def test_fork_pr_without_branch_protection_accepts(github, monkeypatch):
    """Real fork path: the fork has NO branch protection at all, so the GraphQL
    branchProtectionRule is None and /rules/branches returns empty. After
    verifying the PR is OPEN with a resolvable head, collect_acceptance must
    accept (ok=True, classification=success) rather than leaving ok=False with a
    'no required checks' detail. The old behavior stranded the real fork PR #7,
    whose head was valid but whose target ref had no CI gate."""

    def no_protection(endpoint, **_k):
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": "a" * 40, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": None}}}}}
        if "rules/branches" in endpoint:
            return [[]]
        if "check-runs" in endpoint:
            return [{"total_count": 0, "check_runs": []}]
        if "statuses" in endpoint:
            return [[]]
        if "pulls/" in endpoint:
            return {"head": {"sha": "a" * 40}, "base": {"ref": "main"}, "state": "open"}
        return {}

    monkeypatch.setattr("hermes_cli.kanban_pr_acceptance._api", no_protection)
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["classification"] == "success", receipt
    assert receipt["ok"] is True, receipt
    assert receipt["head_sha"] == "a" * 40


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


# --- merged PR: completion must not be stranded, and gh 2.46 has no --slurp ---

MERGED_SHA = "6801d9e1b1b5f09bfa1a2849b18f916e0a04795b"
MERGE_COMMIT = "c" * 40


@pytest.fixture
def merged_github(tmp_path, monkeypatch):
    """A GitHub stand-in for an already-MERGED PR on a private repo whose plan has
    no branch protection/rules (403 "Upgrade to GitHub Pro"), reached through a gh
    that rejects --slurp (2.46, as shipped by Ubuntu 24.04)."""
    state = {"state": "MERGED", "required": [], "merge_runs": [], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            merged = state["state"] == "MERGED"
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": MERGED_SHA, "baseRefName": "main",
                    "state": state["state"],
                    "mergeCommit": {"oid": MERGE_COMMIT} if merged else None,
                    "baseRef": {"branchProtectionRule": {
                        "requiredStatusChecks": [{"context": c, "app": {"databaseId": 1}}
                                                 for c in state["required"]]}}}}}}
            elif "/check-runs" in self.path:
                runs = state["merge_runs"] if MERGE_COMMIT in self.path else []
                runs = [{**r, "head_sha": MERGE_COMMIT} for r in runs]
                value = {"total_count": len(runs), "check_runs": runs}
            elif "/statuses" in self.path:
                value = []
            elif "/pulls/" in self.path:
                value = {"head": {"sha": MERGED_SHA}, "base": {"ref": "main"},
                         "state": "closed" if merged else "open", "merged": merged}
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
        "import sys, urllib.request\n"
        "args = sys.argv[1:]\n"
        "endpoint = args[1] if len(args) > 1 else ''\n"
        "if '--slurp' in args:\n"
        "    sys.stderr.write('unknown flag: --slurp\\n')\n"
        "    sys.exit(1)\n"
        "if 'rules/branches' in endpoint:\n"
        "    sys.stderr.write('gh: Upgrade to GitHub Pro or make this repository public "
        "to enable this feature. (HTTP 403)\\n')\n"
        "    sys.exit(1)\n"
        f"url = 'http://127.0.0.1:{server.server_port}/' + endpoint.split('?')[0]\n"
        "print(urllib.request.urlopen(url).read().decode())\n")
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
def test_merged_pr_closes_normally_without_operator_bypass(merged_github):
    """PR #214 shape: merged into main, no required checks (free-plan private repo
    answers rules/branches with 403). The card must complete through the ordinary
    lifecycle — no --force, no archive, no last_failure_error."""
    with connect() as conn:
        tid = kb.create_task(conn, title="merged PR", completion_contract="acme/repo")
        ok = kb.complete_task(conn, tid, summary="done",
                              metadata={"published_pr": "https://github.com/acme/repo/pull/214"})
        task = kb.get_task(conn, tid)
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
            (tid,)).fetchone()[0])
    assert ok is True, receipt
    assert task.status == "done"
    assert task.last_failure_error is None
    assert receipt["ok"] is True and receipt["classification"] == "success", receipt
    assert receipt["state"] == "MERGED" and receipt["merge_sha"] == MERGE_COMMIT, receipt
    assert receipt["merged"] is True and "merged_override" not in receipt, receipt
    assert receipt["required"] == []


@pytest.mark.platforms("linux")
def test_merged_pr_uses_merge_commit_check_evidence(merged_github):
    merged_github["required"] = ["ci/test"]
    merged_github["merge_runs"] = [{"id": 7, "name": "ci/test", "app": {"id": 1},
                                    "status": "completed", "conclusion": "success",
                                    "html_url": "https://github.com/acme/repo/actions/runs/7"}]
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/214")
    assert receipt["ok"] is True, receipt
    assert receipt["checks"][0]["classification"] == "success"
    assert receipt["checks"][0]["head_sha"] == MERGE_COMMIT


@pytest.mark.platforms("linux")
def test_merged_pr_without_head_evidence_is_accepted_and_reported(merged_github):
    merged_github["required"] = ["ci/test"]
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/214")
    assert receipt["ok"] is True, receipt
    assert "already MERGED" in receipt["detail"]
    assert receipt.get("merged_override") == "evidence-unavailable", receipt
    assert receipt["checks"][0]["classification"] == "missing"


@pytest.mark.platforms("linux")
def test_open_pr_with_missing_required_check_is_still_rejected(merged_github):
    """The merged-PR relaxation must not turn into blanket acceptance."""
    merged_github.update(state="OPEN")
    merged_github["required"] = ["ci/test"]
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/214")
    assert receipt["ok"] is False, receipt
    assert receipt["classification"] == "missing"


@pytest.mark.platforms("posix")
def test_profile_login_in_dot_config_gh_is_found_without_env(tmp_path, monkeypatch):
    """`gh auth login` writes <profile_home>/.config/gh/hosts.yml; the gate must
    find that login even when the profile's .env sets no GH_CONFIG_DIR."""
    from pathlib import Path

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    (assignee_home / ".config" / "gh").mkdir(parents=True)
    (assignee_home / ".config" / "gh" / "hosts.yml").write_text(
        "github.com:\n    user: b\n    oauth_token: b-token\n", encoding="utf-8")
    (assignee_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport json, os, pathlib\n"
                  f"pathlib.Path({str(env_dump)!r}).write_text(json.dumps(dict(os.environ)), encoding='utf-8')\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
    captured = json.loads(Path(env_dump).read_text(encoding="utf-8-sig"))
    assert captured["GH_CONFIG_DIR"] == str(assignee_home / ".config" / "gh")
    assert "GH_TOKEN" not in captured and "GITHUB_TOKEN" not in captured
