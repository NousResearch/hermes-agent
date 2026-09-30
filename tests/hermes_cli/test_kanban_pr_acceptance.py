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
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": [],
             "classic_required": True, "rules_403": None, "pr_state": "OPEN",
             "merge_commit": None, "final_merge_commit": None, "compare_status": "ahead",
             "merge_base": None, "base_sha": "d" * 40, "base_sha_after_compare": None,
             "branch_reads": 0, "check_runs": None, "statuses": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": state["pr_state"],
                    "mergeCommit": {"oid": state["merge_commit"]} if state["pr_state"] == "MERGED" and state["merge_commit"] else None,
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}
                    ] if state["classic_required"] else []}}}}}}
            elif "/rules/branches/" in self.path:
                if state["rules_403"]:
                    body = {"message": "Upgrade to GitHub Pro or make this repository public to enable this feature.",
                            "documentation_url": "https://docs.github.com/rest/repos/rules",
                            "status": "403"}
                    if state["rules_403"] == "slurp":
                        body = [body]
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(json.dumps(body).encode())
                    return
                value = [[]]
            elif "/compare/" in self.path:
                merge_commit = state["merge_commit"]
                value = {"status": state["compare_status"],
                         "base_commit": {"sha": merge_commit},
                         "merge_base_commit": {"sha": state["merge_base"] or merge_commit}}
            elif "/check-runs" in self.path:
                if state["check_runs"] is not None:
                    runs = [dict(run) for run in state["check_runs"]]
                    value = [{"total_count": len(runs), "check_runs": runs}]
                else:
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
                value = [state["statuses"]]
            elif "/branches/" in self.path:
                state["branch_reads"] += 1
                base_sha = (state["base_sha_after_compare"] if state["branch_reads"] > 1
                            and state["base_sha_after_compare"] else state["base_sha"])
                value = {"commit": {"sha": base_sha}}
            elif "/pulls/" in self.path:
                value = {"head": {"sha": state.get("final_head", sha)}, "base": {"ref": "main"},
                         "state": "closed" if state["pr_state"] == "MERGED" else "open",
                         "merged": state["pr_state"] == "MERGED",
                         "merge_commit_sha": state["final_merge_commit"] or state["merge_commit"]}
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
    gh.write_text(f"#!{sys.executable}\nimport json,sys,urllib.request,urllib.error\n"
    f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
    "try:\n print(urllib.request.urlopen(u).read().decode())\n"
    "except urllib.error.HTTPError as error:\n"
    " body=error.read().decode()\n sys.stdout.write(body)\n"
    " try:\n  payload=json.loads(body)\n  while isinstance(payload,list) and len(payload)==1: payload=payload[0]\n  message=payload.get('message','') if isinstance(payload,dict) else ''\n"
    " except ValueError:\n  message=''\n"
    " sys.stderr.write(f'gh: {message} (HTTP {error.code})\\n')\n sys.exit(1)\n")
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


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("body_shape", ["dict", "slurp"])
@pytest.mark.parametrize("conclusion", ["success", "failure"])
def test_rules_feature_403_keeps_classic_required_evidence(github, body_shape, conclusion):
    github.update(rules_403=body_shape, conclusion=conclusion)
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        ok = kb.complete_task(conn, tid, result="done", metadata={
            "published_pr": "https://github.com/acme/repo/pull/7"})
        assert ok is (conclusion == "success")
        receipts = [json.loads(row[0]) for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
        assert receipts[-1]["required"] == [{"context": "required", "app_id": 1}]
        assert receipts[-1]["classification"] == conclusion
        assert "Upgrade to GitHub Pro" not in json.dumps(receipts[-1])


@pytest.mark.platforms("posix")
def test_rules_feature_403_does_not_make_empty_required_set_succeed(github):
    github.update(rules_403="dict", classic_required=False)
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={
            "published_pr": "https://github.com/acme/repo/pull/7"})
        receipts = [json.loads(row[0]) for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
        assert receipts[-1]["required"] == []
        assert receipts[-1]["classification"] == "missing"
        assert "No repository-required checks are configured" in receipts[-1]["detail"]


def test_merged_pr_can_complete_from_landed_all_green_current_head_evidence(github):
    """A landed merge uses exact-head check evidence, never vacuous empty-required success."""
    merge_commit = "c" * 40
    github.update(pr_state="MERGED", merge_commit=merge_commit, rules_403="dict",
                  classic_required=False, check_runs=[
                      {"id": i, "name": name, "head_sha": "a" * 40, "app": {"id": i},
                       "status": "completed", "conclusion": "success"}
                      for i, name in enumerate(("delete-head-branch", "preview", "schema-contract", "guard", "secret-scan"), 1)
                  ], statuses=[
                      {"id": 100, "context": "Vercel", "state": "pending", "target_url": "https://example.test/vercel"},
                      {"id": 101, "context": "Devin Review", "state": "pending", "target_url": "https://example.test/devin"},
                  ])
    with connect() as conn:
        tid = kb.create_task(conn, title="Landed publish", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, result="done", metadata={
            "published_pr": "https://github.com/acme/repo/pull/7"})
        receipts = [json.loads(row[0]) for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
    receipt = receipts[-1]
    assert receipt["ok"] is True
    assert receipt["classification"] == "success"
    assert receipt["merge_commit_sha"] == merge_commit
    assert receipt["landing_verified"] is True
    assert receipt["required"] == []
    assert receipt["ruleset_evidence"] == "feature_unavailable"
    assert receipt["checks_evidence"] == "all_current_head_check_runs_pass"
    assert "was not treated as an empty required set" in receipt["detail"]
    assert receipt["base_sha"] == github["base_sha"]
    assert {check["name"] for check in receipt["checks"]} == {
        "delete-head-branch", "preview", "schema-contract", "guard", "secret-scan"}
    assert all(check["head_sha"] == "a" * 40 and check["classification"] == "success"
               for check in receipt["checks"])
    assert any("/compare/" in path for path in github["requests"])


@pytest.mark.parametrize("change", [
    "missing_merge_commit", "not_landed", "unrelated_merge_base", "failed_check", "required_failure",
    "pending_check", "stale_check", "no_checks", "changed_head", "changed_merge_commit", "base_moved",
    "missing_head",
])
def test_merged_pr_rejects_unknown_or_failed_landing_evidence(github, change):
    merge_commit = "c" * 40
    checks = [{"id": 1, "name": "guard", "head_sha": "a" * 40, "app": {"id": 1},
               "status": "completed", "conclusion": "success"}]
    github.update(pr_state="MERGED", merge_commit=merge_commit, rules_403="dict",
                  classic_required=False, check_runs=checks)
    if change == "missing_merge_commit":
        github["merge_commit"] = None
    elif change == "not_landed":
        github["compare_status"] = "behind"
    elif change == "unrelated_merge_base":
        github["merge_base"] = "b" * 40
    elif change == "failed_check":
        github["check_runs"] = [{**checks[0], "conclusion": "failure"}]
    elif change == "required_failure":
        github.update(classic_required=True, rules_403=None, check_runs=[
            {"id": 1, "name": "required", "head_sha": "a" * 40, "app": {"id": 1},
             "status": "completed", "conclusion": "failure"}])
    elif change == "pending_check":
        github["check_runs"] = [{**checks[0], "conclusion": "pending"}]
    elif change == "stale_check":
        github["check_runs"] = [{**checks[0], "head_sha": "b" * 40}]
    elif change == "no_checks":
        github["check_runs"] = []
    elif change == "changed_head":
        github["final_head"] = "b" * 40
    elif change == "changed_merge_commit":
        github["final_merge_commit"] = "b" * 40
    elif change == "base_moved":
        github["base_sha_after_compare"] = "e" * 40
    elif change == "missing_head":
        github["head"] = None
    with connect() as conn:
        tid = kb.create_task(conn, title=f"Landed publish {change}", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={
            "published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    if change == "missing_merge_commit":
        assert "authoritative merge commit" in receipt["detail"]
    elif change == "not_landed":
        assert "ancestry" in receipt["detail"]
    elif change == "unrelated_merge_base":
        assert "ancestry" in receipt["detail"]
    elif change in {"failed_check", "required_failure"}:
        assert receipt["classification"] == "failure"
    elif change == "pending_check":
        assert receipt["classification"] == "pending"
    elif change == "stale_check":
        assert receipt["classification"] == "stale"
    elif change == "no_checks":
        assert "no current-head check" in receipt["detail"]
    elif change in {"changed_head", "changed_merge_commit", "base_moved"}:
        assert receipt["classification"] == "stale"
    elif change == "missing_head":
        assert receipt["classification"] == "infra"


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
