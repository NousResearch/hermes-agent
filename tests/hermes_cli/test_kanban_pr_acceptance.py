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
        def _json(self, code, value):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/__mode":
                # Out-of-band control channel for the gh shim (see below).
                self._json(200, {"fail_stderr": state.get("fail_stderr")})
                return
            if state.get("deny"):
                # A real identity failure: the token may not see the resource at all.
                self._json(403, {"message": "Resource not accessible by personal access token"})
                return
            if "/rules/branches/" in self.path and state.get("rules_403"):
                # What a private repo on a free plan really returns (#t_518d9dc1).
                self._json(403, {"message": "Upgrade to GitHub Pro or make this repository public to enable this feature."})
                return
            if state.get("checks_403_plan") and "/check-runs" in self.path:
                # The same plan-gating text from an endpoint that is NOT a rules read:
                # this must stay an auth failure, never "no checks configured".
                self._json(403, {"message": "Upgrade to GitHub Pro or make this repository public to enable this feature."})
                return
            if self.path == "/graphql":
                if state.get("no_repository"):
                    self._json(200, {"data": {"repository": None}})
                    return
                rule = None if state.get("no_protection_rule") else {"requiredStatusChecks": [
                    {"context": "required", "app": {"databaseId": 1}}]}
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": rule}}}}}
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
                self._json(404, {"message": "Not Found"})
                return
            self._json(200, value)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Faithful enough to matter: one compact JSON document per page like
    # `gh api --paginate --jq tojson`, and gh's own failure shape
    # ("gh: <message> (HTTP 403)" on stderr, exit 1) so 401/403/404 handling is real.
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, urllib.error, urllib.request\n"
        f"port = {server.server_port}\n"
        "args = sys.argv[1:]\n"
        "mode = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}/__mode').read().decode())\n"
        "if mode.get('fail_stderr'):\n"
        "    sys.stderr.write(mode['fail_stderr'])\n"
        "    sys.exit(1)\n"
        "try:\n"
        "    body = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}/' + args[1]).read().decode())\n"
        "except urllib.error.HTTPError as exc:\n"
        "    payload = exc.read().decode()\n"
        "    try:\n"
        "        msg = json.loads(payload).get('message', payload)\n"
        "    except ValueError:\n"
        "        msg = payload\n"
        "    sys.stderr.write(f'gh: {msg} (HTTP {exc.code})\\n')\n"
        "    sys.exit(1)\n"
        "pages = body if isinstance(body, list) else [body]\n"
        "for page in pages:\n"
        "    print(json.dumps(page, separators=(',', ':')))\n"
    )
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
    (assignee_home / ".env").write_text("GH_TOKEN=b-token\nBWS_ACCESS_TOKEN=b-vault\n"
                                        "OP_SERVICE_ACCOUNT_TOKEN=b-op\n", encoding="utf-8")
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
    # gh never needs the profile's vault bootstrap token, even though its .env holds one.
    assert "BWS_ACCESS_TOKEN" not in captured and "OP_SERVICE_ACCOUNT_TOKEN" not in captured
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


# --- #t_518d9dc1: private repo on a free plan; gh older than --slurp ---

def _last_receipt(conn, tid):
    return json.loads(conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' "
        "ORDER BY rowid DESC", (tid,)).fetchone()[0])


@pytest.mark.platforms("posix")
def test_plan_gated_branch_rules_are_missing_not_auth(github):
    """A private repo on a free plan cannot expose branch rules (HTTP 403
    "Upgrade to GitHub Pro..."). That is 'no repository-required checks visible',
    i.e. classification `missing` with the documented local-only guidance —
    never `auth`, which today tells the operator to fix credentials that work."""
    github.update(rules_403=True, no_protection_rule=True)
    with connect() as conn:
        tid = kb.create_task(conn, title="free-plan-private", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = _last_receipt(conn, tid)
        assert conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()[0] != "done"
    assert receipt["classification"] == "missing", receipt
    assert receipt["plan_gated"] is True
    assert receipt["required"] == []
    assert "explicitly use a local-only contract" in receipt["detail"]
    assert "credentials" not in receipt["detail"]


@pytest.mark.platforms("posix")
def test_identity_refusals_stay_auth_in_both_directions(github):
    """The plan-gated carve-out must not widen the auth net: a 403 the token
    cannot use, and a private repo GraphQL resolves to null, both stay `auth`."""
    github.update(deny=True)
    with connect() as conn:
        tid = kb.create_task(conn, title="denied", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = _last_receipt(conn, tid)
    assert receipt["classification"] == "auth", receipt
    assert "HTTP 403" in receipt["detail"] and "fix that profile's GitHub credentials" in receipt["detail"]
    assert "plan_gated" not in receipt

    github.pop("deny")
    github.update(no_repository=True)
    with connect() as conn:
        tid = kb.create_task(conn, title="invisible", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = _last_receipt(conn, tid)
    assert receipt["classification"] == "auth", receipt
    assert "404" in receipt["detail"] and "acme/repo" in receipt["detail"]


@pytest.mark.platforms("posix")
def test_plan_gated_body_outside_the_rules_read_is_still_auth(github):
    """Same plan-gating sentence on a check-runs read is NOT excused: only the
    rules/protection endpoints may fall through as 'no checks configured'."""
    github.update(checks_403_plan=True)
    with connect() as conn:
        tid = kb.create_task(conn, title="checks-plan-gated", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = _last_receipt(conn, tid)
    assert receipt["classification"] == "auth", receipt
    assert "check-runs" in receipt["detail"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("stderr", ["unknown flag: --slurp\n", "unknown flag: --jq\n"])
def test_unsupported_gh_pagination_flag_is_actionable_infra(github, stderr):
    """A gh too old for the flags we pass (2.46 has no --slurp) must surface as
    infra naming the flag and the remedy, not a generic 'unavailable or incomplete'."""
    github["fail_stderr"] = stderr
    with connect() as conn:
        tid = kb.create_task(conn, title="old-gh", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = _last_receipt(conn, tid)
    assert receipt["classification"] == "infra", receipt
    assert stderr.split(": ")[1].strip() in receipt["detail"]
    assert "upgrade gh on this host" in receipt["detail"]
