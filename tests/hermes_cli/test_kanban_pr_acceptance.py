"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import subprocess
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


@pytest.mark.platforms("posix")
def test_gh_env_carries_only_gh_credentials_not_the_whole_profile_scope(tmp_path, monkeypatch):
    """#134669: gh needs only its own login, so the child env must not overlay the assignee
    profile's entire secret scope — provider keys and adapter secrets stay out — while the
    #122689 identity guarantees still hold (the assignee's own GH_TOKEN wins over ambient)."""
    from hermes_cli.kanban_pr_acceptance import _gh_env

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text(
        "GH_TOKEN=b-token\nWEIXIN_ADAPTER_SECRET=wx-secret-xyz\nPROVIDER_API_KEY=sk-zzz-999\n",
        encoding="utf-8")
    # Ambient residue that must not decide the login.
    monkeypatch.setenv("GH_TOKEN", "launch-token")

    env = _gh_env(str(assignee_home)) or {}
    assert env["GH_TOKEN"] == "b-token"  # assignee identity preserved over ambient
    assert "WEIXIN_ADAPTER_SECRET" not in env  # non-gh profile secret stays out
    assert "PROVIDER_API_KEY" not in env  # the assignee's provider key stays out too
    # Ambient GH_CONFIG_DIR is dropped and no fail-closed pin fires (a GH_TOKEN exists),
    # so gh's config dir is simply absent from the child env.
    assert "GH_CONFIG_DIR" not in env


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("token_key", ["GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"])
def test_enterprise_only_profile_still_gets_the_fail_closed_pin(tmp_path, monkeypatch, token_key):
    """Every acceptance call targets github.com, where gh ignores the enterprise token vars and
    GH_HOST. A profile holding only an enterprise login therefore has no github.com login: it
    must get the pin, not fall through to the launcher's ~/.config/gh/hosts.yml (#134669)."""
    from hermes_cli.kanban_pr_acceptance import _gh_env

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text(f"GH_HOST=ghe.corp\n{token_key}=ghe-token\n", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "launch-token")

    env = _gh_env(str(assignee_home)) or {}
    assert env.get("GH_TOKEN") != "launch-token"
    assert token_key not in env and "GH_HOST" not in env
    assert env["GH_CONFIG_DIR"] == str(assignee_home / "gh")


@pytest.mark.platforms("posix")
def test_gh_env_drops_ambient_enterprise_login_on_loginless_profile(tmp_path, monkeypatch):
    """#134669 follow-up: the enterprise login keys are on no strip list, so an ambient
    GHE token (the launcher's) surviving into a routed profile with no gh login of its
    own would satisfy the fail-closed check and run gh as the launcher — the #122689
    identity leak for GitHub Enterprise. The trio must be dropped, and the pin must fire."""
    from hermes_cli.kanban_pr_acceptance import _gh_env

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("OPENAI_API_KEY=sk-aaa\n", encoding="utf-8")
    # Ambient enterprise login the launcher happens to carry.
    monkeypatch.setenv("GH_HOST", "ghe.corp")
    monkeypatch.setenv("GH_ENTERPRISE_TOKEN", "launch-ghe")
    monkeypatch.setenv("GITHUB_ENTERPRISE_TOKEN", "launch-ghe-2")

    env = _gh_env(str(assignee_home)) or {}
    assert "GH_HOST" not in env and "GH_ENTERPRISE_TOKEN" not in env
    assert "GITHUB_ENTERPRISE_TOKEN" not in env
    # No login of its own survived, so the fail-closed pin fires instead of a fall-through.
    assert env["GH_CONFIG_DIR"] == str(assignee_home / "gh")


@pytest.mark.platforms("posix")
def test_gh_env_keeps_the_assignee_profiles_own_proxy(tmp_path, monkeypatch):
    """A routed child loses the launcher's .env, so a proxy in the assignee's own .env must still
    reach gh or it cannot reach github.com; a proxy is not a login, so the pin still fires."""
    from hermes_cli.kanban_pr_acceptance import _gh_env

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("HTTPS_PROXY=http://corp-proxy:3128\nNO_PROXY=localhost\n",
                                        encoding="utf-8")

    env = _gh_env(str(assignee_home)) or {}
    assert env["HTTPS_PROXY"] == "http://corp-proxy:3128" and env["NO_PROXY"] == "localhost"
    assert env["GH_CONFIG_DIR"] == str(assignee_home / "gh")


@pytest.mark.platforms("posix")
def test_gh_env_empty_transport_var_in_assignee_env_clears_the_inherited_one(tmp_path, monkeypatch):
    """``NO_PROXY=`` in the assignee's .env clears a shell-wide ``NO_PROXY=*`` (else gh would bypass
    the assignee's proxy); an empty login key still is no login, so the pin fires."""
    from hermes_cli.kanban_pr_acceptance import _gh_env

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("NO_PROXY", "*")
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GH_TOKEN=\nHTTPS_PROXY=http://corp-proxy:3128\nNO_PROXY=\n",
                                        encoding="utf-8")

    env = _gh_env(str(assignee_home)) or {}
    assert env["HTTPS_PROXY"] == "http://corp-proxy:3128" and env["NO_PROXY"] == ""
    assert "GH_TOKEN" not in env and env["GH_CONFIG_DIR"] == str(assignee_home / "gh")


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


# --- a gh-side blip must never read as an auth problem, and never lose the failed phase ---

def _gh_shim(tmp_path, monkeypatch, script):
    """A stand-in for the real gh CLI on PATH; `script` is its whole body."""
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\n{script}", encoding="utf-8")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])


# Body shared by the gh-side tests: a counter in $GH_CALLS_FILE, an injected failure
# at $GH_FAIL_ON (substring of argv[2]) for every call while $GH_FAIL_STICKY=1, only
# the first matching call otherwise. Any other call returns a complete green page.
_GH_SCRIPT = (
    "import json, os, sys\n"
    "endpoint = sys.argv[2]\n"
    "path = os.environ['GH_CALLS_FILE']\n"
    "calls = int(open(path).read()) if os.path.exists(path) else 0\n"
    "calls += 1\n"
    "open(path, 'w').write(str(calls))\n"
    "if os.environ['GH_FAIL_ON'] in endpoint and "
    "(os.environ.get('GH_FAIL_STICKY') == '1' or calls == int(os.environ.get('GH_FAIL_AT', '1'))):\n"
    "    sys.stderr.write('HTTP 502 bad gateway token=secret-leak\\n'); sys.exit(1)\n"
    "sha = 'a' * 40\n"
    "run = {'id': 42, 'name': 'required', 'head_sha': sha, 'app': {'id': 1},\n"
    "       'status': 'completed', 'conclusion': 'success', 'html_url': 'https://github.com/acme/repo/c/42'}\n"
    "value = {\n"
    "    'graphql': {'data': {'repository': {'pullRequest': {\n"
    "        'headRefOid': sha, 'baseRefName': 'main', 'state': 'OPEN',\n"
    "        'baseRef': {'branchProtectionRule': {'requiredStatusChecks': [\n"
    "            {'context': 'required', 'app': {'databaseId': 1}}]}}}}}},\n"
    "    'pulls': {'head': {'sha': sha}, 'base': {'ref': 'main'}, 'state': 'open'},\n"
    "}\n"
    "if 'rules/branches' in endpoint:\n"
    "    value = [[]]\n"
    "elif 'check-runs' in endpoint:\n"
    "    # --paginate --slurp returns one entry per page; a truncated set = fewer runs\n"
    "    # than total_count, the shape behind the gate's Incomplete-pagination ValueError.\n"
    "    if os.environ.get('GH_SHORT_PAGES') == '1':\n"
    "        value = [{'total_count': 3, 'check_runs': [run, {**run, 'id': 43, 'name': 'other'}]}]\n"
    "    else:\n"
    "        value = [{'total_count': 1, 'check_runs': [run]}] * 2\n"
    "elif 'statuses' in endpoint:\n"
    "    value = [[]]\n"
    "elif endpoint == 'graphql':\n"
    "    value = value['graphql']\n"
    "elif 'pulls' in endpoint:\n"
    "    value = value['pulls']\n"
    "print(json.dumps(value))\n")


@pytest.mark.platforms("posix")
def test_transient_gh_failure_names_phase_and_exit_code_and_retries_once(tmp_path, monkeypatch):
    """One non-auth gh failure is retried; if it persists, the infra receipt names the failed
    phase + gh's exit code (never stderr) and keeps the retry wording — not a credential hint."""
    from hermes_cli import kanban_pr_acceptance as acc
    monkeypatch.setattr(acc, "_GH_RETRY_DELAY", 0)
    _gh_shim(tmp_path, monkeypatch, _GH_SCRIPT)
    calls = tmp_path / "calls"
    monkeypatch.setenv("GH_CALLS_FILE", str(calls))
    monkeypatch.setenv("GH_FAIL_ON", "check-runs")
    monkeypatch.setenv("GH_FAIL_STICKY", "1")
    receipt = acc.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["classification"] == "infra" and receipt["ok"] is False
    assert "check-runs" in receipt["detail"] and "exited 1" in receipt["detail"]
    assert "(2 attempts)" in receipt["detail"] and "retry" in receipt["detail"].lower()
    assert "secret-leak" not in receipt["detail"]  # gh stderr never persists
    assert calls.read_text() == "4"  # graphql + rules + both attempts of the failing check-runs call


@pytest.mark.platforms("posix")
def test_retry_success_avoids_the_infra_receipt(tmp_path, monkeypatch):
    """The observed production flake (t_5f419e01): one transient gh exit, the identical call
    on the retry succeeds — completion must not be refused for a single blip."""
    from hermes_cli import kanban_pr_acceptance as acc
    monkeypatch.setattr(acc, "_GH_RETRY_DELAY", 0)
    _gh_shim(tmp_path, monkeypatch, _GH_SCRIPT)
    calls = tmp_path / "calls"
    monkeypatch.setenv("GH_CALLS_FILE", str(calls))
    monkeypatch.setenv("GH_FAIL_ON", "check-runs")  # fires on that call's first attempt only
    receipt = acc.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"] is True and receipt["classification"] == "success"
    assert calls.read_text() == "5"


@pytest.mark.platforms("posix")
def test_gh_timeout_after_retry_is_infra_naming_the_phase(tmp_path, monkeypatch):
    """A gh that never returns is an infra blip reporting the timeout and the failed phase,
    after the single retry — it never reaches the auth clause."""
    from hermes_cli import kanban_pr_acceptance as acc
    monkeypatch.setattr(acc, "_GH_RETRY_DELAY", 0)
    calls = tmp_path / "calls"
    calls.write_text("0")

    def hang(*args, **kwargs):
        calls.write_text(str(int(calls.read_text()) + 1))
        raise subprocess.TimeoutExpired(cmd="gh", timeout=30)

    monkeypatch.setattr(acc.subprocess, "run", hang)
    receipt = acc.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["classification"] == "infra"
    assert "timed out" in receipt["detail"] and "graphql" in receipt["detail"]
    assert calls.read_text() == "2"


@pytest.mark.platforms("posix")
def test_gh_auth_refusal_is_not_retried(tmp_path, monkeypatch):
    """The retry must not soften an auth refusal: HTTP 401 stderr stays `auth` naming the
    phase, and gh is invoked exactly once — a refusal is identity, not a blip."""
    from hermes_cli import kanban_pr_acceptance as acc
    monkeypatch.setattr(acc, "_GH_RETRY_DELAY", 0)
    _gh_shim(tmp_path, monkeypatch,
             "import os, sys\n"
             "path = os.environ['GH_CALLS_FILE']\n"
             "calls = (int(open(path).read()) if os.path.exists(path) else 0) + 1\n"
             "open(path, 'w').write(str(calls))\n"
             "sys.stderr.write('gh: HTTP 401: Bad credentials (SQLState HY000)\\n'); sys.exit(1)\n")
    calls = tmp_path / "calls"
    monkeypatch.setenv("GH_CALLS_FILE", str(calls))
    receipt = acc.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["classification"] == "auth"
    assert "HTTP 401" in receipt["detail"] and "graphql" in receipt["detail"]
    assert calls.read_text() == "1"


@pytest.mark.platforms("posix")
def test_incomplete_pagination_is_internal_not_an_auth_suggestion(tmp_path, monkeypatch):
    """The gate's own `ValueError("Incomplete check-run pagination")` — a parse bug the old
    wording dressed up as an auth problem — now reports as a gate bug naming the type."""
    from hermes_cli import kanban_pr_acceptance as acc
    _gh_shim(tmp_path, monkeypatch, _GH_SCRIPT)
    monkeypatch.setenv("GH_CALLS_FILE", str(tmp_path / "calls"))
    monkeypatch.setenv("GH_FAIL_ON", "")
    monkeypatch.setenv("GH_SHORT_PAGES", "1")  # a truncated page set: total_count says 2
    receipt = acc.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["classification"] == "infra"
    assert "evidence collection failed internally" in receipt["detail"]
    assert "ValueError" in receipt["detail"]
    assert "authentication" not in receipt["detail"]
    assert "credentials" not in receipt["detail"]
