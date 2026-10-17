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
             "no_protection": False, "rules_status": 200, "single_run": False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                protection = None if state.get("no_protection") else {"requiredStatusChecks": [
                    {"context": "required", "app": {"databaseId": 1}}]}
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": protection}}}}}
            elif "/rules/branches/" in self.path:
                if state.get("rules_status", 200) != 200:
                    self.send_error(state["rules_status"])
                    return
                value = state.get("rules", [[]])
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                if state.get("single_run"):
                    value = [{"total_count": len(runs), "check_runs": runs}]
                else:
                    value = [{"total_count": 100 + len(runs), "check_runs": [
                        {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                        for i in range(100)]}, {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = [state.get("statuses", [])]
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
    gh.write_text(f"#!{sys.executable}\nimport sys,os,urllib.request,urllib.error\n"
                  "path=sys.argv[2]\n"
                  "if 'rules/branches' in path and os.environ.get('PR_ACCEPTANCE_FORCE_RULES_INFRA'):\n"
                  "    sys.stderr.write('gh: Could not resolve host: github.com\\n')\n"
                  "    sys.exit(1)\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+path\n"
                  "try:\n"
                  "    print(urllib.request.urlopen(u).read().decode())\n"
                  "except urllib.error.HTTPError as e:\n"
                  "    sys.stderr.write('gh: request failed (HTTP %d)\\n' % e.code)\n"
                  "    sys.exit(1)\n")
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


@pytest.mark.platforms("linux")
def test_rules_403_falls_back_to_exact_head_check_runs(github):
    """Private repo, no readable rulesets (403) and no classic branch protection:
    acceptance is decided by every check-run at the exact head, not by
    'no repository-required checks configured'."""
    github["no_protection"] = True
    github["rules_status"] = 403
    github["single_run"] = True
    with connect() as conn:
        # 1. exact head has a successful check-run -> completes.
        github.update(conclusion="success", head="a" * 40)
        tid = kb.create_task(conn, title="green", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
            (tid,)).fetchone()[0])
        assert receipt["ok"] is True
        assert receipt["acceptance_rule"] == "exact_head_all_check_runs"
        assert receipt["rules_readable"] is False

        # 2. exact head has no check-run at all -> cannot complete.
        github.update(conclusion="success", head="a" * 40)
        github["missing"] = True
        tid = kb.create_task(conn, title="no-runs", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        github.pop("missing")

        # 3. exact head has a failing check-run -> cannot complete.
        github.update(conclusion="failure", head="a" * 40)
        tid = kb.create_task(conn, title="red", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"

        # 4. the check-run belongs to a different SHA than the exact head -> cannot complete.
        github.update(conclusion="success", head="a" * 40)
        github["stale"] = True
        tid = kb.create_task(conn, title="stale-run", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        github.pop("stale")


@pytest.mark.platforms("linux")
def test_fallback_accepts_a_green_commit_status_when_no_check_runs_exist(github):
    """A PR whose CI ran LOCALLY (the reviewer posts a `ci` commit status on the
    exact head; no Actions run at all) completes on that status — and only when
    the LATEST status of each context is green."""
    github["no_protection"] = True
    github["rules_status"] = 403
    github["single_run"] = True
    github["missing"] = True  # no check-runs at all
    PR = "https://github.com/acme/repo/pull/7"
    try:
        with connect() as conn:
            github["statuses"] = [{"id": 2, "context": "ci", "state": "success"}]
            tid = kb.create_task(conn, title="local-ci-green", completion_contract="acme/repo")
            assert kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR})

            # The API lists history; the higher id is the newer status.
            github["statuses"] = [{"id": 3, "context": "ci", "state": "failure"},
                                  {"id": 2, "context": "ci", "state": "success"}]
            tid = kb.create_task(conn, title="local-ci-red-after-green", completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR})

            github["statuses"] = [{"id": 4, "context": "ci", "state": "pending"}]
            tid = kb.create_task(conn, title="local-ci-pending", completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR})

            github["statuses"] = []
            tid = kb.create_task(conn, title="no-evidence", completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR})
    finally:
        github.pop("missing", None)
        github.pop("statuses", None)


@pytest.mark.platforms("linux")
def test_rules_200_empty_falls_back_to_exact_head_check_runs(github):
    """Private free-plan repo whose rulesets endpoint answers 200 [] (not 403)
    and has no classic branch protection: same strict fallback as 403 — every
    check-run at the exact head must be green, and at least one must exist.
    Measured on emana-ai/emana-app 2026-10-01: the old 'readable but empty ->
    refuse' branch refused an approved, green PR (#165)."""
    github["no_protection"] = True
    github["rules_status"] = 200
    github["rules"] = [[]]
    github["single_run"] = True
    with connect() as conn:
        github.update(conclusion="success", head="a" * 40)
        tid = kb.create_task(conn, title="empty-rules-green", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
            (tid,)).fetchone()[0])
        assert receipt["ok"] is True
        assert receipt["acceptance_rule"] == "exact_head_all_check_runs"
        assert receipt["rules_readable"] is True

        github.update(conclusion="failure", head="a" * 40)
        tid = kb.create_task(conn, title="empty-rules-red", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"

        github.update(conclusion="success", head="a" * 40)
        github["missing"] = True
        tid = kb.create_task(conn, title="empty-rules-no-runs", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        github.pop("missing")


@pytest.mark.platforms("linux")
def test_rules_403_fallback_skipped_check_runs_do_not_block(github):
    """A job disabled by `if:` (conclusion=skipped/neutral) is not evidence
    against the head under the check-runs fallback: it must not classify
    infra and must not block completion, as long as at least one other
    check-run at the exact head succeeded. A head where *every* check-run
    was skipped/neutral still has no positive evidence and must not pass
    (negative control)."""
    github["no_protection"] = True
    github["rules_status"] = 403
    with connect() as conn:
        # 1. 100 skipped "optional" runs (from the fixture's pagination
        #    padding) plus one successful "required" run -> completes, and
        #    the skipped runs are NOT classified infra.
        github.update(conclusion="success", head="a" * 40)
        tid = kb.create_task(conn, title="skip-plus-success", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
            (tid,)).fetchone()[0])
        assert receipt["ok"] is True
        assert receipt["classification"] == "success"
        assert receipt["acceptance_rule"] == "exact_head_all_check_runs"
        skipped_checks = [c for c in receipt["checks"] if c["conclusion"] == "skipped"]
        assert skipped_checks and all(c["classification"] == "skipped" for c in skipped_checks)

        # 2. Negative control: every check-run at the exact head is skipped
        #    -> no positive evidence, cannot complete.
        github.update(conclusion="skipped", head="a" * 40)
        tid = kb.create_task(conn, title="all-skipped", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
            (tid,)).fetchone()[0])
        assert receipt["ok"] is False
        assert receipt["classification"] != "infra"
        assert all(c["classification"] == "skipped" for c in receipt["checks"])

        # 3. A genuinely failing run alongside skipped ones still blocks.
        github.update(conclusion="failure", head="a" * 40)
        tid = kb.create_task(conn, title="skip-plus-failure", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
            (tid,)).fetchone()[0])
        assert receipt["ok"] is False
        assert receipt["classification"] == "failure"


@pytest.mark.platforms("linux")
def test_rules_403_is_distinguished_from_genuine_infra_failure(github, monkeypatch):
    """A 403/404 reading the rulesets endpoint means 'no rulesets readable' and
    is not classification=infra when the repo has branch protection to fall back
    on. A real auth/network failure (any other non-2xx from gh) must still
    classify infra."""
    with connect() as conn:
        # Branch protection exists; 403 on rules/branches is a benign "not readable".
        github["rules_status"] = 403
        github.update(conclusion="success", head="a" * 40)
        tid = kb.create_task(conn, title="rules-403-with-protection", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
            (tid,)).fetchone()[0])
        assert receipt["classification"] == "success"
        assert receipt["rules_readable"] is False

        # A genuine gh failure on the same endpoint (non-HTTP, e.g. network/auth)
        # must still classify infra and must not silently fall back.
        github["rules_status"] = 200
        monkeypatch.setenv("PR_ACCEPTANCE_FORCE_RULES_INFRA", "1")
        tid = kb.create_task(conn, title="rules-network-failure", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
            (tid,)).fetchone()[0])
        assert receipt["classification"] == "infra"
        monkeypatch.delenv("PR_ACCEPTANCE_FORCE_RULES_INFRA", raising=False)

