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
             "workflow_runs": [{"id": 90, "name": "CI", "head_sha": "a" * 40,
                 "status": "completed", "conclusion": "success", "check_suite_id": 1,
                 "html_url": "https://github.com/acme/repo/actions/runs/90"}]}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if state.get("error_path", "/rules/branches/") in self.path and state.get("api_error"):
                status, message = state["api_error"]
                self.send_response(status)
                self.end_headers()
                self.wfile.write(json.dumps({"message": message}).encode())
                return
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = [state.get("rules", [])]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1, "slug": "github-actions"}, "check_suite": {"id": state.get("suite_id", 1)},
                       "status": state.get("check_status", "in_progress" if state["conclusion"] == "pending" else "completed"), "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                optional_count = state.get("optional_count", 100)
                value = [{"total_count": optional_count + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional",
                     "app": {**run["app"], "slug": state.get("app_slug", "github-actions")},
                     "check_suite": {"id": state.get("optional_suite_id", run["check_suite"]["id"])},
                     "conclusion": state.get("optional_conclusion", "skipped")}
                    for i in range(optional_count)]},
                    {"total_count": optional_count + len(runs), "check_runs": runs}]
                if state.get("incomplete"):
                    value[0]["total_count"] += 1
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/actions/runs" in self.path:
                workflow_runs = state["workflow_runs"]
                if state.get("malformed_workflow_page"):
                    workflow_runs = "invalid"
                value = [{"total_count": len(state["workflow_runs"]) + state.get("workflow_total_delta", 0),
                          "workflow_runs": workflow_runs}]
            elif "/statuses" in self.path:
                value = [state.get("statuses", [])]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha},
                         "base": {"ref": "other" if state.get("base_change") else "main"},
                         "state": "open"}
            else:
                self.send_error(404)
                return
            if self.path == "/graphql" and state.get("no_required"):
                value["data"]["repository"]["pullRequest"]["baseRef"] = None
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
                  "try:\n    print(urllib.request.urlopen(u).read().decode())\n"
                  "except urllib.error.HTTPError as e:\n"
                  "    body=json.loads(e.read().decode())\n"
                  "    print(json.dumps([body] if '--slurp' in sys.argv else body))\n"
                  "    print(f'gh: request failed (HTTP {e.code})', file=sys.stderr)\n"
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


_PLAN_MESSAGE = "Upgrade to GitHub Pro or make this repository public to enable this feature."


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("fault", "classification"), [
    (None, "success"), ("failure", "failure"), ("pending", "pending"),
    ("stale", "stale"), ("head_change", "stale"), ("missing", "missing"),
    ("no_required", "success"),
])
def test_plan_unavailable_rules_preserve_required_exact_head_gate(github, fault, classification):
    github["api_error"] = (403, _PLAN_MESSAGE)
    if fault in {"failure", "pending"}:
        github["conclusion"] = fault
    elif fault:
        github[fault] = True
    with connect() as conn:
        tid = kb.create_task(conn, title="plan-limited", completion_contract="acme/repo")
        ok = kb.complete_task(conn, tid, summary="done", metadata={
            "published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
        task = kb.get_task(conn, tid)
        assert task is not None
        assert (task.status == "done") is ok
    assert ok is (classification == "success")
    assert receipt["classification"] == classification
    assert receipt["branch_rules"] == {"status": "plan_unavailable"}
    assert receipt["required"] == ([] if fault == "no_required" else [{"context": "required", "app_id": 1}])
    assert any("/check-runs" in p for p in github["requests"])
    assert any("/pulls/7" in p for p in github["requests"])


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("status", "message", "classification"), [
    (403, "Resource not accessible by integration", "auth"),
    (401, _PLAN_MESSAGE, "auth"), (404, _PLAN_MESSAGE, "auth"),
    (500, _PLAN_MESSAGE, "infra"), (503, _PLAN_MESSAGE, "infra"),
])
def test_rules_errors_other_than_plan_403_keep_their_classification(github, status, message, classification):
    from hermes_cli.kanban_pr_acceptance import collect_acceptance

    github["api_error"] = (status, message)
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert not receipt["ok"]
    assert receipt["classification"] == classification
    assert "branch_rules" not in receipt
    assert not any("/check-runs" in p for p in github["requests"])


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("endpoint", ["/graphql", "/check-runs", "/statuses", "/pulls/"])
def test_plan_403_from_other_endpoints_still_fails_auth(github, endpoint):
    from hermes_cli.kanban_pr_acceptance import collect_acceptance

    github.update(api_error=(403, _PLAN_MESSAGE), error_path=endpoint)
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert not receipt["ok"]
    assert receipt["classification"] == "auth"
    assert "branch_rules" not in receipt


@pytest.mark.parametrize("body", ["not JSON", "[]", "null", '{"message": null}',
                                    '{"message": 403}', '{"detail": "Upgrade to GitHub Pro"}'])
def test_malformed_rules_error_cannot_enable_plan_exception(monkeypatch, body):
    import subprocess

    from hermes_cli.kanban_pr_acceptance import _api, _GateAuthError

    def denied(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output=body,
                                            stderr=f"gh: {_PLAN_MESSAGE} (HTTP 403)")

    monkeypatch.setattr(subprocess, "run", denied)
    with pytest.raises(_GateAuthError):
        _api("repos/acme/repo/rules/branches/main?per_page=100", paginate=True)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("plan_limited", [False, True])
@pytest.mark.parametrize(("fault", "classification"), [
    (None, "success"),    ("failure", "failure"), ("pending", "pending"),
    ("cancelled", "infra"), ("neutral", "infra"), ("skipped", "success"),
    ("timed_out", "infra"), ("action_required", "infra"),
    ("unknown", "infra"), ("no_conclusion", "infra"),
    ("missing", "missing"), ("stale", "stale"),
    ("head_change", "stale"), ("base_change", "stale"),
    ("incomplete", "infra"),
])
def test_no_required_checks_accept_only_successful_exact_head_ci(github, plan_limited, fault, classification):
    github.update(no_required=True, optional_count=0)
    if plan_limited:
        github["api_error"] = (403, _PLAN_MESSAGE)
    if fault in {"missing", "stale", "head_change", "base_change", "incomplete"}:
        github[fault] = True
    elif fault:
        github["conclusion"] = None if fault == "no_conclusion" else fault
    with connect() as conn:
        tid = kb.create_task(conn, title="no-required", completion_contract="acme/repo")
        ok = kb.complete_task(conn, tid, summary="done", metadata={
            "published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
        task = kb.get_task(conn, tid)
        assert task is not None
        assert (task.status == "done") is ok
    assert ok is (classification == "success")
    assert receipt["classification"] == classification
    assert receipt["head_sha"] == "a" * 40
    assert receipt["required"] == []
    assert any("/statuses" in p for p in github["requests"]) is (fault != "incomplete")
    if plan_limited:
        assert receipt["branch_rules"] == {"status": "plan_unavailable"}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("required_source", [None, "graphql", "rules"])
@pytest.mark.parametrize("optional_conclusion", ["success", "failure", "pending", "cancelled", "neutral", "skipped"])
def test_optional_checks_matter_only_without_required_checks(github, required_source, optional_conclusion):
    from hermes_cli.kanban_pr_acceptance import collect_acceptance

    github.update(no_required=required_source != "graphql", optional_conclusion=optional_conclusion)
    if required_source == "rules":
        github["rules"] = [{"type": "required_status_checks", "parameters": {
            "required_status_checks": [{"context": "required", "integration_id": 1}]}}]
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"] is (required_source is not None or optional_conclusion in {"success", "skipped"})
    assert len(receipt["checks"]) == (1 if required_source else 101)
    assert {c["head_sha"] for c in receipt["checks"]} == {"a" * 40}
    assert any("/actions/runs" in p for p in github["requests"]) is (required_source is None)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("runs_present", [False, True])
@pytest.mark.parametrize("status_state", ["success", "failure", "pending", "error", "unknown"])
def test_no_required_checks_include_legacy_statuses(github, runs_present, status_state):
    from hermes_cli.kanban_pr_acceptance import collect_acceptance

    github.update(no_required=True, optional_count=0, missing=not runs_present,
                  statuses=[{"id": 43, "context": "legacy", "state": status_state}])
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"] is (status_state == "success")
    assert {c["id"] for c in receipt["checks"]} == ({42, 43} if runs_present else {43})


@pytest.mark.platforms("posix")
def test_no_required_checks_do_not_hide_unsuccessful_legacy_status(github):
    from hermes_cli.kanban_pr_acceptance import collect_acceptance

    github.update(no_required=True, optional_count=0, statuses=[
        {"id": 44, "context": "legacy", "state": "success"},
        {"id": 43, "context": "legacy", "state": "pending"}])
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert not receipt["ok"]
    assert receipt["classification"] == "pending"
    assert {c["id"] for c in receipt["checks"]} == {42, 43, 44}


@pytest.mark.platforms("posix")
def test_no_required_checks_accept_mapped_skips_only_with_successful_exact_head_workflow(github):
    from hermes_cli.kanban_pr_acceptance import collect_acceptance

    github.update(no_required=True, optional_count=3, optional_conclusion="skipped",
                  suite_id=77, workflow_runs=[{"id": 99, "name": "Reader CI", "head_sha": "a" * 40,
                      "status": "completed", "conclusion": "success", "check_suite_id": 77,
                      "html_url": "https://github.com/acme/repo/actions/runs/99"}])
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert receipt["ok"]
    assert receipt["classification"] == "success"
    assert len(receipt["workflows"]) == 1
    assert {check["classification"] for check in receipt["checks"] if check["conclusion"] == "skipped"} == {"skipped"}
    assert all(check["workflow_evidence"]["run_id"] == 99 for check in receipt["checks"]
               if check["conclusion"] == "skipped")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("fault", "classification"), [
    ("stale", "stale"), ("in_progress", "pending"),
])
def test_mapped_skips_still_require_current_completed_check(github, fault, classification):
    from hermes_cli.kanban_pr_acceptance import collect_acceptance

    github.update(no_required=True, optional_count=1, optional_conclusion="skipped",
                  conclusion="skipped", suite_id=77,
                  workflow_runs=[{"id": 99, "name": "Reader CI", "head_sha": "a" * 40,
                      "status": "completed", "conclusion": "success", "check_suite_id": 77,
                      "html_url": "https://github.com/acme/repo/actions/runs/99"}])
    if fault == "stale":
        github["stale"] = True
    else:
        github["check_status"] = "in_progress"
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert not receipt["ok"]
    assert receipt["classification"] == classification
    assert all(check["conclusion"] == "skipped" for check in receipt["checks"])
    assert all(check["head_sha"] == ("b" * 40 if fault == "stale" else "a" * 40)
               for check in receipt["checks"])
    assert all("workflow_evidence" not in check for check in receipt["checks"])


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("bad_id", [0, -1, None, True, "99"])
def test_mapped_skips_reject_invalid_workflow_run_id(github, bad_id):
    from hermes_cli.kanban_pr_acceptance import collect_acceptance

    github.update(no_required=True, optional_count=1, optional_conclusion="skipped",
                  conclusion="skipped", suite_id=77,
                  workflow_runs=[{"id": bad_id, "name": "Reader CI", "head_sha": "a" * 40,
                      "status": "completed", "conclusion": "success", "check_suite_id": 77,
                      "html_url": "https://github.com/acme/repo/actions/runs/99"}])
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert not receipt["ok"]
    assert receipt["classification"] == "infra"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("fault", "classification"), [
    ("failure", "infra"), ("pending", "infra"), ("cancelled", "infra"),
    ("skipped", "infra"), ("neutral", "infra"), ("unknown", "infra"),
    ("wrong_head", "infra"), ("zero", "infra"), ("bad_count", "infra"),
    ("malformed", "infra"), ("duplicate", "infra"), ("unknown_suite", "infra"),
    ("external_skip", "infra"), ("external_failure", "failure"), ("external_pending", "pending"),
    ("multiple_failure", "infra"), ("api_403", "auth"), ("api_500", "infra"),
])
def test_no_required_checks_fail_closed_on_unproven_workflow_skip(github, fault, classification):
    from hermes_cli.kanban_pr_acceptance import collect_acceptance

    github.update(no_required=True, optional_count=1, optional_conclusion="skipped")
    workflow = github["workflow_runs"][0]
    if fault in {"failure", "pending", "cancelled", "skipped", "neutral", "unknown"}:
        workflow["conclusion"] = fault
        workflow["status"] = "in_progress" if fault == "pending" else "completed"
    elif fault == "wrong_head":
        workflow["head_sha"] = "b" * 40
    elif fault == "zero":
        github["workflow_runs"] = []
    elif fault == "bad_count":
        github["workflow_total_delta"] = 1
    elif fault == "duplicate":
        github["workflow_runs"] = [workflow, dict(workflow)]
    elif fault == "unknown_suite":
        github["suite_id"] = 55
    elif fault in {"external_skip", "external_failure", "external_pending"}:
        github["app_slug"] = "third-party"
        github["optional_conclusion"] = {"external_skip": "skipped", "external_failure": "failure",
                                         "external_pending": "pending"}[fault]
    elif fault == "malformed":
        github["malformed_workflow_page"] = True
    elif fault == "multiple_failure":
        github["workflow_runs"].append({**workflow, "id": 91, "check_suite_id": 2, "conclusion": "failure"})
    elif fault in {"api_403", "api_500"}:
        github.update(api_error=(403 if fault == "api_403" else 500, "unavailable"),
                      error_path="/actions/runs")
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7")
    assert not receipt["ok"]
    assert receipt["classification"] == classification
