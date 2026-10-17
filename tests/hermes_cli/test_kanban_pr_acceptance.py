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
from hermes_cli import kanban_pr_acceptance as acceptance
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_pr_acceptance_policy import normalize_required_checks


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            for marker, (code, message) in state.get("errors", {}).items():
                if marker in self.path:
                    return self._send(code, {"message": message})
            if self.path == "/graphql":
                protection = state.get("protection", [{"context": "required", "app": {"databaseId": 1}}])
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": protection} if protection else None}}}}}
            elif "/rules/branches/" in self.path:
                value = state.get("rules", [])
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = state["runs"] if "runs" in state else [] if state.get("missing") else [run]
                pages = [{"total_count": 100 + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)]}, {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
                # gh --paginate without --slurp: object pages back to back, not one JSON document.
                return self._send(200, "\n".join(json.dumps(page) for page in pages))
            elif "/statuses" in self.path:
                value = []  # array pages: gh merges them into one array
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": state.get("base", "main")}, "state": "open"}
            else:
                return self._send(404, {"message": "Not Found"})
            self._send(200, value)

        def _send(self, code, value):
            self.send_response(code)
            self.end_headers()
            self.wfile.write((value if isinstance(value, str) else json.dumps(value)).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Real gh reports an HTTP error as one stderr line `gh: <message> (HTTP <code>)`, exit 1.
    gh.write_text(f"#!{sys.executable}\nimport json,sys,urllib.error,urllib.request\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "try:\n"
                  "    print(urllib.request.urlopen(u).read().decode())\n"
                  "except urllib.error.HTTPError as e:\n"
                  "    sys.exit('gh: %s (HTTP %d)' % (json.loads(e.read())['message'], e.code))\n")
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


# --- gh without --slurp: paginated output is a sequence of page documents ---

def test_paginated_reads_never_pass_slurp_and_decode_consecutive_pages(monkeypatch):
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        stdout = ('{"total_count": 2, "check_runs": [{"id": 1}]}\n'
                  '{"total_count": 2, "check_runs": [{"id": 2}]}\n')
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    pages = acceptance._api("repos/acme/repo/commits/x/check-runs", paginate=True)
    assert "--paginate" in seen[0] and "--slurp" not in seen[0]
    assert [page["check_runs"][0]["id"] for page in pages] == [1, 2]


@pytest.mark.parametrize("stdout, pages", [
    ('{"a": 1}{"a": 2}', [{"a": 1}, {"a": 2}]),          # object pages back to back
    (' \n{"a": 1}\n\n{"a": 2}\n', [{"a": 1}, {"a": 2}]),
    ('[1, 2]\n', [[1, 2]]),                                # array pages merged by gh
    ('[1][2]', [[1], [2]]),                                # older gh: array pages back to back
    ('[]', [[]]),
])
def test_pages_decodes_every_document(stdout, pages):
    assert acceptance._pages(stdout) == pages


@pytest.mark.parametrize("stdout", ["", " \n", "null", '{"a": 1} 7', '{"a": 1} trailing', '{"a": 1}{'])
def test_pages_rejects_empty_scalar_or_truncated_output(stdout):
    # Incomplete evidence is never zero evidence.
    with pytest.raises(ValueError):
        acceptance._pages(stdout)


# --- Plan-gated private repos: operator-declared kanban.pr_required_checks ---

PR = "https://github.com/acme/repo/pull/7"
PLAN_GATE = (403, "Upgrade to GitHub Pro or make this repository public to enable this feature.")
DECLARED = "kanban:\n  pr_required_checks:\n    acme/repo: [test]\n"


def _run(run_id, name, conclusion="success", *, app_id=15368, status="completed", head_sha="a" * 40):
    return {"id": run_id, "name": name, "head_sha": head_sha, "app": {"id": app_id}, "status": status,
            "conclusion": conclusion, "html_url": f"https://github.com/acme/repo/actions/runs/{run_id}"}


# Path/condition-gated jobs legitimately skip; only the declared `test` is required.
GATED_RUNS = [_run(1, "test"), _run(2, "deploy", "skipped"), _run(3, "notify", "skipped")]


def _declare(text):
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def _plan_gated(github, runs=GATED_RUNS, **state):
    """No classic protection and the rulesets read hits the exact plan boundary."""
    github.update({"protection": [], "runs": runs, "errors": {"/rules/branches/": PLAN_GATE}, **state})
    return acceptance.collect_acceptance("acme/repo", PR)


def _read_check_runs(github):
    return any("/check-runs" in path for path in github["requests"])


@pytest.mark.platforms("linux")
def test_plan_gated_declared_test_passes_while_unrelated_jobs_skip(github):
    _declare(DECLARED)
    receipt = _plan_gated(github)
    assert receipt["ok"] and receipt["classification"] == "success"
    assert receipt["policy_source"] == "declared"
    assert receipt["required"] == [{"context": "test", "app_id": None}]
    assert [(c["name"], c["id"], c["classification"]) for c in receipt["checks"]] == [("test", 1, "success")]
    assert any("/rules/branches/main" in path for path in github["requests"])


@pytest.mark.platforms("linux")
def test_plan_gated_repo_completes_only_through_its_declaration(github):
    github.update(protection=[], runs=GATED_RUNS, errors={"/rules/branches/": PLAN_GATE})
    with connect() as conn:
        tid = kb.create_task(conn, title="gated", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR})
        task = kb.get_task(conn, tid)
        assert task.status != "done"
        assert "PR acceptance policy" in task.last_failure_error
        assert "kanban.pr_required_checks" in task.last_failure_error
        _declare(DECLARED)
        assert kb.complete_task(conn, tid, result="done", metadata={"published_pr": PR})
        assert kb.get_task(conn, tid).status == "done"
        receipts = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id", (tid,))]
    assert [r["classification"] for r in receipts] == ["policy", "success"]
    assert receipts[-1]["policy_source"] == "declared"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("declaration, runs, expected", [
    ("[test]", [_run(2, "deploy", "skipped"), _run(3, "notify", "skipped")], "missing"),
    ("[test]", [_run(1, "test", "failure"), _run(2, "deploy", "skipped")], "failure"),
    ("[test]", [_run(1, "test", None, status="in_progress")], "pending"),
    ("[test]", [_run(1, "test", head_sha="c" * 40)], "stale"),
    ("[{context: test, app_id: 15368}]", [_run(1, "test", app_id=99)], "missing"),  # same name, wrong app
], ids=["missing", "failure", "pending", "stale", "wrong-app"])
def test_plan_gated_declared_test_must_itself_pass(github, declaration, runs, expected):
    _declare(f"kanban:\n  pr_required_checks:\n    acme/repo: {declaration}\n")
    receipt = _plan_gated(github, runs)
    assert not receipt["ok"]
    assert receipt["classification"] == expected
    assert receipt["policy_source"] == "declared"
    assert [c["name"] for c in receipt["checks"]] == ["test"]


@pytest.mark.platforms("linux")
def test_plan_gated_pinned_declaration_accepts_its_own_app(github):
    _declare("kanban:\n  pr_required_checks:\n    acme/repo: [{context: test, app_id: 15368}]\n")
    receipt = _plan_gated(github, [_run(1, "test", "failure", app_id=99), _run(2, "test", app_id=15368)])
    # The other app's same-named run is not the required check; the pinned one passes.
    assert receipt["required"] == [{"context": "test", "app_id": 15368}]
    assert receipt["ok"] and [c["id"] for c in receipt["checks"]] == [2]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("config", [
    "kanban:\n  pr_required_checks: [test]\n",
    "kanban:\n  pr_required_checks:\n    acme/repo: []\n",
    "kanban:\n  pr_required_checks:\n    acme/repo: test\n",
    "kanban:\n  pr_required_checks:\n    acme: [test]\n",
    "kanban:\n  pr_required_checks:\n    acme/repo: ['  ']\n",
    "kanban:\n  pr_required_checks:\n    acme/repo: [{context: test, app_id: true}]\n",
    "kanban:\n  pr_required_checks:\n    acme/repo: [{context: test, app_id: 0}]\n",
    "kanban:\n  pr_required_checks:\n    acme/repo: [{context: test, app: 15368}]\n",
    "kanban:\n  pr_required_checks:\n    acme/repo: [test, {app_id: 15368}]\n",
    "kanban: [pr_required_checks]\n",
], ids=["list", "empty", "scalar", "bad-repo", "blank", "bool-app", "zero-app", "extra-key",
        "no-context", "kanban-list"])
def test_malformed_declaration_fails_policy_instead_of_being_skipped(github, config):
    _declare(config)
    receipt = _plan_gated(github)
    assert not receipt["ok"] and receipt["classification"] == "policy"
    assert "kanban.pr_required_checks" in receipt["detail"] and "malformed" in receipt["detail"]
    assert not _read_check_runs(github)


@pytest.mark.platforms("linux")
def test_unparseable_config_fails_policy_without_echoing_config_lines(github):
    _declare("kanban:\n  pr_required_checks: {acme/repo: [test]\n  token_line: hunter2\n")
    receipt = _plan_gated(github)
    assert receipt["classification"] == "policy"
    assert "cannot be read or parsed" in receipt["detail"]
    assert "hunter2" not in json.dumps(receipt)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("config", [None, "kanban:\n  failure_limit: 2\n", "kanban:\n  pr_required_checks:\n"],
                         ids=["no-config", "no-key", "null-key"])
def test_absent_declaration_is_policy_never_nothing_required(github, config):
    if config is not None:
        _declare(config)
    receipt = _plan_gated(github)
    assert not receipt["ok"] and receipt["classification"] == "policy"
    assert "declares nothing for acme/repo" in receipt["detail"]
    assert receipt["checks"] == [] and not _read_check_runs(github)


@pytest.mark.platforms("linux")
def test_declarations_are_scoped_to_the_repository_they_name(github):
    _declare("kanban:\n  pr_required_checks:\n    acme/repo-tools: [test]\n    other/repo: [test]\n")
    receipt = _plan_gated(github)
    assert receipt["classification"] == "policy"
    assert "declares nothing for acme/repo" in receipt["detail"]
    # An exact PR contract resolves the same repository's declaration, and only that one.
    _declare("kanban:\n  pr_required_checks:\n    other/repo: [nope]\n    acme/repo: [test]\n")
    receipt = acceptance.collect_acceptance(PR, PR)
    assert receipt["ok"] and receipt["required"] == [{"context": "test", "app_id": None}]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("error, expected", [
    ((403, "Resource not accessible by integration"), "auth"),
    ((403, "Upgrade to GitHub Pro to enable this feature."), "auth"),  # near-miss text
    ((404, "Not Found"), "auth"),
    ((500, "Internal Server Error"), "infra"),
    ((502, "Server Error"), "infra"),
], ids=["403", "403-near-miss", "404", "500", "502"])
def test_other_rules_refusals_are_not_plan_gated(github, error, expected):
    _declare(DECLARED)  # a valid declaration must not paper over these
    github.update(protection=[], runs=GATED_RUNS, errors={"/rules/branches/": error})
    receipt = acceptance.collect_acceptance("acme/repo", PR)
    assert not receipt["ok"] and receipt["classification"] == expected
    assert "policy_source" not in receipt
    assert not _read_check_runs(github)


@pytest.mark.platforms("linux")
def test_plan_boundary_on_another_endpoint_stays_auth(github):
    _declare(DECLARED)
    receipt = _plan_gated(github, errors={"/rules/branches/": PLAN_GATE, "/check-runs": PLAN_GATE})
    assert not receipt["ok"] and receipt["classification"] == "auth"


@pytest.mark.platforms("linux")
def test_github_derived_required_set_beats_declaration(github):
    _declare(DECLARED)
    failing_test = _run(9, "test", "failure")
    # Classic protection still names a check while the rulesets read is plan-gated.
    receipt = _plan_gated(github, [_run(1, "required", app_id=1), failing_test],
                          protection=[{"context": "required", "app": {"databaseId": 1}}])
    assert receipt["ok"] and receipt["policy_source"] == "github"
    assert receipt["required"] == [{"context": "required", "app_id": 1}]
    # Readable rulesets win too.
    github.update(protection=[], errors={}, runs=[_run(1, "lint"), failing_test], rules=[
        {"type": "required_status_checks",
         "parameters": {"required_status_checks": [{"context": "lint", "integration_id": 15368}]}}])
    receipt = acceptance.collect_acceptance("acme/repo", PR)
    assert receipt["ok"] and receipt["policy_source"] == "github"
    assert receipt["required"] == [{"context": "lint", "app_id": 15368}]
    # Readable-but-empty rules are GitHub's answer, not the plan boundary: no declared fallback.
    github.update(rules=[], runs=GATED_RUNS)
    receipt = acceptance.collect_acceptance("acme/repo", PR)
    assert not receipt["ok"] and receipt["classification"] == "missing"
    assert receipt["policy_source"] == "github" and receipt["required"] == []
    assert "No repository-required checks" in receipt["detail"]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("race", [{"head_change": True}, {"base": "release"}], ids=["head", "base"])
def test_declared_success_is_void_when_head_or_base_moves(github, race):
    _declare(DECLARED)
    receipt = _plan_gated(github, **race)
    assert not receipt["ok"] and receipt["classification"] == "stale"
    assert receipt["head_sha"] == "a" * 40
    assert "changed while collecting" in receipt["detail"]
    assert github["requests"][-1].startswith("/repos/acme/repo/pulls/7")


def test_normalize_required_checks():
    assert normalize_required_checks(None) == {}
    assert normalize_required_checks({}) == {}
    assert normalize_required_checks({"acme/repo": ["test", {"context": " build ", "app_id": 15368}, "test"]}) == {
        "acme/repo": frozenset({("test", None), ("build", 15368)})}
    for bad in ([], "acme/repo", {"acme/repo": [None]}, {"acme/repo": [1]}, {1: ["test"]}):
        with pytest.raises(ValueError):
            normalize_required_checks(bad)
