"""A required check skipped by its own workflow's path filter is accepted only with proof.

Real SQLite lifecycle + a local GitHub HTTP contract. Admission requires the exact-head
GitHub Actions skip, its successful run and planner job, the trusted base workflow's
static filter (unchanged at head) and the complete changed-path list proving no filter
that gates the required job fired. Everything else stays rejected.
"""
import base64
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

HEAD, BASE, ACTIONS = "a" * 40, "c" * 40, 15368  # GitHub Actions app id (public, global)
GUARD = "(github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)"
WORKFLOW = f"""name: CI
on: pull_request
jobs:
  plan:
    name: Plan paths
    runs-on: ubuntu-latest
    outputs:
      server: ${{{{ steps.paths.outputs.server }}}}
      schema: ${{{{ steps.paths.outputs.schema }}}}
      fixtures: ${{{{ steps.paths.outputs.fixtures }}}}
    steps:
      - uses: actions/checkout@v4
      - id: paths
        run: |
          bash .github/scripts/path-filter.sh <<'FILTERS'
          # server code and its dependency manifest
          server: ^(server/|pyproject\\.toml$)
          schema: ^server/migrations/[^/]+\\.py$
          fixtures: ^db/pg_
          FILTERS
  unit:
    name: unit-server
    needs: plan
    if: needs.plan.outputs.server == 'true' && {GUARD}
    runs-on: ubuntu-latest
  migrate:
    name: db-migrations
    needs: [plan]
    if: (needs.plan.outputs.schema == 'true' || needs.plan.outputs.fixtures == 'true') && {GUARD}
    runs-on: ubuntu-latest
  web:
    name: lint-web
    needs: plan
    runs-on: ubuntu-latest
"""
WORKFLOW_PATH = ".github/workflows/ci.yml"
RUN = 900


def _check(job_id, name, conclusion, run=RUN, repo="acme/app"):
    return {"id": job_id, "name": name, "head_sha": HEAD, "status": "completed", "conclusion": conclusion,
            "app": {"id": ACTIONS, "slug": "github-actions"},
            "details_url": f"https://github.com/{repo}/actions/runs/{run}/job/{job_id}",
            "html_url": f"https://github.com/{repo}/runs/{job_id}"}


def _scenario():
    return {
        "required": [("unit-server", ACTIONS), ("db-migrations", ACTIONS), ("lint-web", ACTIONS)],
        "checks": [_check(11, "Plan paths", "success"), _check(12, "unit-server", "skipped"),
                   _check(13, "db-migrations", "skipped"), _check(14, "lint-web", "success")],
        "jobs": [{"id": 11, "name": "Plan paths", "status": "completed", "conclusion": "success", "run_id": RUN},
                 {"id": 12, "name": "unit-server", "status": "completed", "conclusion": "skipped", "run_id": RUN},
                 {"id": 13, "name": "db-migrations", "status": "completed", "conclusion": "skipped", "run_id": RUN},
                 {"id": 14, "name": "lint-web", "status": "completed", "conclusion": "success", "run_id": RUN}],
        "run": {"id": RUN, "head_sha": HEAD, "event": "pull_request", "status": "completed", "conclusion": "success",
                "path": WORKFLOW_PATH, "repository": {"full_name": "acme/app"},
                "head_repository": {"full_name": "acme/app"}},
        "pr": {"head": {"sha": HEAD, "repo": {"full_name": "acme/app"}}, "base": {"ref": "main", "sha": BASE},
               "state": "open"},
        # 150 web paths: the changed-file list spans two pages.
        "files": [{"filename": f"web/src/page{i}.tsx", "status": "modified"} for i in range(150)],
        "contents": {(WORKFLOW_PATH, BASE): WORKFLOW, (WORKFLOW_PATH, HEAD): WORKFLOW,
                     (".github/scripts/path-filter.sh", BASE): "#!/bin/sh\n", (".github/scripts/path-filter.sh", HEAD): "#!/bin/sh\n"},
    }


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"s": _scenario(), "requests": []}

    def pages(items, key=None, total=None):
        chunks = [items[i:i + 100] for i in range(0, len(items), 100)] or [[]]
        return [{"total_count": len(items) if total is None else total, key: c} for c in chunks] if key else chunks

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            s = state["s"]
            path = self.path
            state["requests"].append(path)
            if path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": HEAD, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": c, "app": {"databaseId": a} if a else None} for c, a in s["required"]]}}}}}}
            elif "/rules/branches/" in path or "/statuses" in path:
                value = [[]]
            elif "/check-runs" in path:
                value = pages(s["checks"], "check_runs")
            elif re.search(r"/pulls/7/files\?", path):
                value = pages(s["files"])
            elif "/pulls/7" in path:
                value = {**s["pr"], "changed_files": s.get("changed_files", len(s["files"]))}
                if s.get("race"):
                    s["pr"] = {**s["pr"], "head": {**s["pr"]["head"], "sha": "b" * 40}}
                if s.get("base_race"):  # base branch advances on the same ref mid-collection
                    s["pr"] = {**s["pr"], "base": {**s["pr"]["base"], "sha": "e" * 40}}
            elif m := re.search(r"/actions/runs/(\d+)/jobs\?", path):
                value = pages([j for j in s["jobs"] if j["run_id"] == int(m[1])], "jobs", s.get("jobs_total"))
            elif m := re.search(r"/actions/runs/(\d+)$", path):
                value = s["runs"].get(int(m[1])) if "runs" in s else (s["run"] if int(m[1]) == RUN else None)
            elif m := re.search(r"/contents/(.+)\?ref=([0-9a-f]{40})$", path):
                value = s["contents"].get((m[1], m[2]))
                if value is not None:
                    value = {"type": "file", "encoding": "base64",
                             "content": base64.b64encode(value.encode()).decode()}
            else:
                value = None
            if value is None:
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


def _complete(conn):
    tid = kb.create_task(conn, title="publish", completion_contract="acme/app")
    ok = kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/app/pull/7"})
    receipt = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' "
                                      "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()[0])
    assert (kb.get_task(conn, tid).status == "done") is ok
    return ok, receipt


@pytest.mark.platforms("linux")
def test_path_filtered_required_skip_is_accepted_as_not_applicable_never_success(github):
    with connect() as conn:
        ok, receipt = _complete(conn)
    assert ok and receipt["ok"] and receipt["head_sha"] == HEAD
    skipped = {c["name"]: c for c in receipt["checks"] if c["conclusion"] == "skipped"}
    assert set(skipped) == {"unit-server", "db-migrations"}
    for check in skipped.values():
        assert check["classification"] == "not_applicable"
        proof = check["proof"]
        assert proof["run_id"] == RUN and proof["base_sha"] == BASE and proof["workflow"] == WORKFLOW_PATH
        assert proof["changed_paths"] == 150 and not any(proof["filters"].values())
    assert skipped["db-migrations"]["proof"]["filters"] == {"schema": False, "fixtures": False}
    assert next(c for c in receipt["checks"] if c["name"] == "lint-web")["classification"] == "success"


REJECTIONS = {
    # Any changed path matching a gating filter means the skip contradicts the policy.
    "backend path": lambda s: s["files"].append({"filename": "server/app.py", "status": "modified"}),
    "second OR filter": lambda s: s["files"].append({"filename": "db/pg_roles.sql", "status": "added"}),
    "renamed out of backend": lambda s: s["files"].append(
        {"filename": "web/moved.py", "status": "renamed", "previous_filename": "server/old.py"}),
    # Skips the base workflow cannot explain.
    "unknown expression": lambda s: s["contents"].update({k: v.replace(
        "if: needs.plan.outputs.server == 'true'", "if: always() && needs.plan.outputs.server == 'true'")
        for k, v in s["contents"].items() if k[0] == WORKFLOW_PATH}),
    "label condition": lambda s: s["contents"].update({k: v.replace(
        "needs.plan.outputs.server == 'true' &&", "contains(github.event.pull_request.labels.*.name, 'x') &&")
        for k, v in s["contents"].items() if k[0] == WORKFLOW_PATH}),
    "required job absent from workflow": lambda s: s["required"].append(("integration", ACTIONS)) or
        s["checks"].append(_check(15, "integration", "skipped")) or
        s["jobs"].append({"id": 15, "name": "integration", "status": "completed", "conclusion": "skipped", "run_id": RUN}),
    "undefined filter output": lambda s: s["contents"].update({k: v.replace(
        "          fixtures: ^db/pg_\n", "") for k, v in s["contents"].items() if k[0] == WORKFLOW_PATH}),
    "unsupported regex": lambda s: s["contents"].update({k: v.replace(
        "^db/pg_", "^db/pg_\\d{2}") for k, v in s["contents"].items() if k[0] == WORKFLOW_PATH}),
    "unquoted heredoc": lambda s: s["contents"].update({k: v.replace(
        "<<'FILTERS'", "<<FILTERS") for k, v in s["contents"].items() if k[0] == WORKFLOW_PATH}),
    "yaml tag": lambda s: s["contents"].update({k: "x: !!python/object/apply:os.system ['true']\n" + v
                                                for k, v in s["contents"].items() if k[0] == WORKFLOW_PATH}),
    # Workflow/planner evidence that is not a clean, exact-head success.
    "run failed": lambda s: s["run"].update(conclusion="failure"),
    "run cancelled": lambda s: s["run"].update(conclusion="cancelled"),
    "planner failed": lambda s: s["jobs"][0].update(conclusion="failure"),
    "planner missing": lambda s: s["jobs"].pop(0),
    "run on another head": lambda s: s["run"].update(head_sha="d" * 40),
    "run in another repo": lambda s: s["run"].update(repository={"full_name": "other/app"}),
    "run from push": lambda s: s["run"].update(event="pull_request_target"),
    "details url other repo": lambda s: s["checks"][1].update(
        details_url=f"https://github.com/other/app/actions/runs/{RUN}/job/12"),
    "details url other job": lambda s: s["checks"][1].update(
        details_url=f"https://github.com/acme/app/actions/runs/{RUN}/job/13"),
    "details url not actions": lambda s: s["checks"][1].update(details_url="https://ci.example/job/12"),
    "wrong app": lambda s: s["checks"][1].update(app={"id": ACTIONS, "slug": "impostor"}),
    "unpinned requirement": lambda s: s.update(required=[("unit-server", None)]),
    "fork head": lambda s: s["pr"].update(head={"sha": HEAD, "repo": {"full_name": "fork/app"}}),
    # The policy itself, or the evidence about changed paths, is not trustworthy.
    "workflow changed in PR": lambda s: s["files"].append({"filename": WORKFLOW_PATH, "status": "modified"}),
    "workflow differs at head": lambda s: s["contents"].update(
        {(WORKFLOW_PATH, HEAD): s["contents"][(WORKFLOW_PATH, HEAD)] + "# edited\n"}),
    "filter script changed": lambda s: s["contents"].update({(".github/scripts/path-filter.sh", HEAD): "#!/bin/sh\nexit 0\n"}),
    "incomplete file pagination": lambda s: s.update(changed_files=151),
    "file list over api cap": lambda s: s.update(changed_files=3001),
    "incomplete job pagination": lambda s: s.update(jobs_total=99),
    # Same-named context from a second, unexplained run is judged on its own evidence.
    "duplicate context": lambda s: s["checks"].append(_check(22, "unit-server", "skipped", run=901)) or
        s["jobs"].append({"id": 22, "name": "unit-server", "status": "completed", "conclusion": "skipped",
                          "run_id": 901}) or s.update(runs={RUN: s["run"], 901: {**s["run"], "id": 901,
                                                                                    "path": ".github/workflows/other.yml"}}) or
        s["contents"].update({(".github/workflows/other.yml", ref): "jobs:\n  unit:\n    name: unit-server\n"
                              for ref in (BASE, HEAD)}),
    "neutral": lambda s: s["checks"][1].update(conclusion="neutral"),
    "stale head race": lambda s: s.update(race=True),
    "base sha race": lambda s: s.update(base_race=True),
}
RACES = {"stale head race", "base sha race"}


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("case", sorted(REJECTIONS))
def test_unprovable_skip_is_rejected(github, case):
    REJECTIONS[case](github["s"])
    with connect() as conn:
        ok, receipt = _complete(conn)
    assert not ok and not receipt["ok"]
    assert receipt["classification"] != "success"
    assert not any(c["classification"] == "success" and c["conclusion"] == "skipped" for c in receipt["checks"])
    if case in RACES:
        # Evidence read against a head/base that no longer is the PR's is never accepted.
        assert receipt["classification"] == "stale"
    elif case != "neutral":
        # Rejected by the skip proof itself, not by an incidental evidence/API failure.
        assert receipt["classification"] == "unproven_skip"
        assert all(c.get("detail") for c in receipt["checks"] if c["classification"] == "unproven_skip")


@pytest.mark.platforms("linux")
def test_skip_is_judged_by_the_declared_filters_not_by_the_scripts_output(github):
    """The proof never trusts what the planner script emitted: an unchanged script that
    ignores its rules and always writes false cannot launder a backend change."""
    s = github["s"]
    always_false = '#!/bin/sh\ncat >/dev/null\nfor f in server schema fixtures; do echo "$f=false" >>"$GITHUB_OUTPUT"; done\n'
    s["contents"].update({(".github/scripts/path-filter.sh", ref): always_false for ref in (BASE, HEAD)})
    s["files"].append({"filename": "server/app.py", "status": "modified"})
    with connect() as conn:
        ok, receipt = _complete(conn)
    assert not ok and receipt["classification"] == "unproven_skip"
    unit = next(c for c in receipt["checks"] if c["name"] == "unit-server")
    assert unit["classification"] == "unproven_skip" and "server/app.py" in unit["detail"] and "proof" not in unit
