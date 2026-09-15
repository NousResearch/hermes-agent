"""PR acceptance regressions use real SQLite and a local GitHub HTTP contract."""
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
    state = {
        "conclusion": "success", "head": "a" * 40, "reads": 0, "requests": [],
        "pr_state": "OPEN", "rest_state": "open", "rest_merged": False,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": state["pr_state"],
                    "baseRef": {"branchProtectionRule": {
                        "id": state.get("classic_id", 99),
                        "requiredStatusChecks": [
                        *state.get("classic_checks", [
                            {"context": "required", "app": {"databaseId": 1}},
                        ])]}}}}}}
            elif "/rules/branches/" in self.path:
                if state.get("rules_status"):
                    self.send_response(state["rules_status"])
                    self.end_headers()
                    return
                value = state.get("rules_pages", [[]])
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                run.update(state.get("check_run_overrides", {}))
                for key in state.get("check_run_omit", []):
                    run.pop(key, None)
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                total_count = state.get("check_total_count", 100 + len(runs))
                value = [{"total_count": total_count, "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)]}, {"total_count": total_count, "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = state.get("status_pages", [[]])
            elif "/files" in self.path:
                value = state.get("file_pages", [[
                    {"filename": filename, "status": "modified"}
                    for filename in state.get("files", [])
                ]])
            elif "/commits/" in self.path:
                value = state.get("commit_pages", [{
                    "sha": state.get("merge_commit_sha"),
                    "parents": state.get("parents", [
                        {"sha": state.get("base_sha", "b" * 40)},
                        {"sha": state["head"]},
                    ]),
                    "files": [
                        {"filename": filename, "status": "modified"}
                        for filename in state.get("merge_files", state.get("files", []))
                    ],
                }])
            elif "/pulls/" in self.path:
                value = {
                    "head": {"sha": sha},
                    "base": {"ref": "main", "sha": state.get("base_sha", "b" * 40)},
                    "state": state["rest_state"], "merged": state["rest_merged"],
                    "merge_commit_sha": state.get("merge_commit_sha"),
                }
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
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.error,urllib.request\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "try:\n"
                  "    print(urllib.request.urlopen(u).read().decode())\n"
                  "except urllib.error.HTTPError as exc:\n"
                  "    print(f'HTTP {exc.code}', file=sys.stderr)\n"
                  "    raise SystemExit(1)\n")
    gh.chmod(0o755)
    state["gh"] = gh
    state["shim"] = shim
    git = shim / "git"
    git.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "if sys.argv[1:2] in ([\"fetch\"], [\"merge-base\"]):\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:2] == [\"rev-parse\"]:\n"
        "    print(\"d\" * 40)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    git.chmod(0o755)
    state["git"] = git
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _configure_merged_pr(github, *, files=None, merge_files=None):
    reviewed_head = "a" * 40
    allowlist = files or [
        "hermes_cli/kanban_pr_acceptance.py",
        "tests/hermes_cli/test_kanban_pr_acceptance.py",
    ]
    github.update(
        head=reviewed_head, base_sha="b" * 40, merge_commit_sha="c" * 40,
        pr_state="MERGED", rest_state="closed", rest_merged=True,
        files=allowlist, merge_files=merge_files or allowlist,
    )
    return reviewed_head, allowlist


def _configure_git_reachability(github, *, reachable):
    github["git"].write_text(
        f"#!{sys.executable}\nimport sys\n"
        "if sys.argv[1:2] == [\"fetch\"]:\n"
        "    raise SystemExit(0)\n"
        f"if sys.argv[1:2] == [\"merge-base\"]:\n    raise SystemExit({0 if reachable else 1})\n"
        "if sys.argv[1:2] == [\"rev-parse\"]:\n    print(\"d\" * 40)\n    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    github["git"].chmod(0o755)


@pytest.mark.linux_only
def test_pr_completion_requires_current_required_evidence(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    with connect() as conn:
        for conclusion in ("failure", "pending", "cancelled", "timed_out", "action_required", "neutral", "skipped", None, "success"):
            github.update(
                conclusion=conclusion, head="a" * 40,
                pr_state="MERGED", rest_state="closed", rest_merged=True,
            )
            tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
            ok = kb.complete_task(conn, tid, metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            })
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
            github.update(
                conclusion="success", head="a" * 40,
                pr_state="MERGED", rest_state="closed", rest_merged=True,
            )
            github[fault] = True
            tid = kb.create_task(conn, title=fault, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            })
            assert kb.get_task(conn, tid).status != "done"
            github.pop(fault)
        # Omission and a sibling repository cannot downgrade the stored declaration.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, summary="local green")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/other/repo/pull/7"})
        before = len(github["requests"])
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="https://github.com/acme/repo/pull/7 is background context")
        assert len(github["requests"]) == before


@pytest.mark.linux_only
def test_acceptance_receipts_and_terminal_write_share_run_ownership(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
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
            assert not kb.complete_task(conn, tid, expected_run_id=run_id,
                metadata={
                    "published_pr": "https://github.com/acme/repo/pull/7",
                    "reviewed_head_sha": reviewed_head,
                    "approved_allowlist": allowlist,
                })
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


@pytest.mark.linux_only
def test_open_pr_never_satisfies_done_even_when_checks_are_green(github):
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn, tid,
            metadata={"published_pr": "https://github.com/acme/repo/pull/7"},
        ) is False
        assert kb.get_task(conn, tid).status != "done"


@pytest.mark.linux_only
def test_merged_pr_requires_reviewed_head_scope_and_reachable_merge(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    base_sha = "b" * 40
    merge_sha = "c" * 40

    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is True
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["reviewed_head_sha"] == reviewed_head
    assert receipt["base_sha"] == base_sha
    assert receipt["merge_commit_sha"] == merge_sha
    assert receipt["target"]["reachable"] is True
    assert receipt["target"]["sha"] == "d" * 40
    assert receipt["scope"]["whole_pr"] == allowlist
    assert receipt["scope"]["merged_delta"] == allowlist
    assert receipt["check_counts"]["check_runs"] == {
        "pages": 2, "count": 101, "total_count": 101,
    }


@pytest.mark.linux_only
def test_merged_pr_rejects_a_head_that_was_not_independently_reviewed(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": "d" * 40,
                "approved_allowlist": allowlist,
            },
        ) is False
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["classification"] == "stale"
    assert receipt["head_sha"] == reviewed_head


@pytest.mark.linux_only
def test_rules_api_plan_403_falls_back_to_classic_branch_protection(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    github["rules_status"] = 403
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is True
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["policy"]["rules_source"] == "unavailable_http_403"
    assert receipt["required"] == [{"context": "required", "app_id": 1}]
    assert receipt["policy"]["classic_ids"] == [99]


@pytest.mark.linux_only
def test_merged_pr_without_a_merge_commit_is_missing_evidence(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    github["merge_commit_sha"] = None
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is False
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["classification"] == "missing"
    assert receipt["merge_commit_sha"] is None


@pytest.mark.linux_only
@pytest.mark.parametrize(
    ("classic_check", "run_overrides", "run_omit", "classification"),
    [
        ({"context": "required"}, {}, [], "infra"),
        ({"context": "required", "app": {}}, {}, [], "infra"),
        ({"context": "required", "app": None}, {"app": {"id": 2}}, [], "success"),
        ({"context": "required", "app": {"databaseId": None}}, {"app": {"id": 2}}, [], "success"),
        ({"context": "required", "app": {"databaseId": 1}}, {"app": {"id": 2}}, [], "missing"),
        ({"context": "required", "app": {"databaseId": 1}}, {}, ["app"], "infra"),
    ],
)
def test_required_check_identity_presence_and_null_pinning_are_fail_closed(
    github, classic_check, run_overrides, run_omit, classification,
):
    reviewed_head, allowlist = _configure_merged_pr(github)
    github["classic_checks"] = [classic_check]
    github["check_run_overrides"] = run_overrides
    github["check_run_omit"] = run_omit
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        completed = kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        )
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["classification"] == classification
    assert completed is (classification == "success")


@pytest.mark.linux_only
def test_unpinned_rules_check_accepts_any_app_but_pinned_rules_check_rejects_other_apps(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    github["classic_checks"] = []
    github["rules_pages"] = [[{
        "type": "required_status_checks",
        "id": 7,
        "parameters": {"required_status_checks": [{"context": "required", "integration_id": -1}]},
    }]]
    github["check_run_overrides"] = {"app": {"id": 2}}
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is True
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["required"] == [{"context": "required", "app_id": -1}]
    assert receipt["checks"][0]["app_id"] == 2


@pytest.mark.linux_only
def test_legacy_status_uses_latest_exact_head_entry(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    github["classic_checks"] = [{"context": "legacy", "app": None}]
    github["check_run_overrides"] = {"name": "not-legacy"}
    github["status_pages"] = [[
        {"id": 10, "context": "legacy", "sha": reviewed_head, "state": "failure"},
        {"id": 11, "context": "legacy", "sha": reviewed_head, "state": "success"},
    ]]
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is True
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["checks"] == [{
        "name": "legacy",
        "id": 11,
        "type": "status",
        "app_id": None,
        "url": None,
        "head_sha": reviewed_head,
        "classification": "success",
        "conclusion": "success",
    }]


@pytest.mark.linux_only
def test_one_parent_merge_requires_an_explicit_squash_source(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    github["parents"] = [{"sha": "b" * 40}]
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is False
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["classification"] == "stale"


@pytest.mark.linux_only
def test_merged_pr_requires_reachable_merge_commit_from_fresh_target(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    _configure_git_reachability(github, reachable=False)
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is False
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["classification"] == "unreachable"
    assert receipt["target"]["ref"] == "origin/main"
    assert receipt["target"]["fetch"]["outcome"] == "success"
    assert receipt["target"]["reachability"]["outcome"] == "unreachable"


@pytest.mark.linux_only
def test_squash_merge_requires_and_records_its_explicit_method(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    github["parents"] = [{"sha": "b" * 40}]
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
                "merge_method": "squash",
            },
        ) is True
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["merge_method"] == "squash"
    assert receipt["merge_parents"] == ["b" * 40]


@pytest.mark.linux_only
def test_review_evidence_may_supply_head_and_allowlist_as_one_nested_record(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    github["parents"] = [{"sha": "b" * 40}]
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "review": {
                    "reviewed_head_sha": reviewed_head,
                    "approved_allowlist": allowlist,
                    "merge_method": "squash",
                },
            },
        ) is True


@pytest.mark.linux_only
def test_merged_scope_must_match_allowlist_for_both_pr_and_merge_delta(github):
    reviewed_head, allowlist = _configure_merged_pr(
        github,
        merge_files=[
            "hermes_cli/kanban_pr_acceptance.py",
            "tests/hermes_cli/test_kanban_pr_acceptance.py",
            "unapproved.txt",
        ],
    )
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is False
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["classification"] == "failure"
    assert receipt["scope"]["whole_pr"] == allowlist
    assert "unapproved.txt" in receipt["scope"]["merged_delta"]


@pytest.mark.linux_only
def test_check_run_count_mismatch_is_infrastructure_failure(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    github["check_total_count"] = 102
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is False
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["classification"] == "infra"
    assert receipt["checks"] == []


@pytest.mark.linux_only
@pytest.mark.parametrize("race_kind", ["pr", "policy"])
def test_final_pr_or_policy_reread_race_is_stale(github, race_kind):
    reviewed_head, allowlist = _configure_merged_pr(github)

    def race():
        if race_kind == "pr":
            github["pr_state"] = "OPEN"
            github["rest_state"] = "open"
            github["rest_merged"] = False
        else:
            github["rules_status"] = 403

    github["race"] = race
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(
            conn,
            tid,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is False
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0])

    assert receipt["classification"] == "stale"


@pytest.mark.linux_only
def test_acceptance_receipts_are_redacted_append_only_and_do_not_partially_complete(github):
    reviewed_head, allowlist = _configure_merged_pr(github)
    original_gh = github["gh"].read_text()
    github["gh"].write_text(
        f"#!{sys.executable}\nimport sys\n"
        "print('authorization token=super-secret', file=sys.stderr)\n"
        "raise SystemExit(1)\n"
    )
    github["gh"].chmod(0o755)
    metadata = {
        "published_pr": "https://github.com/acme/repo/pull/7",
        "reviewed_head_sha": reviewed_head,
        "approved_allowlist": allowlist,
        "actor": "authorization token=super-secret",
    }
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, metadata=metadata) is False
        github["gh"].write_text(original_gh)
        github["gh"].chmod(0o755)
        github["rules_status"] = 500
        assert kb.complete_task(conn, tid, metadata=metadata) is False
        github.pop("rules_status")
        github["check_run_overrides"] = {
            "html_url": "https://github.com/acme/repo/actions/runs/42?token=super-secret",
        }
        assert kb.complete_task(conn, tid, metadata=metadata) is True
        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id",
            (tid,),
        ).fetchall()
        task = kb.get_task(conn, tid)

    receipts = [json.loads(row[0]) for row in rows]
    assert task.status == "done"
    assert len(receipts) == 3
    assert [receipt["classification"] for receipt in receipts] == ["infra", "infra", "success"]
    assert all("super-secret" not in json.dumps(receipt) for receipt in receipts)
    assert all("authorization token" not in json.dumps(receipt) for receipt in receipts)

    with connect() as conn:
        invalid_tid = kb.create_task(conn, title="Invalid PR", completion_contract="acme/repo")
        assert not kb.complete_task(
            conn,
            invalid_tid,
            metadata={"published_pr": f"https://github.com/acme/repo/pull/7?token=super-secret"},
        )
        invalid_payload = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (invalid_tid,),
        ).fetchone()["payload"]
    assert "super-secret" not in invalid_payload


@pytest.mark.linux_only
def test_success_receipt_rolls_back_when_terminal_cas_loses_after_acceptance(github, monkeypatch):
    reviewed_head, allowlist = _configure_merged_pr(github)
    from hermes_cli import kanban_pr_acceptance_store as store

    original_record = store.record_acceptance

    def record_then_reassign(conn, task_id, acceptance):
        accepted = original_record(conn, task_id, acceptance)
        conn.execute(
            "UPDATE tasks SET status='blocked', current_run_id=current_run_id + 1 WHERE id=?",
            (task_id,),
        )
        return accepted

    monkeypatch.setattr(store, "record_acceptance", record_then_reassign)
    with connect() as conn:
        tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
        run_id = kb.claim_task(conn, tid).current_run_id
        assert kb.complete_task(
            conn,
            tid,
            expected_run_id=run_id,
            metadata={
                "published_pr": "https://github.com/acme/repo/pull/7",
                "reviewed_head_sha": reviewed_head,
                "approved_allowlist": allowlist,
            },
        ) is False
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == run_id
        assert conn.execute(
            "SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)
        ).fetchone()[0] == 0
