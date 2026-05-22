"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Workspace flag parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("scratch",              ("scratch", None)),
        ("worktree",              ("worktree", None)),
        ("worktree:/tmp/wt",       ("worktree", "/tmp/wt")),
        ("dir:/tmp/work",         ("dir", "/tmp/work")),
    ],
)
def test_parse_workspace_flag_valid(value, expected):
    assert kc._parse_workspace_flag(value) == expected


def test_parse_workspace_flag_expands_user():
    kind, path = kc._parse_workspace_flag("dir:~/vault")
    assert kind == "dir"
    assert path.endswith("/vault")
    assert not path.startswith("~")

    kind, path = kc._parse_workspace_flag("worktree:~/trees/t6-wire")
    assert kind == "worktree"
    assert path.endswith("/trees/t6-wire")
    assert not path.startswith("~")

@pytest.mark.parametrize("bad", ["cloud", "dir:", "worktree:", ""])
def test_parse_workspace_flag_rejects(bad):
    if not bad:
        # Empty -> defaults; not an error.
        assert kc._parse_workspace_flag(bad) == ("scratch", None)
        return
    with pytest.raises(argparse.ArgumentTypeError):
        kc._parse_workspace_flag(bad)


def test_parse_branch_flag_rejects_empty_and_option_like():
    assert kc._parse_branch_flag(None) is None
    assert kc._parse_branch_flag(" wt/t6-wire ") == "wt/t6-wire"
    with pytest.raises(argparse.ArgumentTypeError):
        kc._parse_branch_flag("   ")
    with pytest.raises(argparse.ArgumentTypeError):
        kc._parse_branch_flag("-bad")
    with pytest.raises(argparse.ArgumentTypeError):
        kc._parse_branch_flag("bad branch")


# ---------------------------------------------------------------------------
# run_slash smoke tests (end-to-end via the same entry both CLI and gateway use)
# ---------------------------------------------------------------------------

def test_run_slash_no_args_shows_usage(kanban_home):
    out = kc.run_slash("")
    assert "kanban" in out.lower()
    assert "create" in out.lower() or "subcommand" in out.lower() or "action" in out.lower()


def test_run_slash_create_and_list(kanban_home):
    out = kc.run_slash("create 'ship feature' --assignee alice")
    assert "Created" in out
    out = kc.run_slash("list")
    assert "ship feature" in out
    assert "alice" in out


def test_run_slash_create_worktree_path_and_branch(kanban_home, tmp_path):
    target = tmp_path / ".worktrees" / "t6-wire"
    target_arg = target.as_posix()
    out = kc.run_slash(
        f"create 'ship worktree' --workspace worktree:{target_arg} --branch wt/t6-wire"
    )
    assert "Created" in out

    with kb.connect() as conn:
        tasks = kb.list_tasks(conn)
    task = tasks[0]
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == target_arg
    assert task.branch_name == "wt/t6-wire"


def test_run_slash_rejects_branch_without_worktree(kanban_home):
    out = kc.run_slash("create 'bad branch' --workspace scratch --branch wt/bad")
    assert "--branch is only valid with --workspace worktree" in out


def test_run_slash_create_with_parent_and_cascade(kanban_home):
    # Parent then child via --parent
    out1 = kc.run_slash("create 'parent' --assignee alice")
    # Extract the "t_xxxx" id from "Created t_xxxx (ready, ...)"
    import re
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    p = m.group(1)
    out2 = kc.run_slash(f"create 'child' --assignee bob --parent {p}")
    assert "todo" in out2  # child starts as todo

    # Complete parent; list should promote child to ready
    kc.run_slash(f"complete {p}")
    # Explicit filter: child should now be ready (was todo before complete).
    ready_list = kc.run_slash("list --status ready")
    assert "child" in ready_list


def test_run_slash_show_includes_comments(kanban_home):
    out = kc.run_slash("create 'x'")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    kc.run_slash(f"comment {tid} 'remember to include performance section'")
    show = kc.run_slash(f"show {tid}")
    assert "performance section" in show


def test_run_slash_comment_max_len_trims_long_body(kanban_home):
    out = kc.run_slash("create 'x'")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    kc.run_slash(f"comment {tid} '{'x' * 30}' --max-len 20")
    show = kc.run_slash(f"show {tid}")
    assert "trimmed to 20 chars by --max-len" in show
    assert "x" * 30 not in show


def test_run_slash_block_unblock_cycle(kanban_home):
    out = kc.run_slash("create 'x' --assignee alice")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    # Claim first so block() finds it running
    kc.run_slash(f"claim {tid}")
    assert "Blocked" in kc.run_slash(f"block {tid} 'need decision'")
    assert "Unblocked" in kc.run_slash(f"unblock {tid}")


def test_run_slash_json_output(kanban_home):
    out = kc.run_slash("create 'jsontask' --assignee alice --json")
    payload = json.loads(out)
    assert payload["title"] == "jsontask"
    assert payload["assignee"] == "alice"
    assert payload["status"] == "ready"


def test_run_slash_dispatch_dry_run_counts(kanban_home):
    kc.run_slash("create 'a' --assignee alice")
    kc.run_slash("create 'b' --assignee bob")
    out = kc.run_slash("dispatch --dry-run")
    assert "Spawned:" in out


def test_run_slash_context_output_format(kanban_home):
    out = kc.run_slash("create 'tech spec' --assignee alice --body 'write an RFC'")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    kc.run_slash(f"comment {tid} 'remember to include performance section'")
    ctx = kc.run_slash(f"context {tid}")
    assert "tech spec" in ctx
    assert "write an RFC" in ctx
    assert "performance section" in ctx


def test_run_slash_tenant_filter(kanban_home):
    kc.run_slash("create 'biz-a task' --tenant biz-a --assignee alice")
    kc.run_slash("create 'biz-b task' --tenant biz-b --assignee alice")
    a = kc.run_slash("list --tenant biz-a")
    b = kc.run_slash("list --tenant biz-b")
    assert "biz-a task" in a and "biz-b task" not in a
    assert "biz-b task" in b and "biz-a task" not in b


def test_run_slash_session_filter(kanban_home):
    """`hermes kanban list --session <id>` filters by the originating
    chat session id stamped on tasks created from inside an ACP loop."""
    from hermes_cli import kanban_db as kb
    with kb.connect() as conn:
        kb.create_task(
            conn, title="from sess-1 a", assignee="alice", session_id="sess-1"
        )
        kb.create_task(
            conn, title="from sess-1 b", assignee="alice", session_id="sess-1"
        )
        kb.create_task(
            conn, title="from sess-2", assignee="alice", session_id="sess-2"
        )
        kb.create_task(conn, title="cli only", assignee="alice")
    out_1 = kc.run_slash("list --session sess-1")
    out_2 = kc.run_slash("list --session sess-2")
    assert "from sess-1 a" in out_1
    assert "from sess-1 b" in out_1
    assert "from sess-2" not in out_1
    assert "cli only" not in out_1
    assert "from sess-2" in out_2
    assert "from sess-1 a" not in out_2


def test_kanban_list_json_includes_session_id(kanban_home):
    """JSON output exposes `session_id` so external clients (Scarf, web
    dashboards) don't need a side query to filter by chat session."""
    from hermes_cli import kanban_db as kb
    with kb.connect() as conn:
        kb.create_task(
            conn, title="acp task", assignee="alice", session_id="acp-x"
        )
    raw = kc.run_slash("list --json")
    payload = json.loads(raw)
    assert any(
        row.get("title") == "acp task"
        and row.get("session_id") == "acp-x"
        for row in payload
    )


def test_run_slash_usage_error_returns_message(kanban_home):
    # Missing required argument for create
    out = kc.run_slash("create")
    assert "usage" in out.lower() or "error" in out.lower()


def test_run_slash_assign_reassigns(kanban_home):
    out = kc.run_slash("create 'x' --assignee alice")
    import re
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    assert "Assigned" in kc.run_slash(f"assign {tid} bob")
    show = kc.run_slash(f"show {tid}")
    assert "bob" in show


def test_run_slash_link_unlink(kanban_home):
    a = kc.run_slash("create 'a'")
    b = kc.run_slash("create 'b'")
    import re
    ta = re.search(r"(t_[a-f0-9]+)", a).group(1)
    tb = re.search(r"(t_[a-f0-9]+)", b).group(1)
    assert "Linked" in kc.run_slash(f"link {ta} {tb}")
    # After link, b is todo
    show = kc.run_slash(f"show {tb}")
    assert "todo" in show
    assert "Unlinked" in kc.run_slash(f"unlink {ta} {tb}")


# ---------------------------------------------------------------------------
# Integration with the COMMAND_REGISTRY
# ---------------------------------------------------------------------------

def test_kanban_is_resolvable():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("kanban")
    assert cmd is not None
    assert cmd.name == "kanban"


def test_kanban_bypasses_active_session_guard():
    from hermes_cli.commands import should_bypass_active_session

    assert should_bypass_active_session("kanban")


def test_kanban_in_autocomplete_table():
    from hermes_cli.commands import COMMANDS, SUBCOMMANDS

    assert "/kanban" in COMMANDS
    subs = SUBCOMMANDS.get("/kanban") or []
    assert "create" in subs
    assert "dispatch" in subs


def test_kanban_autocomplete_includes_live_subcommands():
    from prompt_toolkit.document import Document

    from hermes_cli.commands import SlashCommandCompleter

    completer = SlashCommandCompleter()
    doc = Document("/kanban sp", cursor_position=len("/kanban sp"))
    texts = {c.text for c in completer.get_completions(doc, None)}

    assert "specify" in texts

    doc = Document("/kanban re", cursor_position=len("/kanban re"))
    texts = {c.text for c in completer.get_completions(doc, None)}

    assert "reclaim" in texts
    assert "reassign" in texts


def test_kanban_not_gateway_only():
    # kanban is available in BOTH CLI and gateway surfaces.
    from hermes_cli.commands import COMMAND_REGISTRY

    cmd = next(c for c in COMMAND_REGISTRY if c.name == "kanban")
    assert not cmd.cli_only
    assert not cmd.gateway_only


# ---------------------------------------------------------------------------
# reclaim + reassign CLI smoke tests
# ---------------------------------------------------------------------------

def test_run_slash_reclaim_running_task(kanban_home):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db as kb

    out1 = kc.run_slash("create 'stuck worker task' --assignee broken-model")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    tid = m.group(1)

    # Simulate a running claim outside TTL.
    conn = kb.connect()
    try:
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 4242, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, int(time.time()) + 3600, 4242, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, tid))
        conn.commit()
    finally:
        conn.close()

    out = kc.run_slash(f"reclaim {tid} --reason 'test'")
    assert "Reclaimed" in out, out
    # Status back to ready.
    out2 = kc.run_slash(f"show {tid}")
    assert "ready" in out2.lower()


def test_run_slash_reassign_with_reclaim_flag(kanban_home):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db as kb

    out1 = kc.run_slash("create 'switch model' --assignee orig")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    tid = m.group(1)

    # Simulate a running claim.
    conn = kb.connect()
    try:
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 4242, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, int(time.time()) + 3600, 4242, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, tid))
        conn.commit()
    finally:
        conn.close()

    out = kc.run_slash(f"reassign {tid} newbie --reclaim --reason 'switch'")
    assert "Reassigned" in out, out
    out2 = kc.run_slash(f"show {tid}")
    assert "newbie" in out2


def test_run_slash_progress_json_is_read_only(kanban_home):
    import re
    import time
    import secrets

    out1 = kc.run_slash("create 'external progress' --assignee codex-deep")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    tid = m.group(1)

    with kb.connect() as conn:
        lock = secrets.token_hex(4)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=?, current_run_id=NULL WHERE id=?",
            (lock, now + 3600, 4242, tid),
        )
        cur = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, "
            "claim_expires, worker_pid, started_at) VALUES (?, ?, 'running', ?, ?, ?, ?)",
            (tid, "codex-deep", lock, now + 3600, 4242, now),
        )
        run_id = cur.lastrowid
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid))
        kb.record_task_event(
            conn,
            tid,
            "worker_progress",
            {"lane": "codex-deep", "items": [{"index": 1, "status": "done", "text": "mock"}]},
            run_id=run_id,
        )
        before = kb.get_task(conn, tid)

    payload = json.loads(kc.run_slash(f"progress {tid} --json"))

    with kb.connect() as conn:
        after = kb.get_task(conn, tid)
    assert payload["task"]["status"] == "running"
    assert payload["task"]["worker_pid"] == 4242
    assert payload["worker_progress"]["items"][0]["text"] == "mock"
    assert after.status == "running"
    assert after.claim_lock == before.claim_lock


def test_run_slash_progress_children_json_summarizes_goal_workers(kanban_home):
    with kb.connect() as conn:
        root = kb.create_task(conn, title="goal", triage=True)
        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "implement", "assignee": "codex-fast"},
                {"title": "review", "assignee": "codex-deep"},
            ],
            author="planner",
        )
        assert child_ids is not None
        running_id, review_id = child_ids

        running = kb.claim_task(conn, running_id, claimer="worker:fast")
        assert running is not None
        kb.record_task_event(
            conn,
            running_id,
            "worker_progress",
            {"lane": "codex-fast", "items": [{"index": 1, "status": "running", "text": "mock"}]},
            run_id=running.current_run_id,
        )
        reviewing = kb.claim_task(conn, review_id, claimer="worker:deep")
        assert reviewing is not None
        assert kb.block_task(
            conn,
            review_id,
            reason="review-required: Codex completed; Hermes review required",
            expected_run_id=reviewing.current_run_id,
            metadata={
                "worker_lane": {"name": "codex-deep", "kind": "codex_cli", "exit_code": 0},
                "verification": {"commands": ["pytest -q"], "summary": "passed"},
                "review": {"required": True, "reason": "Codex completed; Hermes review required"},
            },
        )
        before = kb.get_task(conn, running_id)

    payload = json.loads(kc.run_slash(f"progress {root} --children --json"))

    with kb.connect() as conn:
        after = kb.get_task(conn, running_id)

    assert payload["task"]["id"] == root
    assert payload["child_summary"]["total"] == 2
    assert payload["child_summary"]["running"] == 1
    assert payload["child_summary"]["review_required"] == 1
    assert payload["child_summary"]["relationship_counts"]["decomposed_child"] == 2
    by_id = {child["task"]["id"]: child for child in payload["children"]}
    assert by_id[running_id]["worker_progress"]["items"][0]["text"] == "mock"
    assert by_id[review_id]["worker_lane"]["name"] == "codex-deep"
    assert by_id[review_id]["verification"]["commands"] == ["pytest -q"]
    assert after.status == "running"
    assert after.claim_lock == before.claim_lock


def test_run_slash_reviews_lists_review_required_evidence(
    kanban_home, tmp_path,
):
    metadata = {
        "worker_lane": {"name": "codex-deep", "kind": "codex_cli", "exit_code": 0},
        "verification": {"commands": ["pytest -q"], "summary": "passed"},
        "git": {"changed_files": ["hermes_cli/kanban.py"], "diff_summary": "+2 -0"},
        "review": {"required": True, "reason": "Codex completed; Hermes review required"},
    }
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="external review",
            assignee="codex-deep",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        task = kb.claim_task(conn, tid, claimer="worker:codex-deep")
        assert task is not None
        assert kb.block_task(
            conn,
            tid,
            reason="review-required: Codex completed; Hermes review required",
            expected_run_id=task.current_run_id,
            metadata=metadata,
        )

    payload = json.loads(kc.run_slash("reviews --json"))
    assert [item["task"]["id"] for item in payload] == [tid]
    assert payload[0]["worker_lane"]["name"] == "codex-deep"
    assert payload[0]["verification"]["commands"] == ["pytest -q"]
    assert payload[0]["evidence"]["review"]["required"] is True

    human = kc.run_slash("reviews --lane codex-deep")
    assert tid in human
    assert "codex-deep" in human
    assert "review-required: Codex completed" in human


def test_run_slash_worker_lanes_lists_active_instances(kanban_home):
    from hermes_cli.worker_lanes import WorkerLane, clear_worker_lanes, register_worker_lane

    clear_worker_lanes()
    register_worker_lane(WorkerLane(
        name="codex-deep",
        kind="codex_cli",
        description="Deep Codex lane",
        spawn_fn=lambda task, workspace, **kwargs: 5100,
        max_concurrency=2,
        source="test",
        config={"type": "codex_cli", "model": "gpt-5.5", "secret": "hidden"},
    ))
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="active lane task", assignee="codex-deep")
        res = kb.dispatch_once(conn, max_spawn=1)
        assert res.spawned[0][0] == tid

    payload = json.loads(kc.run_slash("worker-lanes --json"))

    assert payload[0]["name"] == "codex-deep"
    assert payload[0]["active_count"] == 1
    assert payload[0]["available_capacity"] == 1
    assert payload[0]["active"][0]["task_id"] == tid
    assert payload[0]["active"][0]["worker_pid"] == 5100
    assert payload[0]["config"]["model"] == "gpt-5.5"
    assert "secret" not in payload[0]["config"]

    human = kc.run_slash("worker-lanes")
    assert "codex-deep" in human
    assert tid in human
    assert "ACTIVE" in human


def test_run_slash_review_approve_completes_review_required_task(
    kanban_home, tmp_path,
):
    metadata = {
        "worker_lane": {"name": "codex-deep", "kind": "codex_cli", "exit_code": 0},
        "verification": {"commands": ["pytest -q"], "summary": "passed"},
        "review": {"required": True, "reason": "Codex completed; Hermes review required"},
    }
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="approve via slash",
            assignee="codex-deep",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        task = kb.claim_task(conn, tid, claimer="worker:codex-deep")
        assert task is not None
        assert kb.block_task(
            conn,
            tid,
            reason="review-required: Codex completed; Hermes review required",
            expected_run_id=task.current_run_id,
            metadata=metadata,
        )

    payload = json.loads(kc.run_slash(
        f"review {tid} approve --reviewer ralph --summary 'bounded evidence approved' --json"
    ))

    assert payload["task"]["status"] == "done"
    assert payload["evidence"]["review"]["decision"] == "approved"
    assert payload["evidence"]["review"]["reviewer"] == "ralph"
    assert payload["review_required"] is False


def test_run_slash_review_request_changes_unblocks_for_next_worker(
    kanban_home, tmp_path,
):
    metadata = {
        "worker_lane": {"name": "codex-deep", "kind": "codex_cli", "exit_code": 0},
        "verification": {"commands": ["pytest -q"], "summary": "failed"},
        "review": {"required": True, "reason": "Codex completed; Hermes review required"},
    }
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="request changes via slash",
            assignee="codex-deep",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        task = kb.claim_task(conn, tid, claimer="worker:codex-deep")
        assert task is not None
        assert kb.block_task(
            conn,
            tid,
            reason="review-required: Codex completed; Hermes review required",
            expected_run_id=task.current_run_id,
            metadata=metadata,
        )

    out = kc.run_slash(
        f"review {tid} request-changes --reviewer ralph "
        "--comment 'add a focused regression test'"
    )

    assert f"Requested changes for {tid}" in out
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)
    assert task.status == "ready"
    assert "focused regression test" in comments[-1].body
    assert any(event.kind == "worker_review_changes_requested" for event in events)


def test_run_slash_plan_review_json_creates_followups(
    kanban_home, tmp_path,
):
    metadata = {
        "worker_lane": {"name": "codex-deep", "kind": "codex_cli", "exit_code": 0},
        "verification": {"commands": ["pytest -q"], "summary": "passed"},
        "git": {"changed_files": ["hermes_cli/kanban.py"], "diff_summary": "+2 -0"},
        "review": {"required": True, "reason": "Codex completed; Hermes review required"},
    }
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="plan review via slash",
            assignee="codex-deep",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        task = kb.claim_task(conn, tid, claimer="worker:codex-deep")
        assert task is not None
        assert kb.block_task(
            conn,
            tid,
            reason="review-required: Codex completed; Hermes review required",
            expected_run_id=task.current_run_id,
            metadata=metadata,
        )

    payload = json.loads(kc.run_slash(
        f"plan-review {tid} --review-assignee codex-review "
        "--test-assignee codex-test --json"
    ))

    with kb.connect() as conn:
        review_task = kb.get_task(conn, payload["review_task_id"])
        test_task = kb.get_task(conn, payload["test_task_id"])
        progress = kb.task_progress_snapshot(conn, tid, include_children=True)
        repeated = json.loads(kc.run_slash(f"plan-review {tid} --json"))

    assert set(payload["created"]) == {payload["review_task_id"], payload["test_task_id"]}
    assert payload["existing"] == []
    assert review_task.status == "ready"
    assert test_task.status == "ready"
    assert review_task.assignee == "codex-review"
    assert test_task.assignee == "codex-test"
    assert "hermes_cli/kanban.py" in review_task.body
    assert "pytest -q" in test_task.body
    assert progress.child_summary["relationship_counts"]["review_followup"] == 1
    assert progress.child_summary["relationship_counts"]["test_followup"] == 1
    assert progress.review_followup_gate["ready"] is False
    assert progress.review_followup_gate["pending"] == 2
    assert repeated["created"] == []
    assert set(repeated["existing"]) == {payload["review_task_id"], payload["test_task_id"]}

    acceptance = json.loads(kc.run_slash(f"acceptance {tid} --json"))
    assert acceptance["recommended_action"] == "wait_for_followups"
    assert acceptance["approval_allowed"] is False
    assert acceptance["review_followup_gate"]["pending"] == 2
    assert [item["purpose"] for item in acceptance["followups"]] == ["review", "test"]

    out = kc.run_slash(
        f"review {tid} approve --reviewer ralph --summary 'too early' --json"
    )
    assert "review follow-up gate is not satisfied" in out


def test_run_slash_plan_review_dispatch_dry_run_scopes_to_followups(
    kanban_home,
    tmp_path,
    all_assignees_spawnable,
):
    metadata = {
        "worker_lane": {"name": "codex-deep", "kind": "codex_cli", "exit_code": 0},
        "verification": {"commands": ["pytest -q"], "summary": "passed"},
        "review": {"required": True, "reason": "Codex completed; Hermes review required"},
    }
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="plan and dispatch followups via slash",
            assignee="codex-deep",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        unrelated = kb.create_task(
            conn,
            title="unrelated",
            assignee="alice",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        task = kb.claim_task(conn, tid, claimer="worker:codex-deep")
        assert task is not None
        assert kb.block_task(
            conn,
            tid,
            reason="review-required: Codex completed; Hermes review required",
            expected_run_id=task.current_run_id,
            metadata=metadata,
        )

    payload = json.loads(kc.run_slash(
        f"plan-review {tid} --dispatch --dry-run --json"
    ))
    spawned_ids = {item["task_id"] for item in payload["dispatch"]["spawned"]}

    with kb.connect() as conn:
        unrelated_task = kb.get_task(conn, unrelated)
        review_task = kb.get_task(conn, payload["review_task_id"])
        test_task = kb.get_task(conn, payload["test_task_id"])

    assert spawned_ids == {payload["review_task_id"], payload["test_task_id"]}
    assert unrelated_task.status == "ready"
    assert review_task.status == "ready"
    assert test_task.status == "ready"


def test_run_slash_verify_runs_configured_acceptance_check(
    kanban_home,
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "ok.txt").write_text("ok\n", encoding="utf-8")
    (kanban_home / "config.yaml").write_text(
        "kanban:\n"
        "  acceptance_checks:\n"
        "    exact-file:\n"
        "      argv: [python3, -c, \"from pathlib import Path; "
        "assert Path('ok.txt').read_text() == 'ok\\\\n'\"]\n",
        encoding="utf-8",
    )
    metadata = {
        "worker_lane": {"name": "codex-deep", "kind": "codex_cli", "exit_code": 0},
        "review": {"required": True, "reason": "Codex completed; Hermes review required"},
    }
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="verify via slash",
            assignee="codex-deep",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        task = kb.claim_task(conn, tid, claimer="worker:codex-deep")
        assert task is not None
        assert kb.block_task(
            conn,
            tid,
            reason="review-required: Codex completed; Hermes review required",
            expected_run_id=task.current_run_id,
            metadata=metadata,
        )

    payload = json.loads(kc.run_slash(f"verify {tid} exact-file --json"))
    acceptance = json.loads(kc.run_slash(f"acceptance {tid} --json"))

    assert payload["checks"][0]["name"] == "exact-file"
    assert payload["checks"][0]["passed"] is True
    assert acceptance["acceptance_check_gate"]["ready"] is True


def test_run_slash_worker_lane_request_validates_without_enabling(
    kanban_home, tmp_path,
):
    from hermes_cli.worker_lanes import get_worker_lane

    req = tmp_path / "lane.yaml"
    req.write_text(
        "worker_lane_request:\n"
        "  name: codex-cli-request\n"
        "  type: codex_cli\n"
        "  model: gpt-5.4-mini\n"
        "  sandbox: workspace-write\n"
        "  approval: never\n"
        "  max_concurrency: 1\n"
        "  success_policy: block_for_review\n",
        encoding="utf-8",
    )

    payload = json.loads(kc.run_slash(f"worker-lane-request {req} --json"))

    assert payload["valid"] is True
    assert payload["enabled"] is False
    assert payload["config"]["name"] == "codex-cli-request"
    assert get_worker_lane("codex-cli-request") is None


def test_run_slash_worker_lane_request_persist_enables_config_lane(
    kanban_home, tmp_path,
):
    from hermes_cli.worker_lanes import get_worker_lane
    from hermes_cli.config import read_raw_config

    req = tmp_path / "lane.json"
    req.write_text(json.dumps({
        "worker_lane_request": {
            "name": "codex-persisted",
            "type": "codex_cli",
            "model": "gpt-5.5",
            "sandbox": "workspace-write",
            "approval": "never",
            "max_concurrency": 1,
            "success_policy": "block_for_review",
            "reason": "approved by test",
        }
    }), encoding="utf-8")

    payload = json.loads(kc.run_slash(f"worker-lane-request {req} --persist --json"))

    assert payload["enabled"] is True
    assert payload["persisted"] is True
    lane = get_worker_lane("codex-persisted")
    assert lane is not None
    assert lane.source == "config"
    stored = read_raw_config()["kanban"]["worker_lanes"]["codex-persisted"]
    assert stored["type"] == "codex_cli"
    assert stored["model"] == "gpt-5.5"
    assert "reason" not in stored


def test_run_slash_worker_lane_request_rejects_shell_command(
    kanban_home, tmp_path,
):
    req = tmp_path / "lane.json"
    req.write_text(json.dumps({
        "name": "codex-bad",
        "type": "codex_cli",
        "command": "codex exec -",
    }), encoding="utf-8")

    out = kc.run_slash(f"worker-lane-request {req} --json")

    assert "may not include executable command fields" in out


def test_run_slash_goal_creates_top_level_task(kanban_home):
    payload = json.loads(kc.run_slash(
        "goal 'refactor the worker lane bridge' "
        "--session sess-goal-1 --assignee orchestrator --tenant dev --priority 3 --json"
    ))

    assert payload["task"]["status"] == "triage"
    assert payload["task"]["assignee"] == "orchestrator"
    assert payload["task"]["session_id"] == "sess-goal-1"
    assert payload["task"]["tenant"] == "dev"
    assert payload["task"]["priority"] == 3
    assert payload["decompose"] is None
    assert payload["child_ids"] == []

    payload2 = json.loads(kc.run_slash(
        "goal 'refactor the worker lane bridge' "
        "--session sess-goal-1 --assignee orchestrator --json"
    ))
    assert payload2["task_id"] == payload["task_id"]


def test_run_slash_goal_can_decompose_to_worker_lane(kanban_home, monkeypatch):
    from unittest.mock import MagicMock
    from hermes_cli.worker_lanes import WorkerLane, clear_worker_lanes, register_worker_lane

    clear_worker_lanes()
    register_worker_lane(WorkerLane(
        name="codex-deep",
        kind="codex_cli",
        description="Codex CLI lane for implementation work",
        spawn_fn=lambda *args, **kwargs: None,
        source="test",
    ))

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps({
        "fanout": True,
        "rationale": "implementation can go to codex",
        "tasks": [
            {
                "title": "Implement worker lane bridge",
                "body": "Change code and provide evidence.",
                "assignee": "codex-deep",
                "parents": [],
            }
        ],
    })
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(return_value=resp)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda *a, **kw: (fake_client, "test-model"),
    )
    monkeypatch.setattr(
        "agent.auxiliary_client.get_auxiliary_extra_body",
        lambda *a, **kw: {},
    )
    monkeypatch.setattr(
        "hermes_cli.kanban_decompose._load_config",
        lambda: {"kanban": {"orchestrator_profile": "orchestrator", "default_assignee": "fallback"}},
    )
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda: [])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "orchestrator")
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "orchestrator")

    payload = json.loads(kc.run_slash(
        "goal 'ship codex worker lane orchestration' "
        "--assignee orchestrator --decompose --json"
    ))

    assert payload["task"]["status"] == "todo"
    assert payload["decompose"]["ok"] is True
    assert payload["decompose"]["fanout"] is True
    assert len(payload["child_ids"]) == 1
    with kb.connect() as conn:
        child = kb.get_task(conn, payload["child_ids"][0])
    assert child.assignee == "codex-deep"
    assert child.status == "ready"


# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------

def test_run_slash_specify_end_to_end(kanban_home, monkeypatch):
    """The /kanban specify slash command routes through run_slash, which
    both the interactive CLI and every gateway platform use. This test
    covers both surfaces."""
    from unittest.mock import MagicMock

    # Create a triage task via the same slash surface.
    create_out = kc.run_slash("create 'rough idea' --triage")
    import re
    m = re.search(r"(t_[a-f0-9]+)", create_out)
    assert m, f"no task id in: {create_out!r}"
    tid = m.group(1)

    # Mock the auxiliary client so we don't hit a real provider.
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = (
        '{"title": "Spec: rough idea", "body": "**Goal**\\nShip it."}'
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(return_value=resp)
    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda *a, **kw: (fake_client, "test-model"),
    )

    # Specify via slash.
    out = kc.run_slash(f"specify {tid}")
    assert "Specified" in out
    assert tid in out

    # Task is promoted and retitled.
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status in {"todo", "ready"}
    assert task.title == "Spec: rough idea"


def test_run_slash_specify_help_is_reachable(kanban_home):
    """`-h`/`--help` on a subcommand returns the actual help text — see
    issue #21794. argparse writes help to stdout and exits 0; run_slash
    must capture both streams and treat exit 0 as success, not error."""
    out = kc.run_slash("specify --help")
    assert "specify" in out.lower()
    # Help dump should NOT come back wrapped as a usage error.
    assert not out.startswith("⚠")


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------

def test_run_slash_bare_returns_curated_help(kanban_home):
    """Bare `/kanban` returns the curated short-help block — not a 5KB
    argparse usage dump."""
    out = kc.run_slash("")
    assert "/kanban" in out
    assert "list" in out
    assert "show" in out
    # Sanity: should be a chat-friendly size, not the raw usage tree.
    assert len(out) < 2000
    # Shouldn't surface argparse's usage-error sentinel.
    assert "usage error" not in out.lower()


@pytest.mark.parametrize("alias", ["help", "--help", "-h", "?"])
def test_run_slash_help_aliases_match_bare(kanban_home, alias):
    """Every documented help alias produces the same curated output."""
    bare = kc.run_slash("")
    out = kc.run_slash(alias)
    assert out == bare


def test_run_slash_subcommand_help_returns_help_text(kanban_home):
    """`/kanban show -h` returns the actual subcommand help, not a
    fake `(usage error: 0)` sentinel."""
    out = kc.run_slash("show -h")
    assert "task_id" in out
    assert "/kanban show" in out
    assert not out.startswith("⚠")


def test_run_slash_unknown_action_friendly_error(kanban_home):
    """Unknown subcommand surfaces a single-line usage error prefixed
    with our marker — no `(usage error: 2)` wrapping, no doubled
    `kanban kanban` prog string."""
    out = kc.run_slash("frobnicate")
    assert "/kanban" in out
    assert "frobnicate" in out
    assert "/kanban-wrap" not in out
    assert "/kanban kanban" not in out
    assert "(usage error: " not in out


def test_run_slash_missing_required_arg_friendly_error(kanban_home):
    """Missing positional argument shows the subcommand-scoped usage
    line, not the top-level kanban tree."""
    out = kc.run_slash("show")
    assert "/kanban show" in out
    assert "task_id" in out


def test_run_slash_board_override_restores_prior_env(kanban_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "beta")

    kc.run_slash("--board alpha list")

    assert os.environ.get("HERMES_KANBAN_BOARD") == "beta"


def test_run_slash_board_override_does_not_change_boards_show_current(kanban_home):
    kb.create_board("alpha")
    kb.create_board("beta")
    kb.set_current_board("alpha")

    out = kc.run_slash("--board beta boards show")

    assert "Current board: alpha" in out
