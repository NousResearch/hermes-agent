"""Tests for the kanban `promote` verb (issue #28822).

The realistic bug scenario from #28822 is: a child task ends up in
``todo`` with all its parents already ``done`` (because the
auto-promote daemon hasn't run, or a manual close raced it).
Direct-SQL setup is used to construct that state deterministically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _stuck_todo(conn, *, parents_done=True, n_parents=1):
    """Build the #28822 scenario: child in 'todo' whose parents may
    have closed as 'done' without the auto-promote logic firing.
    """
    parent_ids = [
        kb.create_task(conn, title=f"parent{i}", assignee="setup")
        for i in range(n_parents)
    ]
    child_id = kb.create_task(
        conn, title="child", parents=parent_ids, assignee="setup"
    )
    assert kb.get_task(conn, child_id).status == "todo"
    if parents_done:
        for pid in parent_ids:
            conn.execute(
                "UPDATE tasks SET status='done' WHERE id=?", (pid,)
            )
    return child_id, parent_ids


def test_promote_stuck_todo_succeeds(conn):
    child, _ = _stuck_todo(conn, parents_done=True)
    ok, err = kb.promote_task(conn, child, actor="tester")
    assert ok and err is None
    assert kb.get_task(conn, child).status == "ready"


def test_promote_refuses_when_parent_not_done(conn):
    child, parents = _stuck_todo(conn, parents_done=False)
    ok, err = kb.promote_task(conn, child, actor="tester")
    assert ok is False
    assert err is not None and "unsatisfied parent dependencies" in err
    assert parents[0] in err
    assert kb.get_task(conn, child).status == "todo"


def test_promote_with_force_bypasses_dependency_check(conn):
    child, _ = _stuck_todo(conn, parents_done=False)
    ok, err = kb.promote_task(
        conn, child, actor="tester", reason="recovery", force=True
    )
    assert ok and err is None
    assert kb.get_task(conn, child).status == "ready"


def test_promote_emits_audit_event(conn):
    child, _ = _stuck_todo(conn, parents_done=True)
    kb.promote_task(conn, child, actor="tester", reason="manual recovery")
    ev = conn.execute(
        "SELECT kind, payload FROM task_events "
        "WHERE task_id = ? AND kind = 'promoted_manual'",
        (child,),
    ).fetchone()
    assert ev is not None
    payload = json.loads(ev["payload"])
    assert payload["actor"] == "tester"
    assert payload["reason"] == "manual recovery"
    assert payload["forced"] is False


def test_promote_force_records_forced_flag(conn):
    child, _ = _stuck_todo(conn, parents_done=False)
    kb.promote_task(conn, child, actor="tester", force=True, reason="r")
    ev = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'promoted_manual'",
        (child,),
    ).fetchone()
    assert json.loads(ev["payload"])["forced"] is True


def test_promote_does_not_change_assignee(conn):
    child, _ = _stuck_todo(conn, parents_done=True)
    before = kb.get_task(conn, child).assignee
    kb.promote_task(conn, child, actor="someone_else")
    after = kb.get_task(conn, child).assignee
    assert before == after


def test_promote_dry_run_does_not_mutate(conn):
    child, _ = _stuck_todo(conn, parents_done=True)
    ok, err = kb.promote_task(conn, child, actor="tester", dry_run=True)
    assert ok and err is None
    assert kb.get_task(conn, child).status == "todo"
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM task_events "
        "WHERE task_id = ? AND kind = 'promoted_manual'",
        (child,),
    ).fetchone()["n"]
    assert n == 0


def test_promote_dry_run_reports_dependency_failure(conn):
    child, _ = _stuck_todo(conn, parents_done=False)
    ok, err = kb.promote_task(conn, child, actor="tester", dry_run=True)
    assert ok is False
    assert err is not None and "unsatisfied" in err


def test_promote_rejects_non_todo_status(conn):
    tid = kb.create_task(conn, title="standalone")
    assert kb.get_task(conn, tid).status == "ready"
    ok, err = kb.promote_task(conn, tid, actor="tester")
    assert ok is False
    assert "'ready'" in err and "promote only applies" in err


def test_promote_rejects_unknown_task(conn):
    ok, err = kb.promote_task(conn, "t_doesnotexist", actor="tester")
    assert ok is False
    assert err is not None and "not found" in err


def test_promote_blocked_task_works(conn):
    tid = kb.create_task(conn, title="t")
    conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))
    ok, err = kb.promote_task(
        conn, tid, actor="tester", reason="ready now"
    )
    assert ok and err is None
    assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# CLI `_cmd_promote` — bulk via `--ids` (the issue's anti-respawn use case:
# promote all children of a closed parent in one command).
# ---------------------------------------------------------------------------


def _promote_ns(task_id, *, ids=None, reason=None, force=False,
                dry_run=False, as_json=False, from_triage=False):
    return argparse.Namespace(
        task_id=task_id,
        reason=list(reason or []),
        ids=list(ids or []) or None,
        force=force,
        from_triage=from_triage,
        dry_run=dry_run,
        json=as_json,
    )


def _triage_task(
    conn,
    *,
    parent_status="done",
    body="already specified",
    proof_kind="block_loop_detected",
):
    parent = kb.create_task(conn, title="parent", assignee="setup")
    task = kb.create_task(
        conn,
        title="triaged",
        body=body,
        parents=[parent],
        assignee="owner",
        workspace_kind="dir",
        workspace_path="/tmp/hermes-triage-recovery",
        triage=True,
    )
    if parent_status is not None:
        conn.execute(
            "UPDATE tasks SET status=? WHERE id=?", (parent_status, parent)
        )
    conn.execute(
        "UPDATE tasks SET block_kind='transient', block_recurrences=3, "
        "consecutive_failures=4, last_failure_error='kept' WHERE id=?",
        (task,),
    )
    if proof_kind:
        kb._append_event(conn, task, proof_kind, None)
    return task, parent


def test_triage_promote_requires_explicit_flag(conn):
    task, _ = _triage_task(conn)
    ok, err = kb.promote_task(conn, task, actor="tester", reason="audited")
    assert ok is False and "--from-triage" in err
    assert kb.get_task(conn, task).status == "triage"


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_triage_promote_requires_nonempty_reason(conn, reason):
    task, _ = _triage_task(conn)
    ok, err = kb.promote_task(
        conn, task, actor="tester", reason=reason, from_triage=True
    )
    assert ok is False and "audit reason" in err
    assert kb.get_task(conn, task).status == "triage"


def test_triage_promote_rejects_force(conn):
    task, _ = _triage_task(conn)
    ok, err = kb.promote_task(
        conn,
        task,
        actor="tester",
        reason="audited",
        force=True,
        from_triage=True,
    )
    assert ok is False and "cannot be combined" in err
    assert kb.get_task(conn, task).status == "triage"


def test_triage_promote_enforces_parent_gate(conn):
    task, parent = _triage_task(conn, parent_status=None)
    ok, err = kb.promote_task(
        conn, task, actor="tester", reason="audited", from_triage=True
    )
    assert ok is False and parent in err
    assert kb.get_task(conn, task).status == "triage"


@pytest.mark.parametrize("status", ["todo", "blocked"])
def test_from_triage_rejects_non_triage_status(conn, status):
    task, _ = _triage_task(conn)
    conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task))
    ok, err = kb.promote_task(
        conn, task, actor="tester", reason="audited", from_triage=True
    )
    assert ok is False and "status 'triage'" in err
    assert kb.get_task(conn, task).status == status


def test_triage_promote_rejects_fresh_bodyless_card(conn):
    task, _ = _triage_task(conn, body=None, proof_kind=None)
    ok, err = kb.promote_task(
        conn, task, actor="tester", reason="audited", from_triage=True
    )
    assert ok is False and "durable" in err
    assert kb.get_task(conn, task).status == "triage"


def test_triage_promote_requires_event_proof_not_just_body(conn):
    task, _ = _triage_task(conn, proof_kind=None)
    ok, err = kb.promote_task(
        conn, task, actor="tester", reason="audited", from_triage=True
    )
    assert ok is False and "durable" in err
    assert kb.get_task(conn, task).status == "triage"


@pytest.mark.parametrize(
    "ownership",
    [
        "task_claim_lock",
        "task_claim_expires",
        "task_worker_pid",
        "task_current_run_id",
        "run_open",
        "run_running",
        "run_claim_lock",
        "run_claim_expires",
        "run_worker_pid",
    ],
)
def test_triage_promote_rejects_any_runtime_ownership(conn, ownership):
    task, _ = _triage_task(conn)
    if ownership.startswith("task_") and ownership != "task_current_run_id":
        field = ownership.removeprefix("task_")
        value = {
            "claim_lock": "task-lease",
            "claim_expires": 2000000000,
            "worker_pid": 12345,
        }[field]
        conn.execute(f"UPDATE tasks SET {field}=? WHERE id=?", (value, task))
    else:
        run = {
            "status": "reclaimed",
            "claim_lock": None,
            "claim_expires": None,
            "worker_pid": None,
            "ended_at": 2,
        }
        if ownership != "task_current_run_id":
            field = ownership.removeprefix("run_")
            if field == "open":
                run["ended_at"] = None
            elif field == "running":
                run["status"] = "running"
            else:
                run[field] = {
                    "claim_lock": "run-lease",
                    "claim_expires": 2000000000,
                    "worker_pid": 12345,
                }[field]
        run_id = conn.execute(
            "INSERT INTO task_runs "
            "(task_id, profile, status, claim_lock, claim_expires, worker_pid, "
            "started_at, ended_at) VALUES (?, 'owner', ?, ?, ?, ?, 1, ?)",
            (
                task,
                run["status"],
                run["claim_lock"],
                run["claim_expires"],
                run["worker_pid"],
                run["ended_at"],
            ),
        ).lastrowid
        if ownership == "task_current_run_id":
            conn.execute(
                "UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, task)
            )
    task_before = dict(conn.execute(
        "SELECT * FROM tasks WHERE id=?", (task,)
    ).fetchone())
    runs_before = [dict(row) for row in conn.execute(
        "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task,)
    ).fetchall()]

    ok, err = kb.promote_task(
        conn,
        task,
        actor="tester",
        reason="operator reviewed recurrence",
        from_triage=True,
    )

    assert ok is False and "reclaim" in err
    assert dict(conn.execute(
        "SELECT * FROM tasks WHERE id=?", (task,)
    ).fetchone()) == task_before
    assert [dict(row) for row in conn.execute(
        "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task,)
    ).fetchall()] == runs_before
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM task_events WHERE task_id=? "
        "AND kind IN ('triage_recovered_manual', 'reclaimed')",
        (task,),
    ).fetchone()["n"] == 0


@pytest.mark.parametrize("parent_status", ["done", "archived"])
@pytest.mark.parametrize("proof_kind", ["specified", "block_loop_detected"])
def test_triage_promote_is_immediately_claimable(
    conn, parent_status, proof_kind
):
    task, _ = _triage_task(
        conn, parent_status=parent_status, proof_kind=proof_kind
    )
    conn.execute(
        "INSERT INTO task_runs "
        "(task_id, profile, status, last_heartbeat_at, started_at, ended_at, "
        "outcome) VALUES (?, 'owner', 'reclaimed', 2, 1, 3, 'reclaimed')",
        (task,),
    )
    before = kb.get_task(conn, task)

    ok, err = kb.promote_task(
        conn,
        task,
        actor="tester",
        reason="operator reviewed recurrence",
        from_triage=True,
    )

    assert ok and err is None
    after = kb.get_task(conn, task)
    assert after.status == "ready"
    preserved = (
        "title", "body", "assignee", "workspace_kind", "workspace_path",
        "block_kind", "block_recurrences", "consecutive_failures",
        "last_failure_error",
    )
    assert {name: getattr(after, name) for name in preserved} == {
        name: getattr(before, name) for name in preserved
    }
    assert (after.claim_lock, after.claim_expires, after.worker_pid) == (
        None, None, None
    )
    assert after.current_run_id is None

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='triage_recovered_manual'",
        (task,),
    ).fetchone()
    assert event is not None
    assert json.loads(event["payload"]) == {
        "actor": "tester",
        "reason": "operator reviewed recurrence",
        "prior_status": "triage",
        "parent_gate": "satisfied",
        "block_kind": "transient",
        "block_recurrences": 3,
        "consecutive_failures": 4,
    }
    claimed = kb.claim_task(conn, task, claimer="recovery-test")
    assert claimed is not None
    assert claimed.status == "running"
    assert claimed.claim_lock == "recovery-test"
    assert claimed.current_run_id is not None


def test_triage_promote_dry_run_has_no_mutation_or_event(conn):
    task, _ = _triage_task(conn)
    ok, err = kb.promote_task(
        conn,
        task,
        actor="tester",
        reason="audited",
        from_triage=True,
        dry_run=True,
    )
    assert ok and err is None
    assert kb.get_task(conn, task).status == "triage"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM task_events WHERE task_id=? "
        "AND kind='triage_recovered_manual'",
        (task,),
    ).fetchone()["n"] == 0


def test_cli_parser_accepts_from_triage_before_task():
    parser = argparse.ArgumentParser()
    kb_cli.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "kanban", "promote", "--from-triage", "t_example", "operator", "reviewed",
    ])
    assert args.from_triage is True
    assert args.task_id == "t_example"
    assert args.reason == ["operator", "reviewed"]


def test_cli_triage_promote_json(kanban_home, capsys):
    with kb.connect() as conn:
        task, _ = _triage_task(conn)
    rc = kb_cli._cmd_promote(_promote_ns(
        task,
        reason=["operator", "reviewed"],
        from_triage=True,
        as_json=True,
    ))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_id"] == task
    assert payload["from_triage"] is True
    assert payload["promoted"] is True


def test_cli_promote_bulk_ids_promotes_all(kanban_home, capsys):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        children = [
            kb.create_task(conn, title=f"c{i}", parents=[parent])
            for i in range(3)
        ]
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    rc = kb_cli._cmd_promote(_promote_ns(children[0], ids=children[1:]))
    assert rc == 0
    out = capsys.readouterr().out
    for c in children:
        assert c in out
    with kb.connect() as conn:
        for c in children:
            assert kb.get_task(conn, c).status == "ready"


def test_cli_promote_bulk_partial_failure_exits_1(kanban_home, capsys):
    """Bulk with one bad id: good ones still promote, exit code reflects failure."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        good = kb.create_task(conn, title="good", parents=[parent])
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    rc = kb_cli._cmd_promote(_promote_ns(good, ids=["t_nope"]))
    assert rc == 1
    captured = capsys.readouterr()
    assert good in captured.out  # good one promoted
    assert "t_nope" in captured.err and "not found" in captured.err
    with kb.connect() as conn:
        assert kb.get_task(conn, good).status == "ready"


def test_cli_promote_bulk_json_emits_list(kanban_home, capsys):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        a = kb.create_task(conn, title="a", parents=[parent])
        b = kb.create_task(conn, title="b", parents=[parent])
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    rc = kb_cli._cmd_promote(_promote_ns(a, ids=[b], as_json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 2
    assert {r["task_id"] for r in payload} == {a, b}
    assert all(r["promoted"] for r in payload)


def test_cli_promote_single_json_stays_flat_object(kanban_home, capsys):
    """Back-compat: single-id JSON is still a flat object, not a list."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="c", parents=[parent])
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    rc = kb_cli._cmd_promote(_promote_ns(child, as_json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)
    assert payload["task_id"] == child and payload["promoted"] is True


def test_cli_promote_dedupes_duplicate_ids(kanban_home, capsys):
    """Same id in positional + --ids must only attempt the promotion once."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="c", parents=[parent])
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    rc = kb_cli._cmd_promote(_promote_ns(child, ids=[child, child]))
    assert rc == 0
    with kb.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE task_id = ? AND kind = 'promoted_manual'",
            (child,),
        ).fetchone()["n"]
    assert n == 1
