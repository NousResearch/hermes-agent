"""Tests for the kanban `promote` verb (issue #28822).

The realistic bug scenario from #28822 is: a child task ends up in
``todo`` with all its parents already ``done`` (because the
auto-promote daemon hasn't run, or a manual close raced it).
Direct-SQL setup is used to construct that state deterministically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


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
    with kbc.connect() as c:
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


def test_promote_refuses_undone_parent_and_names_the_real_remedy(conn):
    # #106195: promotion must never report a 'ready' that the first claim reverts.
    child, (parent,) = _stuck_todo(conn, parents_done=False)
    ok, err = kb.promote_task(conn, child, actor="tester", reason="recovery")
    assert not ok
    assert err and parent in err
    assert kb.get_task(conn, child).status == "todo"
    assert kb.claim_task(conn, child) is None  # still gated; nothing pretended


# ---------------------------------------------------------------------------
# `triage` exit — a card the unblock-loop breaker parked there is otherwise
# reachable only through `specify`, which replaces the body with an LLM rewrite.
# ---------------------------------------------------------------------------


def test_promote_triage_lands_ready_and_keeps_the_spec(conn):
    tid = kb.create_task(
        conn, title="parked", body="the real spec, do not rewrite this",
        assignee="worker", triage=True,
    )
    before = kb.get_task(conn, tid)
    assert before.status == "triage"
    ok, err = kb.promote_task(conn, tid, actor="operator", reason="cause fixed")
    assert ok and err is None
    after = kb.get_task(conn, tid)
    assert after.status == "ready"
    assert (after.title, after.body) == (before.title, before.body)


def test_promote_triage_with_an_open_parent_lands_todo_and_stays_gated(conn):
    parent = kb.create_task(conn, title="parent", assignee="setup")
    tid = kb.create_task(
        conn, title="parked child", parents=[parent], assignee="worker", triage=True,
    )
    ok, err = kb.promote_task(conn, tid, actor="operator")
    assert ok and err is None
    assert kb.get_task(conn, tid).status == "todo"  # released, not spawned
    assert kb.claim_task(conn, tid) is None
    conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    kb.recompute_ready(conn)
    assert kb.get_task(conn, tid).status == "ready"


def test_promote_triage_dry_run_only_validates(conn):
    tid = kb.create_task(conn, title="parked", assignee="worker", triage=True)
    ok, err = kb.promote_task(conn, tid, actor="operator", dry_run=True)
    assert ok and err is None
    assert kb.get_task(conn, tid).status == "triage"


@pytest.mark.parametrize("status", ["ready", "running", "review", "done"])
def test_promote_refuses_a_card_outside_the_promotable_sources(conn, status):
    # The exit is scoped: a card that is NOT parked in triage (and is not sitting
    # in todo/blocked) is refused, and the error names the promotable set.
    tid = kb.create_task(conn, title="in flight", assignee="worker")
    conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
    ok, err = kb.promote_task(conn, tid, actor="operator")
    assert not ok
    assert err and status in err and "triage" in err
    assert kb.get_task(conn, tid).status == status


def test_promote_triage_appends_the_audit_event_tail_shows(conn):
    tid = kb.create_task(conn, title="parked", assignee="worker", triage=True)
    assert kb.promote_task(conn, tid, actor="operator", reason="gate fixed")[0]
    # `hermes kanban tail` renders kb.list_events, so this is what it shows.
    events = [e for e in kb.list_events(conn, tid) if e.kind == "promoted_manual"]
    assert len(events) == 1
    payload = events[0].payload
    assert payload["from"] == "triage"
    assert payload["status"] == "ready"
    assert payload["actor"] == "operator" and payload["reason"] == "gate fixed"


def test_promote_work_phase_event_payload_is_unchanged(conn):
    # Consumers of the todo/blocked path (dashboard, `watch`) keep the old shape.
    child, _ = _stuck_todo(conn, parents_done=True)
    assert kb.promote_task(conn, child, actor="tester")[0]
    payload = [e for e in kb.list_events(conn, child)
               if e.kind == "promoted_manual"][0].payload
    assert payload["actor"] == "tester" and "from" not in payload


# ---------------------------------------------------------------------------
# CLI `_cmd_promote` — bulk via `--ids` (the issue's anti-respawn use case:
# promote all children of a closed parent in one command).
# ---------------------------------------------------------------------------


def _promote_ns(task_id, *, ids=None, reason=None, force=False,
                dry_run=False, as_json=False):
    return argparse.Namespace(
        task_id=task_id,
        reason=list(reason or []),
        ids=list(ids or []) or None,
        force=force,
        dry_run=dry_run,
        json=as_json,
    )


def test_cli_promote_bulk_ids_promotes_all(kanban_home, capsys):
    with kbc.connect() as conn:
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
    with kbc.connect() as conn:
        for c in children:
            assert kb.get_task(conn, c).status == "ready"


def test_cli_promote_reports_the_status_a_triage_card_reached(kanban_home, capsys):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="parked", assignee="worker", triage=True)
    rc = kb_cli._cmd_promote(_promote_ns(tid))
    assert rc == 0
    assert f"Promoted {tid} -> ready" in capsys.readouterr().out
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"


def test_cli_promote_dry_run_names_the_parent_gated_triage_target(kanban_home, capsys):
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="setup")
        tid = kb.create_task(
            conn, title="parked child", parents=[parent], assignee="worker", triage=True,
        )
    rc = kb_cli._cmd_promote(_promote_ns(tid, dry_run=True))
    assert rc == 0
    assert f"Would promote {tid} -> todo/ready (dry)" in capsys.readouterr().out
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


