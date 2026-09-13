"""``hermes kanban complete`` must name unsatisfied parents (#110315).

The dependency invariant is unchanged: a child with unfinished parents is
still refused. What changes is the report — the refusal reason is captured
at the authoritative check inside ``complete_task`` (opt-in exception) so
the CLI stops claiming ``unknown id or terminal state`` for a task that
exists and is merely dependency-gated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _blocked_parent_with_child(conn):
    parent_id = kb.create_task(conn, title="awaiting approval", assignee="planner")
    # A parent parked as an operator-approval gate (the #110315 scenario);
    # block_task only accepts running/ready sources, so flip directly.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (parent_id,))
    child_id = kb.create_task(conn, title="source work", assignee="builder", parents=[parent_id])
    return parent_id, child_id


# ---------------------------------------------------------------------------
# CLI surface: the refusal names its blockers
# ---------------------------------------------------------------------------

def test_complete_names_unsatisfied_parent_and_status(kanban_home):
    with kbc.connect_closing() as conn:
        parent_id, child_id = _blocked_parent_with_child(conn)

    out = kc.run_slash(f"complete {child_id} --result 'delivered'")

    assert f"cannot complete {child_id}: unsatisfied parent dependencies:" in out
    assert f"{parent_id} (blocked)" in out
    assert "unknown id or terminal state" not in out
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, child_id).status != "done"


def test_complete_reports_every_unfinished_parent_deterministically(kanban_home):
    with kbc.connect_closing() as conn:
        p1 = kb.create_task(conn, title="first", assignee="planner")
        p2 = kb.create_task(conn, title="second", assignee="planner")
        # Park p1 as an operator-approval gate (block_task only accepts
        # running/ready sources; a direct flip is the shortest setup).
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (p1,))
        child_id = kb.create_task(conn, title="child", assignee="builder", parents=[p2, p1])

    out = kc.run_slash(f"complete {child_id}")

    blockers = ", ".join(
        f"{pid} ({status})" for pid, status in sorted([(p1, "blocked"), (p2, "ready")])
    )  # id order (dictionary), independent of the link order
    assert f"unsatisfied parent dependencies: {blockers}" in out


def test_missing_and_terminal_refusals_keep_the_generic_message(kanban_home):
    # Unknown id: still the generic refusal, never the parents message.
    out = kc.run_slash("complete t_deadbeef00")
    assert "unsatisfied parent dependencies" not in out
    assert "unknown id or terminal state" in out

    # Terminal state (already done): also stays generic.
    with kbc.connect_closing() as conn:
        done_id = kb.create_task(conn, title="finished", assignee="builder")
        assert kb.complete_task(conn, done_id)
    out = kc.run_slash(f"complete {done_id}")
    assert "unsatisfied parent dependencies" not in out
    assert "unknown id or terminal state" in out


def test_complete_succeeds_unchanged_once_parents_settle(kanban_home):
    with kbc.connect_closing() as conn:
        parent_id, child_id = _blocked_parent_with_child(conn)
        assert kb.complete_task(conn, parent_id)

    out = kc.run_slash(f"complete {child_id} --result 'source work delivered'")

    assert f"Completed {child_id}" in out
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, child_id).status == "done"


# ---------------------------------------------------------------------------
# Domain layer: opt-in exception keeps the boolean API intact
# ---------------------------------------------------------------------------

def test_complete_task_boolean_contract_unchanged_by_default(tmp_path):
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        parent_id, child_id = _blocked_parent_with_child(conn)

        assert kb.complete_task(conn, child_id, result="x") is False
        assert kb.get_task(conn, child_id).status != "done"

        # Opt-in carries the blockers captured at the refusal point.
        with pytest.raises(kb.ParentsNotSatisfiedError) as excinfo:
            kb.complete_task(conn, child_id, result="x", raise_parents_refusal=True)
        assert excinfo.value.blockers == [(parent_id, "blocked")]
        assert excinfo.value.completing_task_id == child_id
        assert kb.get_task(conn, child_id).status != "done"

        # Settling the parent flips both forms back to success.
        assert kb.complete_task(conn, parent_id)
        assert kb.complete_task(conn, child_id, result="x") is True
    finally:
        conn.close()


def test_transactional_refusal_stays_actionable_after_pre_check_race(tmp_path, monkeypatch):
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        parent_id, child_id = _blocked_parent_with_child(conn)
        real = kb._parents_satisfied
        # The cheap pre-check passes (parent "just completed"), but the parent
        # reopens before the write txn: the hard invariant must still refuse —
        # from inside the txn, with the blocker captured there.
        calls = {"n": 0}

        def flaky_precheck(c, tid):
            calls["n"] += 1
            if calls["n"] == 1:
                return True
            return real(c, tid)

        monkeypatch.setattr(kb, "_parents_satisfied", flaky_precheck)
        with pytest.raises(kb.ParentsNotSatisfiedError) as excinfo:
            kb.complete_task(conn, child_id, result="x", raise_parents_refusal=True)
        assert excinfo.value.blockers == [(parent_id, "blocked")]
        assert kb.get_task(conn, child_id).status != "done"
    finally:
        conn.close()
