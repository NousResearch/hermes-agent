"""Tests for SessionDB.get_delegation_tree_usage — rollup cost/tokens across
a conversation's delegate subagent tree.

Every per-conversation view (web insights, session search) reads one
``sessions`` row and stops at delegate/branch edges by design (see
``hermes_cli/web_routers/sessions.py``). Nothing summed the whole tree,
so a delegation fan-out silently undercounted real spend. See
``hermes_state.SessionDB.get_delegation_tree_usage`` docstring for the
measured real-world impact.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def test_standalone_session_sums_only_itself(db):
    db.create_session("solo", source="cli")
    db.update_token_counts("solo", input_tokens=100, output_tokens=10, absolute=True)

    totals = db.get_delegation_tree_usage("solo")

    assert totals["root_session_id"] == "solo"
    assert totals["session_ids"] == ["solo"]
    assert totals["session_count"] == 1
    assert totals["input_tokens"] == 100
    assert totals["output_tokens"] == 10


def test_sums_parent_plus_delegate_children(db):
    db.create_session("parent", source="cli")
    db.update_token_counts("parent", input_tokens=1000, output_tokens=100, absolute=True)

    for i in range(3):
        child_id = f"child-{i}"
        db.create_session(child_id, source="tool", parent_session_id="parent")
        db.update_token_counts(child_id, input_tokens=500, output_tokens=50, absolute=True)

    totals = db.get_delegation_tree_usage("parent")

    assert totals["session_count"] == 4
    assert totals["input_tokens"] == 1000 + 3 * 500
    assert totals["output_tokens"] == 100 + 3 * 50
    assert set(totals["session_ids"]) == {"parent", "child-0", "child-1", "child-2"}


def test_resolves_root_when_called_on_a_child(db):
    db.create_session("parent", source="cli")
    db.update_token_counts("parent", input_tokens=1000, absolute=True)
    db.create_session("child", source="tool", parent_session_id="parent")
    db.update_token_counts("child", input_tokens=500, absolute=True)

    totals = db.get_delegation_tree_usage("child")

    assert totals["root_session_id"] == "parent"
    assert totals["input_tokens"] == 1500
    assert totals["session_count"] == 2


def test_sums_nested_subagent_of_a_subagent(db):
    db.create_session("parent", source="cli")
    db.update_token_counts("parent", input_tokens=100, absolute=True)
    db.create_session("child", source="tool", parent_session_id="parent")
    db.update_token_counts("child", input_tokens=200, absolute=True)
    db.create_session("grandchild", source="tool", parent_session_id="child")
    db.update_token_counts("grandchild", input_tokens=300, absolute=True)

    totals = db.get_delegation_tree_usage("parent")

    assert totals["input_tokens"] == 600
    assert totals["session_count"] == 3


def test_sums_cost_and_cache_columns(db):
    db.create_session("parent", source="cli")
    db.update_token_counts(
        "parent",
        input_tokens=100,
        cache_read_tokens=5000,
        estimated_cost_usd=0.10,
        actual_cost_usd=0.09,
        absolute=True,
    )
    db.create_session("child", source="tool", parent_session_id="parent")
    db.update_token_counts(
        "child",
        input_tokens=200,
        cache_read_tokens=7000,
        estimated_cost_usd=0.20,
        actual_cost_usd=0.18,
        absolute=True,
    )

    totals = db.get_delegation_tree_usage("parent")

    assert totals["cache_read_tokens"] == 12000
    assert totals["estimated_cost_usd"] == pytest.approx(0.30)
    assert totals["actual_cost_usd"] == pytest.approx(0.27)


def test_does_not_loop_forever_on_cyclic_parent_link(db):
    # Defensive: a corrupt/self-referencing row must not hang the walk.
    db.create_session("a", source="cli")
    db.create_session("b", source="tool", parent_session_id="a")
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?", ("b", "a")
    )
    db._conn.commit()

    totals = db.get_delegation_tree_usage("a")

    assert totals["session_count"] == 2
    assert set(totals["session_ids"]) == {"a", "b"}
