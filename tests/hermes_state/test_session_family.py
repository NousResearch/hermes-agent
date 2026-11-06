"""`hermes_state_family`: the read-only session-family walk behind `/tree`.

One family per store: a root, two branches, a compression chain, a subagent child and a row with
no marker at all. The walk must classify each from the stored markers only, never write, and stay
inside one profile's store.
"""

from __future__ import annotations


import pytest

from hermes_state import SessionDB
from hermes_state_family import (
    KIND_BRANCH, KIND_COMPRESSION, KIND_ROOT, KIND_SUBAGENT, KIND_UNKNOWN, build_session_family,
    classify_kind, visible_nodes,
)


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _family(db: SessionDB) -> SessionDB:
    """Root → compression continuation → branch, plus a second branch and a subagent child."""
    db.create_session("root", source="cli")
    db.append_message("root", "user", "one")
    db.append_message("root", "assistant", "two")
    db.end_session("root", "compression")
    db.create_session("cont", source="cli", parent_session_id="root")
    db.append_message("cont", "user", "after compaction")
    db.create_session("branch-a", source="cli", parent_session_id="cont",
                      model_config={"_branched_from": "cont"})
    db.create_session("branch-b", source="cli", parent_session_id="root",
                      model_config={"_branched_from": "root"})
    db.create_session("sub", source="tool", parent_session_id="branch-a",
                      model_config={"_delegate_from": "branch-a"})
    return db


def test_walks_up_to_the_root_then_every_descendant(db: SessionDB) -> None:
    _family(db)

    family = build_session_family(db, "branch-a")

    assert family.root_id == "root"
    assert family.truncated is False
    assert [node["session_id"] for node in family.nodes] == [
        "root", "cont", "branch-a", "sub", "branch-b"]
    assert [node["depth"] for node in family.nodes] == [0, 1, 2, 3, 1]


def test_kinds_come_from_the_stored_markers(db: SessionDB) -> None:
    _family(db)

    kinds = {node["session_id"]: node["kind"] for node in build_session_family(db, "branch-a").nodes}

    assert kinds == {"root": KIND_ROOT, "cont": KIND_COMPRESSION, "branch-a": KIND_BRANCH,
                     "branch-b": KIND_BRANCH, "sub": KIND_SUBAGENT}


def test_an_unmarked_child_of_an_ordinary_parent_is_unknown(db: SessionDB) -> None:
    db.create_session("root", source="cli")
    db.append_message("root", "user", "hello")
    db.create_session("child", source="cli", parent_session_id="root")

    kinds = {node["session_id"]: node["kind"] for node in build_session_family(db, "child").nodes}

    assert kinds == {"root": KIND_ROOT, "child": KIND_UNKNOWN}


def test_a_branch_marker_that_names_another_parent_does_not_count(db: SessionDB) -> None:
    """The marker must point at THIS row's parent (compression copies model_config forward)."""
    db.create_session("root", source="cli")
    db.create_session("other", source="cli")
    db.create_session("child", source="cli", parent_session_id="root",
                      model_config={"_branched_from": "other"})

    kinds = {node["session_id"]: node["kind"] for node in build_session_family(db, "child").nodes}

    assert kinds["child"] == KIND_UNKNOWN


def test_only_a_compression_continuation_knows_its_branch_point(db: SessionDB) -> None:
    _family(db)

    nodes = {node["session_id"]: node for node in build_session_family(db, "branch-a").nodes}

    assert nodes["cont"]["branch_from"] == 2  # the sealed root ended with two messages
    assert nodes["branch-a"]["branch_from"] is None  # a branch's copy point is not persisted


def test_the_current_session_is_marked(db: SessionDB) -> None:
    _family(db)

    current = [node["session_id"] for node in build_session_family(db, "cont").nodes
               if node["is_current"]]

    assert current == ["cont"]


def test_a_session_absent_from_this_store_yields_no_family(db: SessionDB) -> None:
    _family(db)

    family = build_session_family(db, "not-in-this-profile")

    assert family.nodes == []
    assert family.root_id == "not-in-this-profile"


def test_another_profiles_store_never_appears(tmp_path) -> None:
    """Profile isolation is the store: a session in another profile's state.db is not reachable."""
    shared = _family(SessionDB(db_path=tmp_path / "one" / "state.db"))
    other = SessionDB(db_path=tmp_path / "two" / "state.db")
    try:
        other.create_session("elsewhere", source="cli")
        family = build_session_family(shared, "branch-a")
        assert "elsewhere" not in {node["session_id"] for node in family.nodes}
        assert build_session_family(other, "branch-a").nodes == []
    finally:
        shared.close()
        other.close()


def test_building_a_family_never_writes(db: SessionDB, monkeypatch) -> None:
    _family(db)

    def _refuse(*_args, **_kwargs):
        raise AssertionError("the family walk must not write")

    monkeypatch.setattr(db, "_execute_write", _refuse)
    monkeypatch.setattr(db, "flush_for_read", _refuse)

    assert build_session_family(db, "branch-a").nodes


def test_a_cycle_in_the_parent_links_stops_instead_of_looping(db: SessionDB) -> None:
    db.create_session("a", source="cli")
    db.create_session("b", source="cli", parent_session_id="a")
    # A corrupt store: b's child points back at a, so a -> b -> a is a loop, not a tree.
    db._write_sql("UPDATE sessions SET parent_session_id = 'b' WHERE id = 'a'")

    family = build_session_family(db, "a")

    assert family.root_id in {"a", "b"}
    assert len(family.nodes) <= 2


def test_the_node_ceiling_is_reported_not_hidden(db: SessionDB) -> None:
    db.create_session("root", source="cli")
    for index in range(4):
        db.create_session(f"child-{index}", source="cli", parent_session_id="root")

    family = build_session_family(db, "root", max_nodes=3)

    assert family.truncated is True
    assert len(family.nodes) == 3


def test_kind_classification_agrees_with_the_stores_own_helpers(db: SessionDB) -> None:
    """The tree's rule and the store's rule must not drift apart."""
    _family(db)
    for session_id in ("cont", "branch-a", "branch-b", "sub"):
        row = db.get_session(session_id)
        parent = db.get_session(row["parent_session_id"])
        kind = classify_kind(db, row, parent)
        assert (kind == KIND_COMPRESSION) is db._is_compression_child_row(row)
        assert (kind in {KIND_BRANCH, KIND_SUBAGENT}) is db._is_explicit_fork_child_row(row)


def test_default_view_folds_continuations_and_hides_subagents(db: SessionDB) -> None:
    _family(db)

    nodes = visible_nodes(build_session_family(db, "branch-a").nodes)

    assert [node["session_id"] for node in nodes] == ["root", "branch-a", "branch-b"]
    # The folded continuation re-parents its branch to the root, one level down.
    assert [node["depth"] for node in nodes] == [0, 1, 1]
    assert nodes[0]["folded_segments"] == 1
    assert nodes[0]["has_children"] is True


def test_the_all_view_shows_every_row_at_its_own_depth(db: SessionDB) -> None:
    _family(db)

    nodes = visible_nodes(build_session_family(db, "branch-a").nodes,
                          show_compressions=True, show_subagents=True)

    assert [node["session_id"] for node in nodes] == [
        "root", "cont", "branch-a", "sub", "branch-b"]
    assert [node["depth"] for node in nodes] == [0, 1, 2, 3, 1]


def test_folding_a_row_hides_its_descendants_and_keeps_their_parent_visible(db: SessionDB) -> None:
    _family(db)

    nodes = visible_nodes(build_session_family(db, "branch-a").nodes, folded=["root"])

    assert [node["session_id"] for node in nodes] == ["root"]
    assert nodes[0]["has_children"] is True


def test_an_empty_store_has_no_family(tmp_path) -> None:
    empty = SessionDB(db_path=tmp_path / "empty.db")
    try:
        assert build_session_family(empty, "nothing").nodes == []
    finally:
        empty.close()
