"""Regression coverage for explicit branch replay boundaries (#77375)."""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    database.close()


def _append_turn(db: SessionDB, session_id: str, label: str) -> None:
    db.append_message(session_id, role="user", content=f"{label} prompt")
    db.append_message(session_id, role="assistant", content=f"{label} answer")


def _contents(messages):
    return [message["content"] for message in messages]


def test_explicit_branch_resume_uses_its_fork_snapshot_not_live_parent(db):
    db.create_session("parent", source="desktop")
    _append_turn(db, "parent", "before fork")

    db.create_session(
        "branch",
        source="desktop",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    _append_turn(db, "branch", "before fork")
    _append_turn(db, "parent", "after fork")
    _append_turn(db, "branch", "branch")

    model_history, display_history = db.get_resume_conversations("branch")

    expected = [
        "before fork prompt",
        "before fork answer",
        "branch prompt",
        "branch answer",
    ]
    assert _contents(model_history) == expected
    assert _contents(display_history) == expected
    assert db.get_ancestor_display_prefix("branch") == []


def test_nested_branch_resume_excludes_parent_turns_written_after_nested_fork(db):
    db.create_session("root", source="desktop")
    _append_turn(db, "root", "root before branch")

    db.create_session(
        "branch",
        source="desktop",
        parent_session_id="root",
        model_config={"_branched_from": "root"},
    )
    _append_turn(db, "branch", "root before branch")
    _append_turn(db, "branch", "branch before nested fork")

    db.create_session(
        "nested",
        source="desktop",
        parent_session_id="branch",
        model_config={"_branched_from": "branch"},
    )
    _append_turn(db, "nested", "root before branch")
    _append_turn(db, "nested", "branch before nested fork")
    _append_turn(db, "branch", "branch after nested fork")
    _append_turn(db, "nested", "nested")

    _, display_history = db.get_resume_conversations("nested")

    assert _contents(display_history) == [
        "root before branch prompt",
        "root before branch answer",
        "branch before nested fork prompt",
        "branch before nested fork answer",
        "nested prompt",
        "nested answer",
    ]
    assert db.get_ancestor_display_prefix("nested") == []


def test_compression_continuation_still_replays_ancestors(db):
    db.create_session("root", source="desktop")
    _append_turn(db, "root", "before compression")
    db.end_session("root", "compression")

    db.create_session("continuation", source="desktop", parent_session_id="root")
    _append_turn(db, "continuation", "after compression")

    model_history, display_history = db.get_resume_conversations("continuation")

    assert _contents(model_history) == [
        "after compression prompt",
        "after compression answer",
    ]
    assert _contents(display_history) == [
        "before compression prompt",
        "before compression answer",
        "after compression prompt",
        "after compression answer",
    ]
    assert _contents(db.get_ancestor_display_prefix("continuation")) == [
        "before compression prompt",
        "before compression answer",
    ]


def test_compression_after_branch_replays_from_branch_boundary(db):
    db.create_session("root", source="desktop")
    _append_turn(db, "root", "fork snapshot")

    db.create_session(
        "branch",
        source="desktop",
        parent_session_id="root",
        model_config={"_branched_from": "root"},
    )
    _append_turn(db, "branch", "fork snapshot")
    _append_turn(db, "root", "root after fork")
    _append_turn(db, "branch", "branch before compression")
    db.end_session("branch", "compression")

    db.create_session("continuation", source="desktop", parent_session_id="branch")
    _append_turn(db, "continuation", "branch after compression")

    _, display_history = db.get_resume_conversations("continuation")

    assert _contents(display_history) == [
        "fork snapshot prompt",
        "fork snapshot answer",
        "branch before compression prompt",
        "branch before compression answer",
        "branch after compression prompt",
        "branch after compression answer",
    ]
    assert _contents(db.get_ancestor_display_prefix("continuation")) == [
        "fork snapshot prompt",
        "fork snapshot answer",
        "branch before compression prompt",
        "branch before compression answer",
    ]
