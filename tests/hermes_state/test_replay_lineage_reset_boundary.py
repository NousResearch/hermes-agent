"""Regression guard: reset lineage must not replay a parent's transcript.

A ``/new`` (or auto) reset starts a separate user-visible conversation even
though the gateway retains ``parent_session_id`` for durable lineage
(``hermes_state_common._RESET_CHILD_SQL``). The backward lineage walker used
by display/resume paths (``get_resume_conversations``,
``get_ancestor_display_prefix``, ``get_messages_as_conversation(include_ancestors=True)``)
used to follow reset edges all the way to the root, prepending the reset-away
parent's messages to the child's transcript while ``sessions export`` showed
only the child's own messages. These tests pin the reset fence and confirm
compression continuations and ``get_conversation_root`` are unchanged.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _user_texts(messages):
    return [m.get("content") for m in messages if m.get("role") == "user"]


def test_reset_child_does_not_replay_parent_transcript(db):
    db.create_session("parent", "telegram", session_key="lane")
    db.append_message("parent", "user", "parent msg")
    db.end_session("parent", "session_reset")
    db.create_session(
        "child",
        "telegram",
        session_key="lane",
        parent_session_id="parent",
        model_config={"_reset_from": "parent"},
    )
    db.append_message("child", "user", "child msg")

    _, display = db.get_resume_conversations("child")
    assert _user_texts(display) == ["child msg"]
    assert db.get_ancestor_display_prefix("child") == []
    assert _user_texts(
        db.get_messages_as_conversation("child", include_ancestors=True)
    ) == ["child msg"]


def test_legacy_markerless_reset_child_does_not_replay_parent(db):
    # Pre-marker on-disk shape: no ``_reset_from``, but the child rides the
    # parent's exact routing key and the parent ended at a reset boundary.
    db.create_session("parent", "telegram", session_key="lane")
    db.append_message("parent", "user", "parent msg")
    db.end_session("parent", "daily")
    db.create_session(
        "child", "telegram", session_key="lane", parent_session_id="parent"
    )
    db.append_message("child", "user", "child msg")

    _, display = db.get_resume_conversations("child")
    assert _user_texts(display) == ["child msg"]
    assert db.get_ancestor_display_prefix("child") == []


def test_compression_child_still_replays_ancestors(db):
    db.create_session("parent", "telegram", session_key="lane")
    db.append_message("parent", "user", "parent msg")
    db.end_session("parent", "compression")
    db.create_session(
        "child", "telegram", session_key="lane", parent_session_id="parent"
    )
    db.append_message("child", "user", "child msg")

    _, display = db.get_resume_conversations("child")
    assert _user_texts(display) == ["parent msg", "child msg"]
    assert _user_texts(db.get_ancestor_display_prefix("child")) == ["parent msg"]


def test_conversation_root_still_walks_reset_boundary(db):
    # get_conversation_root is the stable billing/usage-tag id; it must keep
    # walking through reset edges (stop_at_reset=False), unlike display/resume.
    db.create_session("parent", "telegram", session_key="lane")
    db.end_session("parent", "session_reset")
    db.create_session(
        "child",
        "telegram",
        session_key="lane",
        parent_session_id="parent",
        model_config={"_reset_from": "parent"},
    )
    assert db.get_conversation_root("child") == "parent"
