import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import SESSION_SEARCH_SCHEMA, session_search


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    session_db.create_session("session", source="cli")
    return session_db


def test_reasoning_scope_finds_thought_only_token(db):
    reasoning_id = db.append_message(
        "session",
        role="assistant",
        content="Visible answer without the search term.",
        reasoning="reasoning-only-token appears in this thought",
    )
    reasoning_content_id = db.append_message(
        "session",
        role="assistant",
        content="Another visible answer without the search term.",
        reasoning_content="reasoning-only-token appears in this alternate trace",
    )

    assert db.search_messages("reasoning-only-token") == []
    assert {
        row["id"]
        for row in db.search_messages(
            "reasoning-only-token", include_reasoning=True
        )
    } == {reasoning_id, reasoning_content_id}


def test_reasoning_scope_snippet_anchors_on_reasoning(db):
    db.append_message(
        "session",
        role="assistant",
        content="Unrelated visible reply head.",
        reasoning="A distinctive reasoning fragment contains snippet-anchor-token here.",
    )

    matches = db.search_messages("snippet-anchor-token", include_reasoning=True)

    assert len(matches) == 1
    assert "distinctive reasoning fragment" in matches[0]["snippet"]
    assert "Unrelated visible reply head" not in matches[0]["snippet"]


def test_reasoning_scope_composes_with_role_filter_and_visibility(db):
    visible_id = db.append_message(
        "session",
        role="assistant",
        content="Visible reply.",
        reasoning="scope-filter-token in visible reasoning",
    )
    db.append_message(
        "session",
        role="assistant",
        content="Hidden scaffold.",
        reasoning="scope-filter-token in hidden reasoning",
        display_kind="hidden",
    )

    assistant_matches = db.search_messages(
        "scope-filter-token",
        role_filter=["assistant"],
        include_reasoning=True,
    )
    user_matches = db.search_messages(
        "scope-filter-token",
        role_filter=["user"],
        include_reasoning=True,
    )

    assert [row["id"] for row in assistant_matches] == [visible_id]
    assert user_matches == []


def test_session_search_tool_forwards_include_reasoning(db):
    message_id = db.append_message(
        "session",
        role="assistant",
        content="Visible tool result without the search term.",
        reasoning="tool-reasoning-only-token appears only in this thought",
    )

    default_result = json.loads(
        session_search(query="tool-reasoning-only-token", db=db)
    )
    reasoning_result = json.loads(
        session_search(
            query="tool-reasoning-only-token", db=db, include_reasoning=True
        )
    )

    assert "include_reasoning" in SESSION_SEARCH_SCHEMA["parameters"]["properties"]
    assert default_result["results"] == []
    assert reasoning_result["success"] is True
    assert [row["match_message_id"] for row in reasoning_result["results"]] == [
        message_id
    ]
