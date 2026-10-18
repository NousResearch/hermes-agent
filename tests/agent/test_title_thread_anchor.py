"""A messaging thread keeps one name across sessions.

A Telegram topic or Discord thread outlives its sessions: an idle reset or ``/new`` starts a fresh
session in the same thread, and that session used to be titled (and the topic renamed) after its own
opener, often a Kanban wake. The first session names the thread; later ones anchor to that name.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.title_generator import (
    THREAD_ANCHOR_SEPARATOR,
    anchor_title,
    apply_instant_title,
    auto_title_session,
    is_titleable_user_message,
    maybe_auto_title,
)
from hermes_state import SessionDB

WAKE = ("[kanban] Task t_a286d258 routed to triage after repeated blocks; orchestration attention needed.\n"
        "Title: Rave app v4\nAssignee: @hermes-agent-rtrade\nBoard: default\n\nCheck the result or decide the next step.")


def _thread_db(tmp_path, *, first_title: str | None = "OKX integration requirements review"):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("first", source="telegram", chat_id="700", chat_type="dm", thread_id="278991")
    if first_title:
        db.set_auto_title("first", first_title, source="llm")
    db.create_session("later", source="telegram", chat_id="700", chat_type="dm", thread_id="278991")
    return db


def test_kanban_wake_is_not_a_title_source():
    assert not is_titleable_user_message(WAKE)
    assert is_titleable_user_message("kanban board looks wrong, can you check?")


def test_wake_opener_leaves_a_later_session_untitled(tmp_path):
    db = _thread_db(tmp_path)
    with patch("agent.title_generator.generate_title") as model:
        maybe_auto_title(db, "later", WAKE, conversation_history=[{"role": "user", "content": WAKE}])
    model.assert_not_called()
    assert db.get_session_title("later") is None


def test_later_session_in_a_named_thread_carries_the_thread_name(tmp_path):
    db = _thread_db(tmp_path)
    seen = []
    with patch("agent.title_generator.generate_title", return_value="OKX 82000 delisting rule questions"):
        auto_title_session(db, "later", "what does OKX 82000 mean", title_callback=lambda t, s: seen.append((t, s)))
    title = db.get_session_title("later")
    assert title == f"OKX integration requirements review{THREAD_ANCHOR_SEPARATOR}OKX 82000 delisting rule questions"
    assert seen == [(title, "llm")]
    # The first session keeps its name: the anchor never renames the thread's owner.
    assert db.get_session_title("first") == "OKX integration requirements review"


def test_same_subject_keeps_the_bare_thread_name_with_a_lineage_suffix(tmp_path):
    db = _thread_db(tmp_path)
    apply_instant_title(db, "later", "OKX integration requirements review again")
    assert db.get_session_title("later") == "OKX integration requirements review #2"


def test_first_session_in_a_thread_is_titled_from_its_opener(tmp_path):
    db = _thread_db(tmp_path, first_title=None)
    with patch("agent.title_generator.generate_title", return_value="Evaluate OKX Web3 DEX RFQ repos"):
        auto_title_session(db, "first", "look at the okx web3 dex rfq repos")
    assert db.get_session_title("first") == "Evaluate OKX Web3 DEX RFQ repos"


def test_sessions_outside_a_thread_are_not_anchored(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("a", source="cli")
    db.set_auto_title("a", "Older chat", source="llm")
    db.create_session("b", source="cli")
    with patch("agent.title_generator.generate_title", return_value="Fix flaky auth test"):
        auto_title_session(db, "b", "fix the flaky auth test")
    assert db.get_session_title("b") == "Fix flaky auth test"


@pytest.mark.parametrize("subject,expected", [
    ("OKX integration requirements", "OKX integration requirements review"),
    ("", "OKX integration requirements review"),
    ("OKX integration requirements review · latency", "OKX integration requirements review · latency"),
])
def test_anchor_title_shapes(subject, expected):
    assert anchor_title("OKX integration requirements review", subject) == expected


def test_anchor_title_stays_within_the_title_cap():
    long_subject = "Latency of the Singapore edge for Asian firm orders and the retry budget " * 3
    title = anchor_title("OKX integration requirements review", long_subject)
    assert title is not None
    assert title.startswith(f"OKX integration requirements review{THREAD_ANCHOR_SEPARATOR}Latency")
    assert len(title) <= SessionDB.MAX_TITLE_LENGTH - 4
