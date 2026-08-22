"""get_messages_as_conversation(repair_alternation=True) — heal durable
alternation violations at the restore boundary.

A turn that persists a user row but no assistant row (e.g. its reply was
suppressed, or two concurrent turns interleaved their flushes) leaves a
``user;user`` pair in state.db. Without repair at restore, the defensive
pre-request ``repair_message_sequence`` re-fires on EVERY request for the
rest of the session's life, because it mutates only the per-request list.

Default (``repair_alternation=False``) never initiates repair: inspection and
export consumers read the current active transcript as stored. Rows archived
by an earlier live repair remain available through ``include_inactive=True``.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


def _seed_wedged_session(db, session_id="s1"):
    """assistant → user → user (no assistant row between): the durable wedge."""
    db.create_session(session_id, "system prompt")
    db.append_message(session_id=session_id, role="user", content="first ask")
    db.append_message(session_id=session_id, role="assistant", content="first reply")
    db.append_message(session_id=session_id, role="user", content="unanswered turn")
    db.append_message(session_id=session_id, role="user", content="next turn")
    db.append_message(session_id=session_id, role="assistant", content="next reply")




def test_repair_alternation_merges_user_pair(db):
    _seed_wedged_session(db)
    messages = db.get_messages_as_conversation("s1", repair_alternation=True)
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    # Both user texts survive, merged in order — no user input is lost.
    merged = messages[2]["content"]
    assert "unanswered turn" in merged and "next turn" in merged
    assert merged.index("unanswered turn") < merged.index("next turn")


def test_repaired_load_is_stable_under_prerequest_repair(db):
    """The restored list must yield ZERO further repairs — this is the whole
    point: the pre-request defensive repair stops firing every turn."""
    from agent.agent_runtime_helpers import repair_message_sequence

    _seed_wedged_session(db)
    messages = db.get_messages_as_conversation("s1", repair_alternation=True)
    assert repair_message_sequence(None, messages) == 0


def test_repaired_load_reconciles_durable_rows_and_fts(db):
    """A repair-enabled restore heals state.db, not only its return value."""
    from agent.agent_runtime_helpers import repair_message_sequence

    db.create_session("durable", "system prompt")
    db.append_message("durable", "user", "first ask")
    db.append_message("durable", "assistant", "first reply")
    survivor_id = db.append_message(
        "durable",
        "user",
        "unanswered turn",
        api_content="WIRE-SIDECAR: unanswered turn + injected context",
    )
    merged_away_id = db.append_message("durable", "user", "next turn")
    db.append_message(
        "durable",
        "assistant",
        "calling tool",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "demo", "arguments": "{}"},
            }
        ],
    )
    db.append_message(
        "durable", "tool", "valid result", tool_call_id="call-1"
    )
    orphan_id = db.append_message(
        "durable", "tool", "duplicate result", tool_call_id="call-1"
    )
    assistant_survivor_id = db.append_message(
        "durable",
        "assistant",
        "next reply",
        reasoning_content="first reasoning",
        api_content="WIRE-SIDECAR: next reply",
    )
    assistant_merged_away_id = db.append_message(
        "durable",
        "assistant",
        "continued reply",
        tool_calls=[
            {
                "id": "call-2",
                "type": "function",
                "function": {"name": "next_demo", "arguments": "{}"},
            }
        ],
        reasoning_content="later reasoning",
    )

    first = db.get_messages_as_conversation(
        "durable", repair_alternation=True, include_row_ids=True
    )

    assert [message["_row_id"] for message in first] == [
        row["id"]
        for row in db.get_messages("durable")
    ]
    assert first[2]["_row_id"] == survivor_id
    assert first[2]["content"] == "unanswered turn\n\nnext turn"
    assert "api_content" not in first[2]
    assert first[-1]["_row_id"] == assistant_survivor_id
    assert first[-1]["content"] == "next reply\ncontinued reply"
    assert [call["id"] for call in first[-1]["tool_calls"]] == ["call-2"]
    assert first[-1]["reasoning_content"] == "first reasoning"
    assert "api_content" not in first[-1]

    all_rows = {
        row["id"]: row
        for row in db.get_messages("durable", include_inactive=True)
    }
    assert all_rows[survivor_id]["content"] == "unanswered turn\n\nnext turn"
    assert all_rows[survivor_id]["api_content"] is None
    assert all_rows[merged_away_id]["active"] == 0
    assert all_rows[orphan_id]["active"] == 0
    assert all_rows[assistant_survivor_id]["content"] == (
        "next reply\ncontinued reply"
    )
    assert [
        call["id"] for call in all_rows[assistant_survivor_id]["tool_calls"]
    ] == ["call-2"]
    assert all_rows[assistant_survivor_id]["reasoning_content"] == (
        "first reasoning"
    )
    assert all_rows[assistant_survivor_id]["api_content"] is None
    assert all_rows[assistant_merged_away_id]["active"] == 0

    raw_reload = db.get_messages_as_conversation("durable")
    assert repair_message_sequence(None, raw_reload) == 0
    assert raw_reload == db.get_messages_as_conversation(
        "durable", repair_alternation=True
    )

    search_hits = db.search_messages("unanswered next")
    assert [hit["id"] for hit in search_hits] == [survivor_id]




# ---------------------------------------------------------------------------
# The live-replay restore SITES must pass repair_alternation=True. The initial
# fix covered gateway load_transcript + CLI startup resume; these are the other
# live-replay restore paths (ACP session resume, CLI /resume, TUI resume) that
# hand the loaded transcript to a live agent for subsequent turns.
# ---------------------------------------------------------------------------


def _seed_wedged_acp_session(db, session_id="acp1"):
    db.create_session(session_id, "acp")
    db.append_message(session_id=session_id, role="user", content="first ask")
    db.append_message(session_id=session_id, role="assistant", content="first reply")
    db.append_message(session_id=session_id, role="user", content="unanswered turn")
    db.append_message(session_id=session_id, role="user", content="next turn")
    db.append_message(session_id=session_id, role="assistant", content="next reply")


def test_acp_restore_heals_alternation_for_live_replay(db):
    """acp_adapter.SessionManager._restore feeds LIVE REPLAY: the loaded history
    becomes the resumed agent's working conversation. It must be alternation-
    clean so the pre-request repair doesn't re-fire every turn."""
    from acp_adapter.session import SessionManager

    _seed_wedged_acp_session(db, "acp1")

    class _StubAgent:
        model = "stub"

    mgr = SessionManager(agent_factory=lambda: _StubAgent(), db=db)
    state = mgr._restore("acp1")

    assert state is not None
    roles = [m["role"] for m in state.history]
    # No consecutive user turns — the durable user;user wedge was healed.
    assert roles == ["user", "assistant", "user", "assistant"], roles
    for a, b in zip(roles, roles[1:]):
        assert not (a == "user" and b == "user"), "unhealed user;user in ACP live replay"
    # No user input lost — both user texts survive, merged in order.
    merged = state.history[2]["content"]
    assert "unanswered turn" in merged and "next turn" in merged
