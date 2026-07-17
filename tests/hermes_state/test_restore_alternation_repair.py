"""Live restore repairs malformed assistant/tool structure without erasing
canonical user source boundaries.

Adjacent persisted ``user;user`` rows are distinct source turns. Live restore
keeps them separate, while the transient provider copy merges them with an
explicit boundary marker to satisfy strict role alternation. The
``repair_alternation=True`` path remains responsible for malformed assistant
and tool structure.

Default (``repair_alternation=False``) stays verbatim for inspection and
export consumers such as trace upload and the context guard.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


def _seed_adjacent_user_session(db, session_id="s1"):
    """Persist two adjacent user source turns in otherwise clean history."""
    db.create_session(session_id, "system prompt")
    db.append_message(session_id=session_id, role="user", content="first ask")
    db.append_message(session_id=session_id, role="assistant", content="first reply")
    db.append_message(session_id=session_id, role="user", content="unanswered turn")
    db.append_message(session_id=session_id, role="user", content="next turn")
    db.append_message(session_id=session_id, role="assistant", content="next reply")


def test_default_load_is_verbatim(db):
    _seed_adjacent_user_session(db)
    messages = db.get_messages_as_conversation("s1")
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "user", "assistant"]


def test_repair_alternation_preserves_user_pair_until_provider_wire(db):
    from agent.agent_runtime_helpers import drop_thinking_only_and_merge_users

    _seed_adjacent_user_session(db)
    messages = db.get_messages_as_conversation("s1", repair_alternation=True)
    canonical = [dict(message) for message in messages]

    assert [message["role"] for message in messages] == [
        "user", "assistant", "user", "user", "assistant"
    ]
    assert [messages[2]["content"], messages[3]["content"]] == [
        "unanswered turn", "next turn"
    ]
    assert messages[2] is not messages[3]

    provider_messages = drop_thinking_only_and_merge_users(
        [dict(message) for message in messages]
    )

    assert [message["role"] for message in provider_messages] == [
        "user", "assistant", "user", "assistant"
    ]
    assert provider_messages[2]["content"] == (
        "unanswered turn\n\n[Next user message]\n\nnext turn"
    )
    assert messages == canonical


def test_adjacent_user_load_is_stable_under_canonical_repair(db):
    """Repeated canonical repair leaves adjacent source turns untouched."""
    from agent.agent_runtime_helpers import repair_message_sequence

    _seed_adjacent_user_session(db)
    messages = db.get_messages_as_conversation("s1", repair_alternation=True)
    canonical = [dict(message) for message in messages]

    assert repair_message_sequence(None, messages) == 0
    assert messages == canonical


def test_repair_alternation_repairs_malformed_assistant_pair(db):
    from agent.agent_runtime_helpers import repair_message_sequence

    db.create_session("s3", "system prompt")
    db.append_message(session_id="s3", role="user", content="ask")
    db.append_message(session_id="s3", role="assistant", content="first fragment")
    db.append_message(session_id="s3", role="assistant", content="second fragment")

    verbatim = db.get_messages_as_conversation("s3")
    repaired = db.get_messages_as_conversation("s3", repair_alternation=True)

    assert [message["role"] for message in verbatim] == [
        "user", "assistant", "assistant"
    ]
    assert [message["role"] for message in repaired] == ["user", "assistant"]
    assert repaired[1]["content"] == "first fragment\nsecond fragment"
    assert repair_message_sequence(None, repaired) == 0


def test_repair_noop_on_clean_transcript(db):
    db.create_session("s2", "system prompt")
    db.append_message(session_id="s2", role="user", content="ask")
    db.append_message(session_id="s2", role="assistant", content="reply")
    verbatim = db.get_messages_as_conversation("s2")
    repaired = db.get_messages_as_conversation("s2", repair_alternation=True)
    assert [m["role"] for m in repaired] == [m["role"] for m in verbatim]
    assert [m["content"] for m in repaired] == [m["content"] for m in verbatim]


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
