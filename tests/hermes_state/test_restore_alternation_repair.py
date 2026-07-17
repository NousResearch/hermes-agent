"""get_messages_as_conversation(repair_alternation=True) — heal durable
alternation violations at the restore boundary.

A turn that persists a user row but no assistant row (e.g. its reply was
suppressed, or two concurrent turns interleaved their flushes) leaves a
``user;user`` pair in state.db. Without repair at restore, the defensive
pre-request ``repair_message_sequence`` re-fires on EVERY request for the
rest of the session's life, because it mutates only the per-request list.

Default (``repair_alternation=False``) must stay verbatim: inspection and
export consumers (trace upload, context guard) read the transcript as-is.
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


def test_default_load_is_verbatim(db):
    _seed_wedged_session(db)
    messages = db.get_messages_as_conversation("s1")
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "user", "assistant"]


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
