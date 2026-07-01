from agent.message_sanitization import close_interrupted_tool_sequence
from gateway.run import _is_interrupt_close_tail
from hermes_state import SessionDB
from run_agent import AIAgent


def test_interrupt_close_flag_survives_session_db_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess-interrupt-roundtrip"
    db.create_session(session_id=session_id, source="discord")

    persisted_user = {"role": "user", "content": "scan the repo"}
    db.append_message(session_id=session_id, role="user", content=persisted_user["content"])

    live_messages = [
        persisted_user,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "terminal", "arguments": '{"command":"rg TODO"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "tool_name": "terminal",
            "content": "src/a.py: TODO",
        },
    ]
    assert close_interrupted_tool_sequence(live_messages, None) is True

    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.platform = "discord"
    agent._last_flushed_db_idx = 1
    agent._flushed_db_message_ids = {id(persisted_user)}
    agent._flushed_db_message_session_id = session_id

    agent._flush_messages_to_session_db(live_messages)

    reloaded = db.get_messages(session_id)
    assert reloaded[-1]["role"] == "assistant"
    assert reloaded[-1]["content"] == "Operation interrupted."
    assert reloaded[-1]["finish_reason"] == "interrupt_close"
    assert _is_interrupt_close_tail(reloaded) is True


def test_plain_operation_interrupted_text_is_not_interrupt_close(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess-interrupt-control"
    db.create_session(session_id=session_id, source="discord")
    db.append_message(session_id=session_id, role="user", content="hello")
    db.append_message(
        session_id=session_id,
        role="assistant",
        content="Operation interrupted.",
        finish_reason=None,
    )

    reloaded = db.get_messages(session_id)
    assert reloaded[-1].get("finish_reason") is None
    assert _is_interrupt_close_tail(reloaded) is False


def test_interrupt_close_flag_survives_INPLACE_assistant_tail_roundtrip(tmp_path, monkeypatch):
    """C1 (pass-2 review): the in-place assistant-text-tail shape.

    When a turn is interrupted while the transcript already ends on a plain
    assistant text message (no tool_calls), close_interrupted_tool_sequence
    mutates that EXISTING dict in place (last["finish_reason"]="interrupt_close")
    rather than appending a new turn. If that assistant dict was ALREADY flushed
    to state.db mid-turn, _flush_messages_to_session_db's identity tracking
    (_flushed_db_message_ids) would skip re-writing it -> the flag would be lost
    on reload -> resume silently falls to the "skip unfinished work" branch
    (the INV-D2 defect). The fix tracks each flushed row id and re-persists the
    finish_reason column when an already-flushed message is later flagged.

    This test drives the REAL flush path for the initial persist (so row-id
    tracking is populated exactly as in production), then mutates in place and
    re-flushes.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess-interrupt-inplace"
    db.create_session(session_id=session_id, source="discord")

    persisted_user = {"role": "user", "content": "summarize the diff"}
    db.append_message(session_id=session_id, role="user", content=persisted_user["content"])

    # A plain-text assistant tail (partial streamed text, NO tool_calls).
    assistant_tail = {"role": "assistant", "content": "Here is what I found so far"}
    live_messages = [persisted_user, assistant_tail]

    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.platform = "discord"
    agent._last_flushed_db_idx = 1
    agent._flushed_db_message_ids = {id(persisted_user)}
    agent._flushed_db_message_session_id = session_id

    # FIRST flush through the real method — persists assistant_tail AND records
    # its row id in _flushed_db_row_ids (exactly the mid-turn incremental flush).
    agent._flush_messages_to_session_db(live_messages)
    reloaded = db.get_messages(session_id)
    assert reloaded[-1]["content"] == "Here is what I found so far"
    assert reloaded[-1].get("finish_reason") is None  # not yet flagged

    # Interrupt lands on the assistant-text tail -> in-place mutation branch.
    assert close_interrupted_tool_sequence(
        live_messages, None, interrupted_assistant_tail=True
    ) is True
    assert assistant_tail["finish_reason"] == "interrupt_close"

    # Re-flush after the in-place mutation.
    agent._flush_messages_to_session_db(live_messages)

    reloaded = db.get_messages(session_id)
    assert reloaded[-1]["role"] == "assistant"
    assert reloaded[-1]["content"] == "Here is what I found so far"
    # THE CRITICAL ASSERTION: does the flag survive the in-place re-flush?
    assert reloaded[-1]["finish_reason"] == "interrupt_close", (
        "in-place finish_reason mutation was LOST on re-flush (C1 silent-no-op): "
        f"got {reloaded[-1].get('finish_reason')!r}"
    )
    assert _is_interrupt_close_tail(reloaded) is True
    # And no duplicate row was appended (still user + one assistant).
    assert len(reloaded) == 2


def test_interrupt_close_inplace_appended_with_flag_not_double_written(tmp_path, monkeypatch):
    """If a message is appended already carrying the flag, it is durable with no
    later re-persist needed (repersisted set short-circuits)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess-interrupt-preflagged"
    db.create_session(session_id=session_id, source="discord")
    user = {"role": "user", "content": "hi"}
    db.append_message(session_id=session_id, role="user", content=user["content"])
    tail = {"role": "assistant", "content": "done", "finish_reason": "interrupt_close"}
    live = [user, tail]

    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.platform = "discord"
    agent._last_flushed_db_idx = 1
    agent._flushed_db_message_ids = {id(user)}
    agent._flushed_db_message_session_id = session_id

    agent._flush_messages_to_session_db(live)
    # second flush must not duplicate or error
    agent._flush_messages_to_session_db(live)
    reloaded = db.get_messages(session_id)
    assert len(reloaded) == 2
    assert reloaded[-1]["finish_reason"] == "interrupt_close"
