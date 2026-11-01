"""Unclassified stops must still leave a user-visible ``Why:`` on the durable notice.

Interrupt-before-tool used to return an interrupted result with no classified fields,
then ``clear_interrupt(hard_cancel=True)`` wiped ``_tool_interrupt_reason``. No tool
ran, so ``close_interrupted_tool_sequence`` did not append, the durable tail stayed
``user``, and the closer wrote a bare notice.
"""

from types import SimpleNamespace

from agent.conversation_loop import _close_durable_failed_turn
from agent.turn_recovery import abort_turn_on_interrupt
from hermes_state import SessionDB


def test_interrupt_before_any_tool_persists_a_why(tmp_path, monkeypatch):
    """After the interrupt reason is cleared, the durable notice still names why."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    sid = "s-interrupt-before-tool"
    db.create_session(session_id=sid, source="cli", model="m")
    db.append_message(sid, "user", "do the thing")

    messages = [{"role": "user", "content": "do the thing"}]

    def flush(msgs):
        last = msgs[-1]
        db.append_message(
            sid,
            last["role"],
            last.get("content") or "",
            display_kind=last.get("display_kind"),
            display_metadata=last.get("display_metadata"),
        )

    agent = SimpleNamespace(
        _session_db=db,
        session_id=sid,
        _flush_messages_to_session_db=flush,
        _tool_interrupt_reason="user interrupt",
        _interrupt_message=None,
        _interrupt_requested=True,
        log_prefix="",
        provider="test",
        model="test-model",
        _vprint=lambda *args, **kwargs: None,
        _drop_trailing_empty_response_scaffolding=lambda msgs: None,
        _persist_session=lambda *args, **kwargs: None,
    )

    def clear_interrupt(*, preserve_redirect=False, hard_cancel=False):
        agent._tool_interrupt_reason = None
        agent._interrupt_message = None
        agent._interrupt_requested = False
        return True

    agent.clear_interrupt = clear_interrupt

    result = abort_turn_on_interrupt(
        agent,
        messages,
        [],
        0,
        abort_message="Interrupt detected before any tool ran.",
        interrupt_text="Operation interrupted: user interrupt.",
    )
    # Production wipes the agent reason before the closer runs.
    assert agent._tool_interrupt_reason is None
    assert result.get("interrupted") is True
    assert result.get("completed") is False
    assert messages[-1]["role"] == "user"

    _close_durable_failed_turn(agent, result)

    notice = messages[-1]["content"]
    stored = db.get_messages(sid)[-1]["content"]
    assert messages[-1]["role"] == "assistant"
    assert "Why:" in notice
    assert "Why:" in stored
    assert "stopped by user" in notice
    db.close()


def test_unrecorded_interrupt_before_any_tool_is_not_a_user_stop(tmp_path, monkeypatch):
    """A pre-tool stop with no recorded issuer must not be booked as the user."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    sid = "s-interrupt-unrecorded"
    db.create_session(session_id=sid, source="cli", model="m")
    db.append_message(sid, "user", "do the thing")
    messages = [{"role": "user", "content": "do the thing"}]

    def flush(msgs):
        last = msgs[-1]
        db.append_message(sid, last["role"], last.get("content") or "")

    agent = SimpleNamespace(
        _session_db=db,
        session_id=sid,
        _flush_messages_to_session_db=flush,
        _tool_interrupt_reason=None,
        _interrupt_message=None,
        _interrupt_requested=True,
        log_prefix="",
        provider="test",
        model="test-model",
        _vprint=lambda *args, **kwargs: None,
        _drop_trailing_empty_response_scaffolding=lambda msgs: None,
        _persist_session=lambda *args, **kwargs: None,
    )

    def clear_interrupt(*, preserve_redirect=False, hard_cancel=False):
        agent._tool_interrupt_reason = None
        return True

    agent.clear_interrupt = clear_interrupt
    result = abort_turn_on_interrupt(
        agent, messages, [], 0,
        abort_message="Interrupt detected before any tool ran.",
        interrupt_text="Operation interrupted.",
    )
    _close_durable_failed_turn(agent, result)
    notice = messages[-1]["content"]
    assert "Why:" in notice
    assert "stopped by user" not in notice
    assert "unrecorded" in notice
    db.close()
