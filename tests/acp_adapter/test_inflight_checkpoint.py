from __future__ import annotations

from acp_adapter.session import SessionManager
from hermes_state import SessionDB


class FakeAgent:
    model = "fake-model"
    provider = "fake-provider"
    base_url = "http://fake"
    api_mode = "chat_completions"


def make_manager(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    return SessionManager(agent_factory=lambda: FakeAgent(), db=db), db


def test_inflight_checkpoint_restores_partial_turn_after_restart(tmp_path):
    manager, db = make_manager(tmp_path)
    state = manager.create_session(cwd=str(tmp_path))

    messages = [
        {"role": "user", "content": "Build the thing"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command":"pwd"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_name": "terminal",
            "tool_call_id": "call-1",
            "content": "output from pwd",
        },
    ]

    manager.checkpoint_inflight_turn(
        state.session_id,
        messages,
        current_prompt_text="Build the thing",
        started_at=1234.0,
    )

    restored_manager = SessionManager(agent_factory=lambda: FakeAgent(), db=db)
    restored = restored_manager.get_session(state.session_id)

    assert restored is not None
    assert restored.history == messages
    assert restored.incomplete_turn is True
    assert restored.current_prompt_text == "Build the thing"
    assert restored.current_prompt_started_at == 1234.0


def test_complete_inflight_turn_clears_marker_but_keeps_history(tmp_path):
    manager, db = make_manager(tmp_path)
    state = manager.create_session(cwd=str(tmp_path))

    partial = [{"role": "user", "content": "Build the thing"}]
    manager.checkpoint_inflight_turn(
        state.session_id,
        partial,
        current_prompt_text="Build the thing",
        started_at=1234.0,
    )

    final = partial + [{"role": "assistant", "content": "Done"}]
    manager.complete_inflight_turn(state.session_id, final)

    restored_manager = SessionManager(agent_factory=lambda: FakeAgent(), db=db)
    restored = restored_manager.get_session(state.session_id)

    assert restored is not None
    assert restored.history == final
    assert restored.incomplete_turn is False
    assert restored.current_prompt_text == ""
    assert restored.current_prompt_started_at is None
