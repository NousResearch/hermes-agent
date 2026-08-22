"""Regression tests for retrying direct transcript flushes after a failed start."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_agent(db, session_id: str):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._session_db_created = True
    return agent


def _seed(db, session_id: str) -> None:
    db.create_session(session_id, source="tui")
    db.append_message(session_id, role="user", content="old question")
    db.append_message(session_id, role="assistant", content="old answer")


def _contents(db, session_id: str) -> list[str]:
    return [row["content"] for row in db.get_messages(session_id)]


def test_tool_progress_flush_reuses_baseline_after_turn_start_failure(tmp_path: Path):
    from agent.tool_executor import _flush_session_db_after_tool_progress
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id = "flush-recovery-baseline"
        _seed(db, session_id)
        agent = _make_agent(db, session_id)
        history = db.get_messages_as_conversation(session_id)
        messages = [*history, {"role": "user", "content": "new question"}]
        agent._turn_persistence_history = history

        real_append = db.append_messages_batch
        append_calls = 0

        def fail_once(*args, **kwargs):
            nonlocal append_calls
            append_calls += 1
            if append_calls == 1:
                raise RuntimeError("transient SQLite busy")
            return real_append(*args, **kwargs)

        with patch.object(db, "append_messages_batch", side_effect=fail_once):
            assert agent._flush_messages_to_session_db(
                messages, conversation_history=history
            ) is False
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "tool output",
                }
            )
            assert _flush_session_db_after_tool_progress(
                agent, messages, stage="tool result"
            ) is True

        assert _contents(db, session_id) == [
            "old question",
            "old answer",
            "new question",
            "tool output",
        ]
    finally:
        db.close()


def test_codex_projection_flush_reuses_turn_baseline():
    from agent.codex_runtime import run_codex_app_server_turn

    baseline = [{"role": "user", "content": "old"}]
    agent = MagicMock()
    agent._codex_session.run_turn.return_value = SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[{"role": "assistant", "content": "answer"}],
        tool_iterations=0,
        final_text="answer",
        should_retire=False,
    )
    agent._codex_session.close = MagicMock()
    agent._session_db = object()
    agent._session_db_created = True
    agent._turn_persistence_history = baseline
    agent._flush_messages_to_session_db.return_value = True
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._memory_manager = None
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._session_messages = []

    messages = [{"role": "user", "content": "new"}]
    result = run_codex_app_server_turn(
        agent,
        user_message="new",
        original_user_message="new",
        messages=messages,
        effective_task_id="task-1",
    )

    assert result["agent_persisted"] is True
    agent._flush_messages_to_session_db.assert_called_once_with(
        messages,
        conversation_history=baseline,
    )


def test_compression_boundary_publishes_new_direct_flush_baseline():
    from agent.conversation_compression import conversation_history_after_compression

    messages = [{"role": "assistant", "content": "compacted"}]
    agent = SimpleNamespace(
        _last_compression_attempt_recorded=False,
        _last_compaction_in_place=True,
    )

    baseline = conversation_history_after_compression(agent, messages)

    assert baseline == messages
    assert baseline is not messages
    assert agent._turn_persistence_history is baseline
