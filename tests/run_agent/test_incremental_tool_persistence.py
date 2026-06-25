"""Regression coverage for crash-safe tool-loop persistence.

A long TUI turn can run dozens of API/tool iterations before producing a final
assistant response. If Hermes is rebuilt/restarted during that window, only the
early user turn was durable before this regression test existed. The next resume
then loaded a one-message session even though logs proved substantial tool work
had happened.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _tool_call_response():
    tool_call = SimpleNamespace(
        id="call_read_1",
        type="function",
        function=SimpleNamespace(
            name="read_file",
            arguments=json.dumps({"path": "README.md"}),
        ),
    )
    message = SimpleNamespace(content=None, reasoning=None, tool_calls=[tool_call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=None,
    )


def _final_response():
    message = SimpleNamespace(content="done", reasoning=None, tool_calls=[])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )


class _CheckpointAwareCompletions:
    def __init__(self, session_db):
        self.session_db = session_db
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _tool_call_response()

        # This second request is assembled after the first tool call was
        # executed but before the final assistant response exists. The durable
        # store must already contain the assistant tool-call turn and its tool
        # result, otherwise a rebuild/restart in this window resumes a
        # one-message session.
        roles = [call.kwargs.get("role") for call in self.session_db.append_message.call_args_list]
        assert roles == ["user", "assistant", "tool"]
        return _final_response()


def test_tool_results_checkpoint_before_next_model_call(monkeypatch):
    session_db = MagicMock()
    completions = _CheckpointAwareCompletions(session_db)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [{"function": {"name": "read_file"}}],
    )
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True}),
    )

    agent = AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform="tui",
        session_id="checkpoint-session",
        session_db=session_db,
        max_iterations=3,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    setattr(agent, "_disable_streaming", True)

    result = agent.run_conversation("read the file")

    assert result["final_response"] == "done"
    roles = [call.kwargs.get("role") for call in session_db.append_message.call_args_list]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_checkpoint_continues_to_db_when_json_log_fails():
    agent = SimpleNamespace(
        session_id="checkpoint-session",
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _session_messages=[],
        _stash_api_user_message_override=lambda messages: None,
        _apply_persist_user_message_override=lambda messages: None,
        _strip_internal_message_fields=lambda messages: [m.copy() for m in messages],
        _save_session_log=MagicMock(side_effect=OSError("disk full")),
        _flush_messages_to_session_db=MagicMock(),
    )

    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "", "tool_calls": []},
    ]

    getattr(AIAgent, "_checkpoint_session_progress")(agent, messages, reason="test")

    agent._save_session_log.assert_called_once()
    agent._flush_messages_to_session_db.assert_called_once_with(messages, None)
