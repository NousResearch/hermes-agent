import json
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_compress_context_emits_reasoning_json_message():
    from run_agent import AIAgent

    emitted = []
    agent = object.__new__(AIAgent)
    agent.session_id = "session-parent"
    agent.model = "test-model"
    agent.tools = []
    agent._memory_manager = None
    agent._session_db = None
    agent._todo_store = SimpleNamespace(format_for_injection=lambda: "")
    agent.context_compressor = SimpleNamespace(
        compression_count=1,
        last_prompt_tokens=0,
        last_completion_tokens=0,
        _last_summary_error=None,
        _last_aux_model_failure_model=None,
        _last_aux_model_failure_error=None,
        compress=MagicMock(return_value=[
            {"role": "user", "content": "compressed handoff"},
            {"role": "assistant", "content": "ready"},
        ]),
    )
    agent._invalidate_system_prompt = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="system prompt")
    agent._cached_system_prompt = None
    agent._vprint = MagicMock()
    agent._emit_warning = MagicMock()
    agent.commit_memory_session = MagicMock()
    agent.reasoning_callback = emitted.append

    compressed, system_prompt = agent._compress_context(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ],
        "base system",
        approx_tokens=1234,
        task_id="test-task",
    )

    assert compressed == [
        {"role": "user", "content": "compressed handoff"},
        {"role": "assistant", "content": "ready"},
    ]
    assert system_prompt == "system prompt"
    assert emitted

    message = json.loads(emitted[-1])
    assert message["type"] == "context_compression"
    assert message["status"] == "completed"
    assert message["messages_before"] == 3
    assert message["messages_after"] == 2
    assert message["tokens_before"] == 1234
    assert message["tokens_after"] >= 0
    assert message["compression_count"] == 1
