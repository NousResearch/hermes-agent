"""Tool results carry a model-only ``Wall time`` header (port of MoonshotAI/kimi-code#3966).

Invariants: the executor records the measured duration as ``duration_ms`` (its own DB column,
absent for synthetic/blocked results; display_metadata untouched) and ``build_api_messages``
renders it into the wire copy only — persisted content stays the raw tool output.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.message_metadata import append_message
from agent.tool_dispatch_helpers import make_tool_result_message
from agent.tool_executor import _ToolCallRef, _commit_tool_result
from agent.turn_context import build_api_messages
from tools.budget_config import BudgetConfig


def _commit(agent, messages, result, *, blocked, duration):
    ref = _ToolCallRef("terminal", {"command": "ls"}, "task-1", f"call-{len(messages)}", [])
    return _commit_tool_result(
        agent, messages, ref, result, budget=BudgetConfig(), tool_duration=duration,
        is_error=False, blocked=blocked, effect_disposition="none", observed=False,
    )


def test_executor_records_duration_ms_but_not_for_blocked_calls():
    agent = MagicMock()
    agent.verbose_logging = False
    agent._tool_result_content_for_active_model = lambda name, result: result
    agent.tool_result_metadata_callback = lambda call_id, name, args, result: {"inline_diff": "x"}
    agent._flush_messages_to_session_db = lambda messages: True
    agent._subdirectory_hints.check_tool_call.return_value = ""
    messages: list = []

    assert _commit(agent, messages, "exit_code=0\nok", blocked=False, duration=2.3456) is not None
    real = messages[-1]
    assert real["content"] == "exit_code=0\nok"  # persisted content is the raw output
    assert real["duration_ms"] == 2346
    assert real["display_metadata"] == {"inline_diff": "x"}  # display sidecar untouched

    _commit(agent, messages, '{"error": "blocked by policy"}', blocked=True, duration=0.5)
    assert "duration_ms" not in messages[-1]


def test_wire_copy_carries_wall_time_header_and_stays_byte_stable():
    agent = SimpleNamespace(
        _current_turn_timestamp=1_000_000.0, ephemeral_system_prompt="", provider="openai", model="m",
        _copy_reasoning_content_for_api=lambda msg, api_msg: None, _should_sanitize_tool_calls=lambda: False,
    )
    messages: list = []
    append_message(messages, {"role": "user", "content": "run it"})
    append_message(messages, {"role": "assistant", "content": "", "tool_calls": [
        {"id": c, "type": "function", "function": {"name": "t", "arguments": "{}"}} for c in ("c1", "c2", "c3")
    ]})
    timed = make_tool_result_message("terminal", "hello", "c1")
    timed["duration_ms"] = 2346
    multimodal = make_tool_result_message(
        "browser_exec", [{"type": "text", "text": "shot"}, {"type": "image_url", "image_url": {"url": "data:x"}}], "c2")
    multimodal["duration_ms"] = 42
    synthetic = make_tool_result_message("terminal", "[Tool execution cancelled]", "c3")
    for m in (timed, multimodal, synthetic):
        append_message(messages, m)

    kwargs = dict(current_turn_user_idx=0, ext_prefetch_cache=None, plugin_user_context=None,
                  moa_config=None, active_system_prompt="sys")
    api, _ = build_api_messages(agent, messages, **kwargs)
    wire = {m["tool_call_id"]: m for m in api if m.get("role") == "tool"}

    assert wire["c1"]["content"] == "Wall time: 2.346 seconds\nhello"
    assert wire["c2"]["content"][0] == {"type": "text", "text": "Wall time: 0.042 seconds\nshot"}
    assert wire["c2"]["content"][1]["type"] == "image_url"
    assert wire["c3"]["content"] == "[Tool execution cancelled]"
    assert all("duration_ms" not in m for m in wire.values())
    # Persisted rows untouched; a second build sends identical bytes (prompt-cache prefix).
    assert timed["content"] == "hello" and multimodal["content"][0]["text"] == "shot"
    assert build_api_messages(agent, messages, **kwargs)[0] == api
