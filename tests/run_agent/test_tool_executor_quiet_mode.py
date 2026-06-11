from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.tool_executor import execute_tool_calls_concurrent


class _Allows:
    allows_execution = True


def _tool_call(idx: int):
    return SimpleNamespace(
        id=f"call-{idx}",
        function=SimpleNamespace(
            name="web_search",
            arguments=json.dumps({"query": f"query {idx}"}),
        ),
    )


def _quiet_agent():
    agent = MagicMock()
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent.tool_progress_mode = "all"
    agent.log_prefix = ""
    agent.log_prefix_chars = 80
    agent._interrupt_requested = False
    agent._current_tool = None
    agent._tool_worker_threads = set()
    agent._tool_worker_threads_lock = threading.Lock()
    agent._checkpoint_mgr = SimpleNamespace(enabled=False)
    agent._tool_guardrails = SimpleNamespace(before_call=lambda name, args: _Allows())
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    agent.tool_complete_callback = None
    agent._subdirectory_hints = SimpleNamespace(check_tool_call=lambda name, args: "")
    agent._touch_activity = lambda *args, **kwargs: None
    agent._append_guardrail_observation = lambda name, args, result, failed=False: result
    agent._record_file_mutation_result = lambda *args, **kwargs: None
    agent._should_emit_quiet_tool_messages = lambda: False
    agent._should_start_quiet_spinner = lambda: False
    agent._tool_result_content_for_active_model = lambda name, result: result
    agent._apply_pending_steer_to_tool_results = lambda *args, **kwargs: None
    agent._invoke_tool = lambda name, args, task_id, call_id, **kwargs: json.dumps(
        {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Contractual nursing officers leaked preview",
                        "url": "https://example.test",
                    }
                ]
            },
        }
    )
    return agent


def test_concurrent_tool_executor_respects_quiet_mode(capsys):
    """Quiet child agents must not print per-tool result previews.

    delegate_task builds children with quiet_mode=True. The sequential executor
    already respects that flag, but the concurrent executor used a separate
    print gate and leaked lines like:

        ✅ Tool 2 completed in 8.02s - {"success": true, "data": ...}

    That exposes subagent intermediate tool output in the parent CLI.
    """
    agent = _quiet_agent()
    assistant_message = SimpleNamespace(tool_calls=[_tool_call(1), _tool_call(2), _tool_call(3)])
    messages = []

    execute_tool_calls_concurrent(agent, assistant_message, messages, effective_task_id="test-task")

    captured = capsys.readouterr()
    assert "✅ Tool" not in captured.out
    assert "Contractual nursing officers leaked preview" not in captured.out
    assert len(messages) == 3
