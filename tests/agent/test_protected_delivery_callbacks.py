from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.tool_executor import (
    _ToolCallRef,
    _begin_tool_execution,
    _emit_tool_complete_and_risk,
)
from agent.turn_response_intake import normalize_model_response


def test_protected_turn_suppresses_tool_callbacks():
    agent = SimpleNamespace(
        _pre_delivery_gate_active=True,
        quiet_mode=False,
        tool_progress_mode="all",
        verbose_logging=False,
        log_prefix_chars=0,
        log_prefix="",
        tool_progress_callback=Mock(),
        tool_start_callback=Mock(),
        tool_complete_callback=Mock(),
        _checkpoint_mgr=SimpleNamespace(enabled=False),
        _current_tool=None,
        _touch_activity=Mock(),
    )
    ref = _ToolCallRef(
        name="search",
        args={"query": "sensitive clinical query"},
        task_id="task",
        call_id="call",
        trace=[],
    )

    with patch("agent.tool_executor._set_worker_activity_callback"):
        _begin_tool_execution(agent, ref, 1)
    _emit_tool_complete_and_risk(
        agent,
        ref,
        result="sensitive clinical result",
        risk_metadata={"risk": "high"},
        blocked=False,
    )

    agent.tool_progress_callback.assert_not_called()
    agent.tool_start_callback.assert_not_called()
    agent.tool_complete_callback.assert_not_called()


def test_protected_turn_suppresses_reasoning_callback():
    callback = Mock()
    agent = SimpleNamespace(
        _pre_delivery_gate_active=True,
        quiet_mode=True,
        verbose_logging=False,
        log_prefix="",
        tool_progress_callback=callback,
        _incomplete_scratchpad_retries=0,
        api_mode="",
    )
    assistant_message = SimpleNamespace(
        content="sensitive clinical reasoning",
        finish_reason="stop",
    )

    with (
        patch(
            "agent.turn_response_intake.normalize_response_for_agent",
            return_value=assistant_message,
        ),
        patch("agent.turn_response_intake.splice_provider_projection"),
        patch("agent.turn_response_intake._fire_post_api_request_hook"),
    ):
        verdict = normalize_model_response(
            agent,
            response=object(),
            messages=[],
            api_messages=[],
            conversation_history=[],
            api_call_count=1,
            api_duration=0.1,
            api_start_time=0.0,
            api_request_id="request",
            effective_task_id="task",
            turn_id="turn",
        )

    assert verdict.action == "fallthrough"
    callback.assert_not_called()