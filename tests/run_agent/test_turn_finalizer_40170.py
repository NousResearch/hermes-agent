import copy
import logging
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.memory_manager import build_memory_context_block

_concurrent_log_handler = ModuleType("concurrent_log_handler")


class _ConcurrentRotatingFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs):
        filename = args[0] if args else os.devnull
        mode = kwargs.get("mode", "a")
        encoding = kwargs.get("encoding")
        delay = kwargs.get("delay", False)
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)

    def emit(self, record):
        return None

    def doRollover(self):
        return None


_concurrent_log_handler.ConcurrentRotatingFileHandler = _ConcurrentRotatingFileHandler
sys.modules.setdefault("concurrent_log_handler", _concurrent_log_handler)

from agent.turn_finalizer import finalize_turn


def _bare_finalizer_agent():
    return SimpleNamespace(
        max_iterations=20,
        iteration_budget=SimpleNamespace(remaining=19, used=1, max_total=20),
        quiet_mode=True,
        session_id="sess-40170",
        model="anthropic/claude-sonnet-4",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        platform="whatsapp",
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_estimated_cost_usd=0.0,
        session_cost_status="unknown",
        session_cost_source="none",
        context_compressor=SimpleNamespace(last_prompt_tokens=0),
        _tool_guardrail_halt_decision=None,
        _response_was_previewed=False,
        _skill_nudge_interval=0,
        _iters_since_skill=0,
        valid_tool_names=set(),
        _interrupt_message=None,
        _stream_callback=None,
        _turn_failed_file_mutations={},
        _save_trajectory=MagicMock(),
        _cleanup_task_resources=MagicMock(),
        _drop_trailing_empty_response_scaffolding=MagicMock(),
        _persist_session=MagicMock(),
        _file_mutation_verifier_enabled=lambda: False,
        _turn_completion_explainer_enabled=lambda: False,
        _drain_pending_steer=lambda: None,
        clear_interrupt=MagicMock(),
        _sync_external_memory_for_turn=MagicMock(),
        _spawn_background_review=MagicMock(),
    )


def test_finalize_turn_persists_and_syncs_sanitized_recall_output():
    agent = _bare_finalizer_agent()
    trajectory_messages = []
    agent._save_trajectory.side_effect = (
        lambda saved_messages, *_args: trajectory_messages.append(copy.deepcopy(saved_messages))
    )
    leaked = (
        "Visible intro\n\n"
        + build_memory_context_block("operator-only peer card")
        + "\n\nVisible answer"
    )
    messages = [
        {"role": "user", "content": "I need a refund"},
        {"role": "assistant", "content": leaked, "finish_reason": "stop"},
    ]

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = finalize_turn(
            agent,
            final_response=leaked,
            api_call_count=1,
            interrupted=False,
            failed=False,
            messages=messages,
            conversation_history=None,
            effective_task_id="task-40170",
            turn_id="turn-40170",
            user_message="I need a refund",
            original_user_message="I need a refund",
            _should_review_memory=False,
            _turn_exit_reason="text_response(finish_reason=stop)",
        )

    assert result["final_response"] == "Visible intro\n\nVisible answer"
    assert trajectory_messages[-1][-1]["content"] == "Visible intro\n\nVisible answer"
    persisted_messages = agent._persist_session.call_args.args[0]
    assert persisted_messages[-1]["content"] == "Visible intro\n\nVisible answer"
    agent._sync_external_memory_for_turn.assert_called_once_with(
        original_user_message="I need a refund",
        final_response="Visible intro\n\nVisible answer",
        interrupted=False,
        messages=persisted_messages,
    )
