"""Each automatic compression entry point names itself in the attempt record.

``agent._compress_context(..., trigger=<label>)`` is the only place the content-free attempt
telemetry learns ``trigger_source``; a caller that drops its label is reported as plain ``auto``.
These tests drive the real turn-start and turn-loop callers with a minimal agent double whose
``_compress_context`` records the label and then stops the caller, so each driver only needs the
preconditions that reach the compression call.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from agent.turn_context_compaction import (
    CompactionOutcome, _engine_preflight_maintenance, _idle_compaction, _run_preflight_passes,
)
from agent.turn_preflight import (
    PreflightGateVerdict, compress_after_tool_results, run_preflight_compression,
)


class _CompressionReached(Exception):
    """Raised by the recording double once the caller has invoked ``_compress_context``."""


def _messages() -> list:
    return [
        {"role": "user", "content": "x" * 8_000},
        {"role": "assistant", "content": "y" * 8_000},
        {"role": "user", "content": "current ask"},
    ]


def _agent(labels: list) -> SimpleNamespace:
    def _record_and_stop(_messages, _system, **kwargs):
        labels.append(kwargs.get("trigger"))
        raise _CompressionReached

    compressor = SimpleNamespace(
        threshold_tokens=1_000, context_length=200_000, summary_target_ratio=0.2,
        protect_first_n=1, protect_last_n=1, last_prompt_tokens=5_000,
        last_compression_rough_tokens=0, awaiting_real_usage_after_compression=False,
        should_compress=lambda _tokens: True,
    )
    return SimpleNamespace(
        context_compressor=compressor, compression_enabled=True, model="m", session_id="s1",
        max_compression_attempts=3, _compression_feasibility_checked=True,
        _emit_status=lambda _msg: None, _safe_print=lambda *_a: None,
        _compress_context=_record_and_stop,
    )


def _outcome(messages: list) -> CompactionOutcome:
    return CompactionOutcome(
        messages=messages, active_system_prompt="sys", conversation_history=None,
        current_turn_user_idx=len(messages) - 1,
    )


def _drive_idle(agent: SimpleNamespace) -> None:
    agent.compression_idle_compact_after_seconds = 60
    agent._last_activity_ts = time.time() - 3_600
    _idle_compaction(agent, _outcome(_messages()), "sys", "current ask", "task")


def _drive_turn_start_threshold(agent: SimpleNamespace) -> None:
    _run_preflight_passes(agent, _outcome(_messages()), agent.context_compressor, 5_000, "sys", "task")


def _drive_engine_preflight(agent: SimpleNamespace) -> None:
    agent.context_compressor.should_compress_preflight = lambda _messages: True
    _engine_preflight_maintenance(
        agent, _outcome(_messages()), agent.context_compressor, 500, "sys", "task"
    )


def _drive_pre_api(agent: SimpleNamespace) -> None:
    messages = _messages()
    verdict = PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=None, messages=messages,
        active_system_prompt="sys", conversation_history=None, api_call_count=1,
        compression_attempts=0, final_response=None, failed=False, _turn_exit_reason=None,
        _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=False, _last_preflight_pressure=0,
    )
    run_preflight_compression(
        agent, verdict, compressor=agent.context_compressor, request_pressure_tokens=5_000,
        provider_overflow_preflight=False, defer_preflight=lambda _tokens: False,
        moa_prepared_request=None, system_message="sys", user_message="current ask",
        max_compression_attempts=3, effective_task_id="task",
    )


def _drive_post_tool(agent: SimpleNamespace) -> None:
    messages = _messages()
    compress_after_tool_results(
        agent, messages=messages, system_message="sys", user_message="current ask",
        active_system_prompt="sys", conversation_history=None, compression_attempts=0,
        max_compression_attempts=3, effective_task_id="task", final_response=None,
        turn_exit_reason=None, current_turn_user_idx=len(messages) - 1,
    )


@pytest.mark.parametrize(
    ("drive", "expected_label"),
    [
        pytest.param(_drive_idle, "idle", id="idle"),
        pytest.param(_drive_turn_start_threshold, "turn_start_threshold", id="turn_start_threshold"),
        pytest.param(_drive_engine_preflight, "engine_preflight", id="engine_preflight"),
        pytest.param(_drive_pre_api, "pre_api", id="pre_api"),
        pytest.param(_drive_post_tool, "post_tool", id="post_tool"),
    ],
)
def test_automatic_caller_passes_its_trigger_label(drive, expected_label):
    labels: list = []

    with pytest.raises(_CompressionReached):
        drive(_agent(labels))

    assert labels == [expected_label]
