"""No-progress compression stall must not re-enter after a shorter cooldown (#112420).

A silent summarizer burns the configured idle window (default 120s). The timeout
ladder's first rung is 60s, so automatic compression used to become eligible
again before one stall duration had elapsed — the reporter's 62s retry after
``Recorded stall-interrupted compression backoff``. Summary-LLM ``timeout``
failures keep the 60s rung; only host stall kinds are floored at the idle
window.

When the LLM path degrades, the host applies the existing no-LLM tool-result
prune so the oversized transcript is not returned unchanged. That path does not
require ``proactive_prune_tokens`` (opt-in hysteresis for a different trigger).
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor, _PRUNED_TOOL_PLACEHOLDER
from agent.conversation_compression import (
    CompressionCommitFence,
    _rebuild_system_prompt_at_boundary,
    run_compress_context_with_progress_timeout,
)


def _compressor(**kw):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(model="main-model", quiet_mode=True, **kw)


def test_summary_timeout_keeps_first_ladder_rung():
    compressor = _compressor()
    with patch("agent.context_compressor.time.monotonic", return_value=5_000.0):
        compressor.record_timeout_failure("Request timed out.", failure_kind="timeout")
    assert compressor._summary_failure_cooldown_until == 5_060.0
    assert compressor._consecutive_timeout_failures == 1


def test_stall_interrupted_cooldown_is_at_least_the_idle_window():
    compressor = _compressor()
    with (
        patch("agent.context_compressor.time.monotonic", return_value=5_000.0),
        patch(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            return_value=(120.0, 600.0),
        ),
    ):
        compressor.record_timeout_failure(
            "stall_interrupted:msgs=436:tokens=524788:model=deepseek/deepseek-v4-flash",
            failure_kind="stall_interrupted",
        )
    remaining = compressor._summary_failure_cooldown_until - 5_000.0
    assert remaining >= 120.0
    assert compressor._consecutive_timeout_failures == 1


def test_stalled_host_kind_is_also_floored_at_idle_window():
    compressor = _compressor()
    with (
        patch("agent.context_compressor.time.monotonic", return_value=1_000.0),
        patch(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            return_value=(120.0, 600.0),
        ),
    ):
        compressor.record_timeout_failure(
            "host compress_context timeout (no summary progress)",
            failure_kind="stalled",
        )
    assert compressor._summary_failure_cooldown_until - 1_000.0 >= 120.0


def _assistant_call(cid):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'},
            }
        ],
    }


def _tool_msg(cid, content):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _oversized_tool_transcript():
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(8):
        cid = f"call_{i}"
        msgs.append(_assistant_call(cid))
        payload = ("A" if i < 3 else "ok") if i >= 3 else ("B" * 9000)
        if i < 3:
            payload = chr(65 + i) * 9000
        msgs.append(_tool_msg(cid, payload))
    msgs.append({"role": "user", "content": "continue"})
    return msgs


def test_no_progress_prune_fallback_shrinks_old_tool_results_without_proactive_opt_in():
    from agent.conversation_compression import _apply_no_progress_prune_fallback

    compressor = _compressor(protect_last_n=2, protect_first_n=1)
    assert compressor.proactive_prune_tokens == 0
    original = _oversized_tool_transcript()
    agent = SimpleNamespace(context_compressor=compressor, session_id="s-stall")

    result = _apply_no_progress_prune_fallback(agent, original, progress_observed=False)

    assert result is not original
    old = [m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "call_0"][0]
    assert len(old["content"]) < 9000
    assert old["content"] != _PRUNED_TOOL_PLACEHOLDER


def test_no_progress_prune_fallback_skips_when_summary_progress_was_observed():
    from agent.conversation_compression import _apply_no_progress_prune_fallback

    compressor = _compressor(protect_last_n=2)
    original = _oversized_tool_transcript()
    agent = SimpleNamespace(context_compressor=compressor, session_id="s-stall")
    result = _apply_no_progress_prune_fallback(agent, original, progress_observed=True)
    assert result is original


class _SilentWorker:
    def __init__(self):
        self.release = threading.Event()
        self.attempts = 0

    def __call__(self, fence: CompressionCommitFence):
        self.attempts += 1
        self.release.wait(timeout=5)
        return [{"role": "user", "content": "should-not-publish"}], "unused-prompt"


def test_idle_stall_returns_pruned_transcript_when_llm_path_degrades():
    compressor = _compressor(protect_last_n=2, protect_first_n=1)
    original = _oversized_tool_transcript()
    agent = SimpleNamespace(context_compressor=compressor, session_id="s-stall")
    worker = _SilentWorker()
    timeouts = []
    try:
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"fallback_chain": []},
        ):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=0.05,
                total_ceiling_seconds=2.0,
                on_timeout=lambda *args: timeouts.append(args),
                telemetry_agent=agent,
            )
    finally:
        worker.release.set()

    assert worker.attempts == 1
    assert len(timeouts) == 1
    assert prompt == "degraded-prompt"
    assert msgs is not original
    old = [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == "call_0"][0]
    assert len(old["content"]) < 9000


def test_drifted_system_prompt_logs_once_per_session():
    builds = ["NEW PROMPT A", "NEW PROMPT B"]

    agent = SimpleNamespace(
        session_id="sess-drift",
        _cached_system_prompt="OLD PROMPT",
        _invalidate_system_prompt=lambda: None,
        _build_system_prompt=lambda _system_message: builds.pop(0),
    )

    with (
        patch("agent.conversation_compression._refresh_agent_tool_definitions"),
        patch("agent.system_prompt.reconstruct_static_prefix"),
    ):
        caplog = logging.getLogger("agent.conversation_compression")
        with patch.object(caplog, "info") as info, patch.object(caplog, "debug") as debug:
            first = _rebuild_system_prompt_at_boundary(agent, "system")
            agent._cached_system_prompt = "OLD PROMPT"
            second = _rebuild_system_prompt_at_boundary(agent, "system")

    assert first == "NEW PROMPT A"
    assert second == "NEW PROMPT B"
    drift_infos = [
        call for call in info.call_args_list
        if call.args and "drifted system prompt" in str(call.args[0])
    ]
    drift_debugs = [
        call for call in debug.call_args_list
        if call.args and "drifted system prompt" in str(call.args[0])
    ]
    assert len(drift_infos) == 1
    assert len(drift_debugs) == 1
