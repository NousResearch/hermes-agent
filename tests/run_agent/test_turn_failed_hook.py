"""Tests for the ``turn_failed`` plugin hook emitted by ``finalize_turn``.

Phase-0 agent-observability (T1): the turn finalizer resolves a 13-value
``_turn_exit_reason`` and logs a "Turn ended: reason=..." diagnostic, but there
was no programmatic hook for downstream observability. This adds ``turn_failed``,
fired ONLY for non-clean turn exits:

  * any error / exhaustion / guardrail reason, OR
  * ``last_msg_role == "tool"`` (the agent stopped mid-work — the
    ``protocol_violation`` / ``breads-pc`` premature-stop class).

A healthy ``text_response(finish_reason=stop)`` exit with
``last_msg_role != "tool"`` must NOT fire it, and neither must a deliberate
user ``/stop`` (``interrupted`` True) — that is a clean exit, not a failure.

The classification lives in the pure guard ``_should_emit_turn_failed`` so it is
deterministic and testable without a full agent. A small integration-style test
drives the real ``finalize_turn`` path with a mocked ``invoke_hook`` to confirm
the emit wiring + kwargs.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from hermes_cli.plugins import VALID_HOOKS
from agent.turn_finalizer import _should_emit_turn_failed, finalize_turn


# --------------------------------------------------------------------------- #
# Hook registration
# --------------------------------------------------------------------------- #
def test_turn_failed_in_valid_hooks():
    assert "turn_failed" in VALID_HOOKS


# --------------------------------------------------------------------------- #
# Pure guard: _should_emit_turn_failed(reason, last_msg_role, interrupted)
# --------------------------------------------------------------------------- #
def test_guard_fires_on_protocol_violation_shaped_premature_stop():
    # Worker stopped mid-work: last message is a tool result. This is the
    # protocol_violation class — fire regardless of reason text.
    assert (
        _should_emit_turn_failed("text_response(finish_reason=stop)", "tool", False)
        is True
    )


def test_guard_fires_on_breads_pc_premature_text_response_with_pending_tool():
    # breads-pc: premature text_response while a tool call is still pending,
    # i.e. last_msg_role == "tool".
    assert (
        _should_emit_turn_failed(
            "text_response(finish_reason=tool_calls)", "tool", False
        )
        is True
    )


def test_guard_fires_on_error_and_exhaustion_reasons():
    for reason in (
        "max_iterations_reached(50/50)",
        "empty_response_exhausted",
        "api_request_error",
        "guardrail_triggered",
    ):
        assert _should_emit_turn_failed(reason, "assistant", False) is True, reason


def test_guard_does_not_fire_on_healthy_completion():
    # Healthy: text_response(...) AND last message is NOT a tool result.
    assert (
        _should_emit_turn_failed(
            "text_response(finish_reason=stop)", "assistant", False
        )
        is False
    )
    assert (
        _should_emit_turn_failed("text_response(finish_reason=stop)", None, False)
        is False
    )


def test_guard_does_not_fire_on_user_interrupt():
    # A deliberate user /stop is a clean exit, not a failure. The interrupt
    # flag suppresses the hook regardless of reason text or last_msg_role —
    # including the interrupted_by_user reason (not a text_response(...)) and a
    # mid-tool stop, both of which would otherwise trip the reason/tool arms.
    for reason, role in (
        ("interrupted_by_user", "assistant"),
        ("interrupted_by_user", "tool"),
        ("text_response(finish_reason=stop)", "tool"),
        ("api_request_error", "assistant"),
    ):
        assert (
            _should_emit_turn_failed(reason, role, True) is False
        ), (reason, role)


# --------------------------------------------------------------------------- #
# Integration: real finalize_turn path with a mocked invoke_hook
# --------------------------------------------------------------------------- #
def _make_agent():
    budget = SimpleNamespace(remaining=10, used=3, max_total=50)
    return SimpleNamespace(
        model="anthropic/claude-x",
        provider="anthropic",
        base_url="https://example.test",
        session_id="sess-123",
        max_iterations=50,
        iteration_budget=budget,
        quiet_mode=True,
        platform="cli",
        # Token / cost accounting referenced when assembling the result dict.
        session_input_tokens=0,
        session_output_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_reasoning_tokens=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_estimated_cost_usd=0.0,
        session_cost_status="ok",
        session_cost_source="estimate",
        context_compressor=SimpleNamespace(last_prompt_tokens=0),
        _tool_guardrail_halt_decision=None,
        _response_was_previewed=False,
        _interrupt_message=None,
        _stream_callback=None,
        _skill_nudge_interval=0,
        _iters_since_skill=0,
        valid_tool_names=set(),
        # finalize_turn calls these — stub them as no-ops.
        _emit_status=lambda *a, **k: None,
        _safe_print=lambda *a, **k: None,
        _save_trajectory=lambda *a, **k: None,
        _cleanup_task_resources=lambda *a, **k: None,
        _drop_trailing_empty_response_scaffolding=lambda *a, **k: None,
        _persist_session=lambda *a, **k: None,
        _file_mutation_verifier_enabled=lambda: False,
        _turn_completion_explainer_enabled=lambda: False,
        _turn_failed_file_mutations={},
        _drain_pending_steer=lambda *a, **k: None,
        clear_interrupt=lambda *a, **k: None,
        _sync_external_memory_for_turn=lambda *a, **k: None,
        _spawn_background_review=lambda *a, **k: None,
    )


def _capture_turn_failed_calls(monkeypatch_hook):
    """Patch invoke_hook so we record turn_failed kwargs; other hooks no-op."""
    calls = []

    def _fake_invoke_hook(name, **kwargs):
        if name == "turn_failed":
            calls.append(kwargs)
        return []

    return calls, _fake_invoke_hook


def test_finalize_fires_turn_failed_on_pending_tool_premature_stop():
    agent = _make_agent()
    messages = [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "kanban_complete"}}],
        },
        {"role": "tool", "content": "ok"},
    ]
    calls, fake = _capture_turn_failed_calls(None)
    with patch("hermes_cli.plugins.invoke_hook", side_effect=fake):
        finalize_turn(
            agent,
            final_response="partial",
            api_call_count=5,
            interrupted=False,
            failed=False,
            messages=messages,
            conversation_history=[],
            effective_task_id="task-9",
            turn_id="turn-1",
            user_message="do the thing",
            original_user_message="do the thing",
            _should_review_memory=False,
            _turn_exit_reason="text_response(finish_reason=stop)",
        )
    assert len(calls) == 1, "turn_failed should fire exactly once on pending-tool stop"
    kw = calls[0]
    assert kw["reason"] == "text_response(finish_reason=stop)"
    assert kw["last_msg_role"] == "tool"
    assert kw["model"] == "anthropic/claude-x"
    assert kw["session_id"] == "sess-123"
    assert kw["api_calls"] == 5
    assert kw["response_len"] == len("partial")
    assert kw["turn_id"] == "turn-1"
    assert kw["interrupted"] is False
    assert "tool_turns" in kw


def test_finalize_does_not_fire_turn_failed_on_user_interrupt_mid_tool():
    # A user /stop while a tool result is pending: finalize appends a synthetic
    # assistant close (so last_msg_role flips to "assistant"), and the interrupt
    # flag suppresses the hook regardless. This is the clean-stop path that must
    # NOT surface as a failure signal.
    agent = _make_agent()
    messages = [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "kanban_complete"}}],
        },
        {"role": "tool", "content": "ok"},
    ]
    calls, fake = _capture_turn_failed_calls(None)
    with patch("hermes_cli.plugins.invoke_hook", side_effect=fake):
        finalize_turn(
            agent,
            final_response="partial",
            api_call_count=5,
            interrupted=True,
            failed=False,
            messages=messages,
            conversation_history=[],
            effective_task_id="task-9",
            turn_id="turn-3",
            user_message="do the thing",
            original_user_message="do the thing",
            _should_review_memory=False,
            _turn_exit_reason="interrupted_by_user",
        )
    assert calls == [], "turn_failed must NOT fire on a deliberate user interrupt"


def test_finalize_does_not_fire_turn_failed_on_healthy_completion():
    agent = _make_agent()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Done."},
    ]
    calls, fake = _capture_turn_failed_calls(None)
    with patch("hermes_cli.plugins.invoke_hook", side_effect=fake):
        finalize_turn(
            agent,
            final_response="Done.",
            api_call_count=2,
            interrupted=False,
            failed=False,
            messages=messages,
            conversation_history=[],
            effective_task_id="task-9",
            turn_id="turn-2",
            user_message="hi",
            original_user_message="hi",
            _should_review_memory=False,
            _turn_exit_reason="text_response(finish_reason=stop)",
        )
    assert calls == [], "turn_failed must NOT fire on a healthy completed turn"
