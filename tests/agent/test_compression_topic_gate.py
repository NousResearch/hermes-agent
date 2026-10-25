"""Regression coverage for the opt-in automatic compression topic gate (#138646)."""

from types import SimpleNamespace
from unittest.mock import Mock

from agent.compression_topic_gate import should_defer_automatic_compression
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression


def _agent(*, enabled=True, context_length=10_000):
    return SimpleNamespace(
        compression_topic_gate_enabled=enabled,
        context_compressor=SimpleNamespace(context_length=context_length),
    )


def test_unrelated_turn_defers_automatic_compression_when_it_fits_the_window():
    messages = [
        {"role": "user", "content": "Review the Acme quarterly sales pipeline."},
        {"role": "assistant", "content": "The Acme pipeline has three enterprise deals."},
        {"role": "user", "content": "How do I replace a leaking kitchen faucet?"},
    ]

    assert should_defer_automatic_compression(_agent(), messages, current_tokens=5_000) is True


def test_related_turn_and_over_window_turn_keep_automatic_compression():
    messages = [
        {"role": "user", "content": "Review the Acme quarterly sales pipeline."},
        {"role": "assistant", "content": "The Acme pipeline has three enterprise deals."},
        {"role": "user", "content": "Which Acme deal should we prioritize?"},
    ]

    assert should_defer_automatic_compression(_agent(), messages, current_tokens=5_000) is False
    assert should_defer_automatic_compression(_agent(), messages, current_tokens=10_000) is False


def test_disabled_topic_gate_preserves_existing_automatic_compression():
    messages = [
        {"role": "user", "content": "Review the Acme quarterly sales pipeline."},
        {"role": "user", "content": "How do I replace a leaking kitchen faucet?"},
    ]

    assert should_defer_automatic_compression(
        _agent(enabled=False), messages, current_tokens=5_000
    ) is False


def test_pre_api_gate_does_not_call_the_compressor_for_an_unrelated_turn():
    messages = [
        {"role": "user", "content": "Review the Acme quarterly sales pipeline."},
        {"role": "assistant", "content": "The Acme pipeline has three enterprise deals."},
        {"role": "user", "content": "How do I replace a leaking kitchen faucet?"},
    ]
    compressor = SimpleNamespace(
        context_length=10_000, threshold_tokens=1_000, should_compress=lambda _tokens: True,
        get_active_compression_failure_cooldown=lambda: None,
    )
    agent = SimpleNamespace(
        compression_topic_gate_enabled=True, compression_enabled=True, context_compressor=compressor,
        _compression_feasibility_checked=True, model="test", _compress_context=Mock(),
    )
    verdict = PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=None, messages=messages,
        active_system_prompt="system", conversation_history=None, api_call_count=1,
        compression_attempts=0, final_response=None, failed=False, _turn_exit_reason=None,
        _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=False, _last_preflight_pressure=None,
    )

    result = run_preflight_compression(
        agent, verdict, compressor=compressor, request_pressure_tokens=5_000,
        provider_overflow_preflight=False, defer_preflight=lambda _tokens: False,
        moa_prepared_request=None, system_message="system", user_message=messages[-1]["content"],
        max_compression_attempts=3, effective_task_id="task",
    )

    assert result.action == "fallthrough"
    agent._compress_context.assert_not_called()
