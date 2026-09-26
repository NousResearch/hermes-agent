from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression


def _verdict():
    return PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=None,
        messages=[{"role": "user", "content": "old"}, {"role": "user", "content": "new"}],
        active_system_prompt="system", conversation_history=[], api_call_count=1,
        compression_attempts=0, final_response="", failed=False, _turn_exit_reason=None,
        _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=False, _last_preflight_pressure=None,
    )


def _agent(fail_closed_tokens):
    compressor = SimpleNamespace(
        threshold_tokens=24_000, context_length=65_536,
        prefill_cost_cap=SimpleNamespace(fail_closed_tokens=fail_closed_tokens),
        should_compress=lambda _tokens: True,
        get_active_compression_failure_cooldown=lambda: None,
    )
    agent = SimpleNamespace(
        compression_enabled=True, context_compressor=compressor,
        model="qwen38-27b-q2-64k", session_id="test", _api_call_count=1,
        iteration_budget=SimpleNamespace(refund=MagicMock()),
        _emit_status=MagicMock(), _flush_status_buffer=MagicMock(),
        _persist_session=MagicMock(), _compression_blocked_transient=None,
        _compression_skipped_due_to_lock=None,
    )

    def _compress(messages, *_args, **_kwargs):
        agent._compression_skipped_due_to_lock = "held"
        return messages, "system"

    agent._compress_context = _compress
    return agent


def _run(agent, pressure):
    return run_preflight_compression(
        agent, _verdict(), compressor=agent.context_compressor,
        request_pressure_tokens=pressure, provider_overflow_preflight=False,
        defer_preflight=lambda _: False, moa_prepared_request=None,
        system_message={"role": "system", "content": "system"}, user_message="user",
        max_compression_attempts=3, effective_task_id=None,
    )


def test_prefill_cap_lock_skip_above_fail_closed_defers():
    verdict = _run(_agent(28_000), 29_000)
    assert verdict.action == "return"
    assert verdict.result is not None
    assert verdict.result["compression_deferred"] is True
    assert verdict.result["failed"] is False
    assert verdict.result["api_calls"] == 0


def test_prefill_cap_lock_skip_below_fail_closed_falls_through():
    verdict = _run(_agent(28_000), 25_000)
    assert verdict.action == "fallthrough"
    assert verdict.api_call_count == 1


def test_prefill_warning_emitted_once_before_compression():
    agent = _agent(28_000)
    agent.context_compressor.prefill_cost_cap.warn_tokens = 16_000
    agent.context_compressor.should_compress = lambda _tokens: False
    first = _run(agent, 17_000)
    second = _run(agent, 18_000)

    assert first.action == second.action == "fallthrough"
    agent._emit_status.assert_called_once()
    assert "prefill" in agent._emit_status.call_args.args[0].lower()


def test_prefill_warning_rearms_after_pressure_drops_below_warning():
    agent = _agent(28_000)
    agent.context_compressor.prefill_cost_cap.warn_tokens = 16_000
    agent.context_compressor.should_compress = lambda _tokens: False
    _run(agent, 17_000)
    _run(agent, 15_000)
    _run(agent, 17_000)
    assert agent._emit_status.call_count == 2
