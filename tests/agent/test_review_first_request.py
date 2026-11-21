"""First-review admission when the compressor cannot detach safely."""
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.background_review import _detach_fork_compression
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression


@pytest.mark.parametrize("tokens, action", [(150, "fallthrough"), (1000, "return")])
def test_oversized_first_review_is_not_sent_when_compression_unavailable(tokens, action):
    compressor = SimpleNamespace(
        threshold_tokens=100, context_length=200,
        get_active_compression_failure_cooldown=lambda: None,
    )
    agent = SimpleNamespace(
        _review_defer_compaction_before_first_response=True,
        _turn_received_provider_response=False, compression_enabled=False,
        context_compressor=compressor, _persist_session=Mock(), _flush_status_buffer=Mock(),
        log_prefix="", model="test/model",
    )
    # This is the actual failed-detachment path, not an artificially set flag.
    agent._review_defer_compaction_before_first_response = False
    _detach_fork_compression(agent)
    state = {field.name: None for field in fields(PreflightGateVerdict)}
    state.update(messages=[{"role": "user", "content": "too large"}],
                 compression_attempts=0, api_call_count=0)
    result = run_preflight_compression(
        agent, PreflightGateVerdict(**state), compressor=compressor, request_pressure_tokens=tokens,
        provider_overflow_preflight=False, defer_preflight=lambda _: False,
        moa_prepared_request=None, system_message=None, user_message="review",
        max_compression_attempts=3, effective_task_id=None,
    )
    assert result.action == action
    if action == "return":
        assert result.result["compression_exhausted"]
