from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import conversation_loop
from agent.chat_completion_helpers import interruptible_api_call
from agent.error_classifier import FailoverReason
from provider_gateway.config import GatewayConfig
from provider_gateway.circuit_breaker import CircuitState
from provider_gateway.policy import ProviderRouteCandidate
from provider_gateway.runtime import (
    _estimate_cost_usd,
    observe_gateway_route_selection,
    record_provider_error_usage,
    record_provider_response_usage,
)


class CapturingTracker:
    def __init__(self) -> None:
        self.records = []

    def record_usage(self, record):
        self.records.append(record)
        return len(self.records)


class FakeRequestClient:
    def __init__(self, responder) -> None:
        self._responder = responder
        self.closed = False
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        return self._responder(**kwargs)

    def close(self) -> None:
        self.closed = True


class FakeAgent:
    api_mode = "chat_completions"
    provider = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    model = "anthropic/claude-sonnet-4.6"
    api_key = ""
    session_id = "session-1"
    _interrupt_requested = False
    _base_url_lower = ""
    _base_url_hostname = ""

    def __init__(self, response_or_exc, tracker: CapturingTracker) -> None:
        self._response_or_exc = response_or_exc
        self._provider_gateway_config = GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        )
        self._provider_usage_tracker = tracker
        
        # Gunakan database cache sementara agar terisolasi dari database global
        import tempfile
        from pathlib import Path
        from provider_gateway.semantic_cache import SemanticCache
        self._temp_dir = tempfile.TemporaryDirectory()
        self._provider_semantic_cache = SemanticCache(db_path=Path(self._temp_dir.name) / "fake_agent_cache.db")

    def _create_request_openai_client(self, *, reason, api_kwargs):
        def responder(**kwargs):
            if isinstance(self._response_or_exc, Exception):
                raise self._response_or_exc
            return self._response_or_exc

        return FakeRequestClient(responder)

    def _close_request_openai_client(self, client, *, reason) -> None:
        client.close()

    def _abort_request_openai_client(self, client, *, reason) -> None:
        client.close()

    def _compute_non_stream_stale_timeout(self, api_payload) -> float:
        return 5.0

    def _touch_activity(self, message: str) -> None:
        pass


def _response_usage():
    return SimpleNamespace(
        prompt_tokens=13,
        completion_tokens=7,
        total_tokens=20,
        prompt_tokens_details=SimpleNamespace(cached_tokens=3),
    )


def test_response_usage_disabled_is_noop() -> None:
    tracker = CapturingTracker()
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(enabled=False),
        _provider_usage_tracker=tracker,
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        api_mode="chat_completions",
    )

    recorded = record_provider_response_usage(
        agent,
        SimpleNamespace(usage=_response_usage()),
        latency_seconds=1.25,
    )

    assert recorded is False
    assert tracker.records == []


def test_response_usage_records_when_enabled() -> None:
    tracker = CapturingTracker()
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        ),
        _provider_usage_tracker=tracker,
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        api_mode="chat_completions",
        session_id="session-1",
    )

    recorded = record_provider_response_usage(
        agent,
        SimpleNamespace(usage=_response_usage()),
        latency_seconds=1.25,
    )

    assert recorded is True
    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record.provider == "openrouter"
    assert record.model == "anthropic/claude-sonnet-4.6"
    assert record.api_mode == "chat_completions"
    assert record.input_tokens == 10
    assert record.cache_read_tokens == 3
    assert record.output_tokens == 7
    assert record.total_tokens == 20
    assert record.estimated_cost_usd == 0.0
    assert record.latency_ms == 1250.0
    assert record.status == "success"
    assert record.session_id == "session-1"


def test_error_usage_records_when_enabled() -> None:
    tracker = CapturingTracker()
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        ),
        _provider_usage_tracker=tracker,
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        api_mode="chat_completions",
        session_id="session-1",
    )

    recorded = record_provider_error_usage(
        agent,
        RuntimeError("rate limited"),
        latency_seconds=0.5,
    )

    assert recorded is True
    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record.status == "error"
    assert record.error_type == "RuntimeError"
    assert record.total_tokens == 0
    assert record.latency_ms == 500.0


def test_interruptible_api_call_records_success_when_gateway_enabled() -> None:
    tracker = CapturingTracker()
    response = SimpleNamespace(usage=_response_usage(), choices=[SimpleNamespace()])
    agent = FakeAgent(response, tracker)

    result = interruptible_api_call(agent, {"model": agent.model, "messages": []})

    assert result is response
    assert len(tracker.records) == 1
    assert tracker.records[0].status == "success"
    assert tracker.records[0].total_tokens == 20


def test_interruptible_api_call_records_error_when_gateway_enabled() -> None:
    tracker = CapturingTracker()
    agent = FakeAgent(ValueError("provider exploded"), tracker)

    with pytest.raises(ValueError, match="provider exploded"):
        interruptible_api_call(agent, {"model": agent.model, "messages": []})

    assert len(tracker.records) == 1
    assert tracker.records[0].status == "error"
    assert tracker.records[0].error_type == "ValueError"


def test_observe_route_selection_disabled_is_noop() -> None:
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(enabled=False),
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[{"provider": "openai", "model": "gpt-4o"}],
    )

    candidate = observe_gateway_route_selection(agent, FailoverReason.rate_limit)

    assert candidate is None
    assert not hasattr(agent, "_provider_gateway_last_route_candidate")


def test_observe_route_selection_records_candidate_without_mutating_route() -> None:
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(
            enabled=True,
            fallback_models=["openai/gpt-5.4"],
        ),
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[{"provider": "openai", "model": "gpt-4o"}],
    )

    candidate = observe_gateway_route_selection(agent, FailoverReason.rate_limit)

    assert candidate == ProviderRouteCandidate(
        provider="openrouter",
        model="openai/gpt-5.4",
        source="provider_gateway.routing.fallback_models",
        base_url="https://openrouter.ai/api/v1",
    )
    assert agent.provider == "openrouter"
    assert agent.model == "anthropic/claude-sonnet-4.6"
    assert agent._provider_gateway_last_route_candidate == candidate


def test_observe_route_selection_ignores_non_fallback_reason() -> None:
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(
            enabled=True,
            fallback_models=["openai/gpt-5.4"],
        ),
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        base_url="https://openrouter.ai/api/v1",
        _fallback_chain=[],
    )

    candidate = observe_gateway_route_selection(
        agent,
        FailoverReason.content_policy_blocked,
    )

    assert candidate is None
    assert not hasattr(agent, "_provider_gateway_last_route_candidate")


def test_conversation_loop_invokes_gateway_route_observation_after_classification() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "observe_gateway_route_selection" in source
    assert "classified.reason" in source


# --- New cost tracking tests below ---


def test_estimate_cost_returns_zero_when_track_cost_disabled() -> None:
    """Cost estimation should return 0.0 when track_cost is False."""
    config = GatewayConfig(enabled=True, track_cost=False)
    agent = SimpleNamespace(
        provider="anthropic",
        model="claude-sonnet-4-6",
        base_url="",
        api_key="",
    )
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    )

    cost = _estimate_cost_usd(agent, usage, config)

    assert cost == 0.0


def test_estimate_cost_returns_value_when_track_cost_enabled() -> None:
    """Cost estimation should return a positive value for known models."""
    config = GatewayConfig(enabled=True, track_cost=True)
    agent = SimpleNamespace(
        provider="anthropic",
        model="claude-sonnet-4-6",
        base_url="",
        api_key="",
    )
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        request_count=0,
    )

    cost = _estimate_cost_usd(agent, usage, config)

    # Anthropic claude-sonnet-4-6: $3/M input, $15/M output
    # 1000 input tokens = $0.003, 500 output tokens = $0.0075
    # Total: $0.0105
    assert cost > 0.0
    assert abs(cost - 0.0105) < 0.001


def test_estimate_cost_returns_zero_for_unknown_model() -> None:
    """Cost estimation should return 0.0 for models without pricing data."""
    config = GatewayConfig(enabled=True, track_cost=True)
    agent = SimpleNamespace(
        provider="unknown_provider_xyz",
        model="non_existent_model",
        base_url="",
        api_key="",
    )
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        request_count=0,
    )

    cost = _estimate_cost_usd(agent, usage, config)

    assert cost == 0.0


def test_response_usage_includes_cost_when_tracking_enabled() -> None:
    """Full pipeline test: record_provider_response_usage with cost tracking on."""
    tracker = CapturingTracker()
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=True,
        ),
        _provider_usage_tracker=tracker,
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_mode="chat_completions",
        base_url="",
        api_key="",
        session_id="cost-test-session",
    )

    # Anthropic-style usage fields (normalize_usage uses these when provider='anthropic')
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    response = SimpleNamespace(usage=usage)

    recorded = record_provider_response_usage(
        agent, response, latency_seconds=2.0
    )

    assert recorded is True
    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record.estimated_cost_usd > 0.0
    assert record.status == "success"
    assert record.input_tokens == 1000
    assert record.output_tokens == 500


def test_non_chat_completions_api_mode_is_not_tracked() -> None:
    """Usage tracking should be skipped for non-chat_completions api_mode."""
    tracker = CapturingTracker()
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(
            enabled=True,
            track_usage=True,
        ),
        _provider_usage_tracker=tracker,
        provider="anthropic",
        model="claude-opus-4.6",
        api_mode="anthropic_messages",
        session_id="session-1",
    )

    recorded = record_provider_response_usage(
        agent,
        SimpleNamespace(usage=_response_usage()),
        latency_seconds=1.0,
    )

    assert recorded is False
    assert tracker.records == []


def test_negative_latency_is_clamped_to_zero() -> None:
    """Negative latency values should be clamped to 0."""
    tracker = CapturingTracker()
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        ),
        _provider_usage_tracker=tracker,
        provider="openrouter",
        model="anthropic/claude-sonnet-4.6",
        api_mode="chat_completions",
        session_id="session-1",
    )

    record_provider_response_usage(
        agent,
        SimpleNamespace(usage=_response_usage()),
        latency_seconds=-5.0,
    )

    assert len(tracker.records) == 1
    assert tracker.records[0].latency_ms == 0.0


def test_runtime_updates_circuit_breaker() -> None:
    """Test that runtime success/error calls update the circuit breaker health."""
    from provider_gateway.runtime import get_circuit_breaker
    tracker = CapturingTracker()
    agent = SimpleNamespace(
        _provider_gateway_config=GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        ),
        _provider_usage_tracker=tracker,
        provider="test-circuit-provider",
        model="claude-sonnet",
        api_mode="chat_completions",
        session_id="session-1",
    )

    # Initially not tracked or CLOSED
    breaker = get_circuit_breaker(agent)
    assert breaker.is_available("test-circuit-provider") is True

    # Record success
    record_provider_response_usage(
        agent,
        SimpleNamespace(usage=_response_usage()),
        latency_seconds=0.15,
    )
    health = breaker.get_health("test-circuit-provider")
    assert health is not None
    assert health.state == CircuitState.CLOSED
    assert health.total_requests == 1
    assert health.latency_samples == [150.0]

    # Record failure
    record_provider_error_usage(
        agent,
        RuntimeError("API error"),
        latency_seconds=0.1,
    )
    assert health.total_requests == 2
    assert health.total_failures == 1
    assert health.consecutive_failures == 1

