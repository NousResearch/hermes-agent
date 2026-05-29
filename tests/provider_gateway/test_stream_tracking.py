import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import interruptible_streaming_api_call
from provider_gateway.config import GatewayConfig
from provider_gateway.runtime import get_circuit_breaker, get_provider_router


class CapturingTracker:
    def __init__(self) -> None:
        self.records = []

    def record_usage(self, record):
        self.records.append(record)
        return len(self.records)


class MockAgent:
    def __init__(self, tracker: CapturingTracker, temp_db_path: Path | None = None) -> None:
        self.provider = "openrouter"
        self.model = "anthropic/claude-sonnet-4.6"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_mode = "chat_completions"
        self.api_key = "key-123"
        self.session_id = "session-stream-test"
        self._provider_gateway_config = GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        )
        self._provider_usage_tracker = tracker
        self._interrupt_requested = False
        self._disable_streaming = False
        self._ollama_num_ctx = None
        self._rate_limited_until = 0.0
        
        # Gunakan database cache sementara agar terisolasi dari database global
        from provider_gateway.semantic_cache import SemanticCache
        from provider_gateway.circuit_breaker import CircuitBreaker
        self._provider_circuit_breaker = CircuitBreaker()
        if temp_db_path is not None:
            self._provider_semantic_cache = SemanticCache(db_path=temp_db_path)
        else:
            import tempfile
            self._temp_dir = tempfile.TemporaryDirectory()
            self._provider_semantic_cache = SemanticCache(db_path=Path(self._temp_dir.name) / "stream_mock_cache.db")

    def _touch_activity(self, text: str) -> None:
        pass

    def _compute_non_stream_stale_timeout(self, api_kwargs: dict) -> float:
        return 60.0

    def _has_stream_consumers(self) -> bool:
        return True

    def _reset_stream_delivery_tracking(self) -> None:
        pass

    def _fire_stream_delta(self, text: str) -> None:
        pass

    def _record_streamed_assistant_text(self, text: str) -> None:
        pass

    def _is_provider_stream_parse_error(self, exc: Exception) -> bool:
        return False

    def _stream_diag_init(self) -> dict:
        return {}

    def _stream_diag_capture_response(self, diag: dict, response: Any) -> None:
        pass

    def _capture_rate_limits(self, response: Any) -> None:
        pass

    def _check_openrouter_cache_status(self, response: Any) -> None:
        pass


@patch("agent.chat_completion_helpers.is_local_endpoint")
@patch("agent.chat_completion_helpers.get_provider_request_timeout")
@patch("agent.chat_completion_helpers.get_provider_stale_timeout")
def test_streaming_api_call_records_usage_successfully(
    mock_stale, mock_timeout, mock_local
) -> None:
    """Test that successful streaming completions write usage stats to the DB."""
    mock_local.return_value = False
    mock_timeout.return_value = None
    mock_stale.return_value = None

    tracker = CapturingTracker()
    agent = MockAgent(tracker)

    # Mock the request client
    mock_client = MagicMock()
    agent._create_request_openai_client = MagicMock(return_value=mock_client)
    agent._close_request_openai_client = MagicMock()

    # Mock the stream iterator return chunks
    chunk_usage = SimpleNamespace(
        prompt_tokens=15,
        completion_tokens=10,
        total_tokens=25,
    )
    chunk1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello ", tool_calls=None), finish_reason=None)])
    chunk2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world!", tool_calls=None), finish_reason=None)])
    # Final chunk contains usage
    chunk3 = SimpleNamespace(choices=[], usage=chunk_usage)

    mock_stream = [chunk1, chunk2, chunk3]
    mock_client.chat.completions.create.return_value = mock_stream

    # Reset circuit breaker
    breaker = get_circuit_breaker(agent)
    breaker.record_success("openrouter", latency_ms=0.0)

    # Execute
    res = interruptible_streaming_api_call(agent, {"messages": []})

    assert res is not None
    assert res.choices[0].message.content == "Hello world!"
    assert res.usage == chunk_usage

    # Verify usage was captured by the gateway
    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record.provider == "openrouter"
    assert record.model == "anthropic/claude-sonnet-4.6"
    assert record.status == "success"
    assert record.total_tokens == 25
    assert record.latency_ms > 0.0

    # Verify health updated
    health = breaker.get_health("openrouter")
    assert health.total_requests == 2
    assert health.consecutive_failures == 0


@patch("agent.chat_completion_helpers.is_local_endpoint")
@patch("agent.chat_completion_helpers.get_provider_request_timeout")
@patch("agent.chat_completion_helpers.get_provider_stale_timeout")
def test_streaming_api_call_records_error(
    mock_stale, mock_timeout, mock_local
) -> None:
    """Test that transient connection errors before deltas record to Circuit Breaker."""
    mock_local.return_value = False
    mock_timeout.return_value = None
    mock_stale.return_value = None

    tracker = CapturingTracker()
    agent = MockAgent(tracker)

    # Mock client raises error during create
    mock_client = MagicMock()
    agent._create_request_openai_client = MagicMock(return_value=mock_client)
    agent._close_request_openai_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Connection timed out")

    breaker = get_circuit_breaker(agent)
    breaker.record_success("openrouter", latency_ms=0.0)

    with pytest.raises(RuntimeError, match="Connection timed out"):
        interruptible_streaming_api_call(agent, {"messages": []})

    # Verify error was logged
    assert len(tracker.records) == 1
    record = tracker.records[0]
    assert record.provider == "openrouter"
    assert record.status == "error"
    assert record.error_type == "RuntimeError"

    # Verify Circuit Breaker was updated
    health = breaker.get_health("openrouter")
    assert health.total_requests == 2
    assert health.total_failures == 1
    assert health.consecutive_failures == 1
