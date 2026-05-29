import time
import tempfile
import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import interruptible_api_call, interruptible_streaming_api_call
from provider_gateway.config import GatewayConfig
from provider_gateway.quota_manager import QuotaManager, QuotaExceededError
from provider_gateway.usage_tracker import ProviderUsageRecord, ProviderUsageTracker


class CapturingTracker:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.records = []

    def record_usage(self, record: ProviderUsageRecord) -> int:
        self.records.append(record)
        # Manually persist records into SQLite for budget queries to consume
        tracker = ProviderUsageTracker(db_path=self.db_path)
        tracker.record_usage(record)
        return len(self.records)


class MockAgent:
    def __init__(self, tracker: CapturingTracker, temp_db_path: Path) -> None:
        self.provider = "openrouter"
        self.model = "anthropic/claude-sonnet-4.6"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_mode = "chat_completions"
        self.api_key = "key-123"
        self.session_id = "session-quota-test"
        self._provider_gateway_config = GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=True,
            daily_limit_usd=0.05,
            monthly_limit_usd=1.00,
            quota_action="block",
            fallback_models=["llama3-free"],
        )
        self._provider_usage_tracker = tracker
        self._provider_quota_manager = QuotaManager(db_path=temp_db_path)
        self._interrupt_requested = False
        self._disable_streaming = False
        self._ollama_num_ctx = None
        self._rate_limited_until = 0.0
        self.verbose_logging = False
        
        # Gunakan database cache sementara agar terisolasi dari database global
        from provider_gateway.semantic_cache import SemanticCache
        self._provider_semantic_cache = SemanticCache(db_path=temp_db_path)

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


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_quota.db"
        yield db_path


def test_quota_manager_spend_calculations(temp_db) -> None:
    tracker = ProviderUsageTracker(db_path=temp_db)
    manager = QuotaManager(db_path=temp_db)

    # Clean check
    assert manager.get_daily_spend() == 0.0
    assert manager.get_monthly_spend() == 0.0

    # Insert historical spend records
    # 1. Record for today
    r1 = ProviderUsageRecord(
        provider="openrouter",
        model="claude",
        api_mode="chat_completions",
        input_tokens=100,
        output_tokens=100,
        total_tokens=200,
        estimated_cost_usd=0.02,
        latency_ms=100,
        status="success",
        session_id="s1",
    )
    tracker.record_usage(r1)

    # 2. Another record for today
    r2 = ProviderUsageRecord(
        provider="openrouter",
        model="claude",
        api_mode="chat_completions",
        input_tokens=100,
        output_tokens=100,
        total_tokens=200,
        estimated_cost_usd=0.015,
        latency_ms=100,
        status="success",
        session_id="s1",
    )
    tracker.record_usage(r2)

    assert manager.get_daily_spend() == 0.035
    assert manager.get_monthly_spend() == 0.035


@patch("agent.chat_completion_helpers.is_local_endpoint")
@patch("agent.chat_completion_helpers.get_provider_request_timeout")
@patch("agent.chat_completion_helpers.get_provider_stale_timeout")
def test_quota_manager_within_limits(
    mock_stale, mock_timeout, mock_local, temp_db
) -> None:
    """Test that a request goes through fine when spend is within limits."""
    mock_local.return_value = False
    mock_timeout.return_value = None
    mock_stale.return_value = None

    tracker = CapturingTracker(temp_db)
    agent = MockAgent(tracker, temp_db)

    mock_client = MagicMock()
    agent._create_request_openai_client = MagicMock(return_value=mock_client)
    agent._close_request_openai_client = MagicMock()

    mock_client.chat.completions.create.return_value = SimpleNamespace(
        id="resp-ok",
        model="anthropic/claude-sonnet-4.6",
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Respon aman"))],
        usage=None,
    )

    res = interruptible_api_call(agent, {"messages": []})
    assert res.choices[0].message.content == "Respon aman"
    assert mock_client.chat.completions.create.call_count == 1


@patch("agent.chat_completion_helpers.is_local_endpoint")
@patch("agent.chat_completion_helpers.get_provider_request_timeout")
@patch("agent.chat_completion_helpers.get_provider_stale_timeout")
def test_quota_manager_blocks_on_exceeded(
    mock_stale, mock_timeout, mock_local, temp_db
) -> None:
    """Test that a request is blocked and raises QuotaExceededError when limits are exceeded."""
    mock_local.return_value = False
    mock_timeout.return_value = None
    mock_stale.return_value = None

    tracker = CapturingTracker(temp_db)
    agent = MockAgent(tracker, temp_db)

    # 1. Pre-fill spend database beyond daily_limit (daily_limit is 0.05)
    r = ProviderUsageRecord(
        provider="openrouter",
        model="claude",
        api_mode="chat_completions",
        input_tokens=1000,
        output_tokens=1000,
        total_tokens=2000,
        estimated_cost_usd=0.06,  # Exceeded daily limit!
        latency_ms=100,
        status="success",
        session_id="s1",
    )
    tracker.record_usage(r)

    # 2. Try calling API. It should be blocked.
    mock_client = MagicMock()
    agent._create_request_openai_client = MagicMock(return_value=mock_client)

    with pytest.raises(QuotaExceededError, match="Daily budget limit exceeded"):
        interruptible_api_call(agent, {"messages": []})

    # Client should NOT have been called
    assert mock_client.chat.completions.create.call_count == 0


@patch("agent.chat_completion_helpers.is_local_endpoint")
@patch("agent.chat_completion_helpers.get_provider_request_timeout")
@patch("agent.chat_completion_helpers.get_provider_stale_timeout")
def test_quota_manager_fallback_on_exceeded(
    mock_stale, mock_timeout, mock_local, temp_db
) -> None:
    """Test that exceed limit switches provider and model to local Ollama when action is fallback."""
    mock_local.return_value = True  # local endpoint
    mock_timeout.return_value = None
    mock_stale.return_value = None

    tracker = CapturingTracker(temp_db)
    agent = MockAgent(tracker, temp_db)
    # Set action to fallback
    agent._provider_gateway_config = GatewayConfig(
        enabled=True,
        track_usage=True,
        track_cost=True,
        daily_limit_usd=0.05,
        quota_action="fallback",
        fallback_models=["llama3-free"],
    )

    # 1. Pre-fill spend beyond limits
    r = ProviderUsageRecord(
        provider="openrouter",
        model="claude",
        api_mode="chat_completions",
        input_tokens=1000,
        output_tokens=1000,
        total_tokens=2000,
        estimated_cost_usd=0.06,  # Exceeded limit
        latency_ms=100,
        status="success",
        session_id="s1",
    )
    tracker.record_usage(r)

    # 2. Call API. It should fallback to local Ollama and make the call.
    mock_client = MagicMock()
    agent._create_request_openai_client = MagicMock(return_value=mock_client)
    agent._close_request_openai_client = MagicMock()

    mock_client.chat.completions.create.return_value = SimpleNamespace(
        id="resp-ollama",
        model="llama3-free",
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Respon dari Ollama lokal"))],
        usage=None,
    )

    res = interruptible_api_call(agent, {"messages": []})

    # Assert agent state was rerouted
    assert agent.provider == "ollama"
    assert agent.model == "llama3-free"
    assert agent.base_url == "http://localhost:11434/v1"
    assert agent.api_key == "ollama"

    assert res.choices[0].message.content == "Respon dari Ollama lokal"
    assert mock_client.chat.completions.create.call_count == 1
