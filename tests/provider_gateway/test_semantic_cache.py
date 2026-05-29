import time
import uuid
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import interruptible_api_call, interruptible_streaming_api_call
from provider_gateway.config import GatewayConfig
from provider_gateway.semantic_cache import SemanticCache


class CapturingTracker:
    def __init__(self) -> None:
        self.records = []

    def record_usage(self, record):
        self.records.append(record)
        return len(self.records)


class MockAgent:
    def __init__(self, tracker: CapturingTracker, temp_db_path: Path) -> None:
        self.provider = "openrouter"
        self.model = "anthropic/claude-sonnet-4.6"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_mode = "chat_completions"
        self.api_key = "key-123"
        self.session_id = "session-cache-test"
        self._provider_gateway_config = GatewayConfig(
            enabled=True,
            track_usage=True,
            track_cost=False,
        )
        self._provider_usage_tracker = tracker
        self._provider_semantic_cache = SemanticCache(db_path=temp_db_path)
        self._interrupt_requested = False
        self._disable_streaming = False
        self._ollama_num_ctx = None
        self._rate_limited_until = 0.0
        self.verbose_logging = False

        # Mock stream-related tracking
        self.deltas = []

    def _touch_activity(self, text: str) -> None:
        pass

    def _compute_non_stream_stale_timeout(self, api_kwargs: dict) -> float:
        return 60.0

    def _has_stream_consumers(self) -> bool:
        return True

    def _reset_stream_delivery_tracking(self) -> None:
        pass

    def _fire_stream_delta(self, text: str) -> None:
        self.deltas.append(text)

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
        db_path = Path(tmpdir) / "test_cache.db"
        yield db_path


def test_semantic_cache_hash_consistency(temp_db) -> None:
    cache = SemanticCache(db_path=temp_db)
    messages = [
        {"role": "user", "content": "Halo, siapa kamu?"},
        {"role": "assistant", "content": "Saya Hermes Agent."},
    ]
    h1 = cache.compute_hash(messages)
    h2 = cache.compute_hash(messages)
    assert h1 == h2

    # Different content produces different hash
    messages_diff = [
        {"role": "user", "content": "Halo, siapa kamu?"},
        {"role": "assistant", "content": "Saya adalah Hermes Agent."},
    ]
    h3 = cache.compute_hash(messages_diff)
    assert h1 != h3


@patch("agent.chat_completion_helpers.is_local_endpoint")
@patch("agent.chat_completion_helpers.get_provider_request_timeout")
@patch("agent.chat_completion_helpers.get_provider_stale_timeout")
def test_semantic_cache_basic_miss_and_hit(
    mock_stale, mock_timeout, mock_local, temp_db
) -> None:
    """Test that a cache miss records to the cache, and a subsequent identical request hits."""
    mock_local.return_value = False
    mock_timeout.return_value = None
    mock_stale.return_value = None

    tracker = CapturingTracker()
    agent = MockAgent(tracker, temp_db)

    # 1. Mock API call behavior for cache miss
    mock_client = MagicMock()
    agent._create_request_openai_client = MagicMock(return_value=mock_client)
    agent._close_request_openai_client = MagicMock()

    mock_msg = SimpleNamespace(role="assistant", content="Jawaban pertama.")
    mock_choice = SimpleNamespace(message=mock_msg, finish_reason="stop")
    mock_response = SimpleNamespace(
        id="resp-123",
        model="anthropic/claude-sonnet-4.6",
        choices=[mock_choice],
        usage=None,
    )
    mock_client.chat.completions.create.return_value = mock_response

    # First call (Miss)
    messages = [{"role": "user", "content": "Pertanyaan unik"}]
    res1 = interruptible_api_call(agent, {"messages": messages})
    assert res1.choices[0].message.content == "Jawaban pertama."
    assert mock_client.chat.completions.create.call_count == 1

    # Second call (Hit)
    # We alter the mock return just in case it reaches the client, but it shouldn't!
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        id="resp-456",
        model="anthropic/claude-sonnet-4.6",
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Harusnya tidak terpanggil"))],
        usage=None,
    )

    res2 = interruptible_api_call(agent, {"messages": messages})
    assert res2.choices[0].message.content == "Jawaban pertama."
    # Client create count should STILL be 1, because it hit the cache!
    assert mock_client.chat.completions.create.call_count == 1
    assert res2.id.startswith("cache-")


@patch("agent.chat_completion_helpers.is_local_endpoint")
@patch("agent.chat_completion_helpers.get_provider_request_timeout")
@patch("agent.chat_completion_helpers.get_provider_stale_timeout")
def test_semantic_cache_disabled_by_config(
    mock_stale, mock_timeout, mock_local, temp_db
) -> None:
    """Test that cache is bypassed entirely when the gateway config is disabled."""
    mock_local.return_value = False
    mock_timeout.return_value = None
    mock_stale.return_value = None

    import dataclasses
    tracker = CapturingTracker()
    agent = MockAgent(tracker, temp_db)
    agent._provider_gateway_config = dataclasses.replace(agent._provider_gateway_config, enabled=False)

    mock_client = MagicMock()
    agent._create_request_openai_client = MagicMock(return_value=mock_client)
    agent._close_request_openai_client = MagicMock()

    # Pre-populate cache directly
    cache = agent._provider_semantic_cache
    messages = [{"role": "user", "content": "Bypass test"}]
    cache.set_cached_response(agent, messages, "Sudah di-cache.")

    # Call API
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        id="direct-resp",
        model="anthropic/claude-sonnet-4.6",
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Respon langsung API"))],
        usage=None,
    )

    res = interruptible_api_call(agent, {"messages": messages})
    assert res.choices[0].message.content == "Respon langsung API"
    assert mock_client.chat.completions.create.call_count == 1


@patch("agent.chat_completion_helpers.is_local_endpoint")
@patch("agent.chat_completion_helpers.get_provider_request_timeout")
@patch("agent.chat_completion_helpers.get_provider_stale_timeout")
def test_semantic_cache_streaming_hit(
    mock_stale, mock_timeout, mock_local, temp_db
) -> None:
    """Test that a cache hit on streaming triggers on_first_delta, fires stream deltas, and returns mock response."""
    mock_local.return_value = False
    mock_timeout.return_value = None
    mock_stale.return_value = None

    tracker = CapturingTracker()
    agent = MockAgent(tracker, temp_db)

    # Directly store a response in cache first
    messages = [{"role": "user", "content": "Bagaimana cuaca?"}]
    cache = agent._provider_semantic_cache
    cache.set_cached_response(agent, messages, "Cuaca sangat cerah.")

    # Mock first delta callback
    first_delta_called = {"yes": False}
    def on_first_delta():
        first_delta_called["yes"] = True

    # Call streaming API call
    res = interruptible_streaming_api_call(agent, {"messages": messages}, on_first_delta=on_first_delta)

    # Verification
    assert res is not None
    assert res.choices[0].message.content == "Cuaca sangat cerah."
    assert res.id.startswith("cache-")
    assert first_delta_called["yes"] is True
    # The whole content was streamed in a single delta fire
    assert agent.deltas == ["Cuaca sangat cerah."]
