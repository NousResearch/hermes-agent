"""A synchronous stop owns a transport failure before the monitor polls it."""
from unittest.mock import Mock

import httpx
import pytest

from agent.chat_completion_helpers import _StreamingCall
from run_agent import AIAgent


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
@pytest.mark.parametrize("error", [httpx.ReadError("socket shut down"), RuntimeError("streaming not supported")])
def test_stop_before_monitor_skips_stream_recovery(monkeypatch, api_mode, error):
    agent = AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:1/v1", model="test-model",
        provider="custom", quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[],
    )
    agent.api_mode = api_mode
    call = _StreamingCall(agent, {"model": "test-model"}, None)
    retry = Mock()
    disable_streaming = Mock()
    monkeypatch.setattr(call, "_retry_after_drop", retry)
    monkeypatch.setattr(call, "_maybe_disable_streaming", disable_streaming)

    def stopped_transport(attempt_id):
        # Eviction/STOP can abort the socket synchronously. The monitor has not
        # yet set its request-local cancellation flag when the worker unwinds.
        agent._interrupt_requested = True
        raise error

    monkeypatch.setattr(call, "_call_wire", stopped_transport)
    # Run the actual worker without a monitor: the race order is deterministic.
    call._call()

    retry.assert_not_called()
    disable_streaming.assert_not_called()
    assert call.result["response"] is None
    assert call.result["error"] is None


def test_transport_failure_without_stop_still_retries(monkeypatch):
    agent = AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:1/v1", model="test-model",
        provider="custom", quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[],
    )
    call = _StreamingCall(agent, {"model": "test-model"}, None)
    response = object()
    wire = Mock(side_effect=[httpx.ReadError("peer reset"), response])
    retry = Mock()
    monkeypatch.setattr(call, "_call_wire", wire)
    monkeypatch.setattr(call, "_retry_after_drop", retry)

    call._call()

    assert wire.call_count == 2
    retry.assert_called_once()
    assert call.result["response"] is response
    assert call.result["error"] is None
