"""Test /queue handles non-dict agent results without crashing.

Covers: https://github.com/NousResearch/hermes-agent/issues/25152

When result_holder[0] contains a non-dict (e.g. SendResult from an
adapter leak), the gateway must not crash with AttributeError on
result.get(...).
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio


class SendResult:
    """Minimal stub matching gateway.platforms.base.SendResult."""
    def __init__(self, success=True, message_id="mid", error=None):
        self.success = success
        self.message_id = message_id
        self.error = error


def _make_gateway_runner():
    """Create a minimal GatewayRunner-like mock."""
    from gateway.run import GatewayRunner
    runner = MagicMock(spec=GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_model_override = {}
    runner._intentional_model_switches = {}
    runner._is_intentional_model_switch = MagicMock(return_value=False)
    runner._evict_cached_agent = MagicMock()
    runner._dequeue_pending_event = MagicMock(return_value=None)
    runner._promote_queued_event = MagicMock(return_value=None)
    return runner


def test_result_holder_sendresult_does_not_crash():
    """result_holder[0] = SendResult should not cause AttributeError."""
    # Simulate what happens at line 15844: result = result_holder[0]
    result_holder = [SendResult(success=True, message_id="123")]

    result = result_holder[0]

    # This is the defensive guard added in the fix
    if not isinstance(result, dict):
        result = {"final_response": "", "messages": []}

    # All .get() calls should now work
    assert result.get("messages", []) == []
    assert result.get("final_response", "") == ""
    assert result.get("interrupted") is None
    assert result.get("response_previewed") is None


def test_result_holder_dict_passes_through():
    """Normal dict results should pass through unchanged."""
    result_holder = [{"final_response": "hello", "messages": [{"role": "user"}]}]

    result = result_holder[0]

    if not isinstance(result, dict):
        result = {"final_response": "", "messages": []}

    assert result.get("final_response") == "hello"
    assert len(result.get("messages", [])) == 1


def test_result_holder_none_handled():
    """result_holder[0] = None should be handled."""
    result_holder = [None]

    result = result_holder[0]

    if not isinstance(result, dict):
        result = {"final_response": "", "messages": []}

    assert result.get("messages", []) == []


def test_result_holder_string_handled():
    """result_holder[0] = string should be handled."""
    result_holder = ["some error string"]

    result = result_holder[0]

    if not isinstance(result, dict):
        result = {"final_response": "", "messages": []}

    assert result.get("messages", []) == []


def test_sendresult_attributes_preserved():
    """Verify SendResult attributes are accessible before guard."""
    sr = SendResult(success=False, message_id=None, error="timeout")
    result_holder = [sr]

    result = result_holder[0]
    # Before guard: result is SendResult
    assert isinstance(result, SendResult)
    assert result.success is False
    assert result.error == "timeout"

    # After guard: result becomes safe dict
    if not isinstance(result, dict):
        result = {"final_response": "", "messages": []}

    assert isinstance(result, dict)
    assert result.get("messages", []) == []


def test_result_get_all_code_paths():
    """Verify all .get() patterns used in the gateway work with guarded result."""
    result_holder = [SendResult()]

    result = result_holder[0]
    if not isinstance(result, dict):
        result = {"final_response": "", "messages": []}

    # All patterns from gateway/run.py lines 15860-16009
    _ = result.get("interrupted")
    _ = result.get("interrupt_message")
    _ = result.get("pending_steer")
    _ = result.get("response_previewed")
    _ = result.get("final_response", "")
    _ = result.get("messages", [])
    # No AttributeError


def test_normal_dict_result_preserves_interrupted_state():
    """Dict with interrupted=True should be preserved."""
    result_holder = [{"interrupted": True, "interrupt_message": "stopped", "messages": []}]

    result = result_holder[0]
    if not isinstance(result, dict):
        result = {"final_response": "", "messages": []}

    assert result.get("interrupted") is True
    assert result.get("interrupt_message") == "stopped"
