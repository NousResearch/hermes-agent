"""Regression coverage for the request-bound public interim-text seam.

Codex Responses can complete a tool round without a public ``commentary``
phase (#142663).  The generic emitter is intentionally text-only: provider
adapters may publish a safe status without exposing reasoning or tool details.
"""

from typing import Any
from unittest.mock import MagicMock

from agent.stream_delivery import StreamDeliveryMixin


def _agent() -> Any:
    agent: Any = StreamDeliveryMixin()
    agent.interim_assistant_callback = MagicMock()
    agent._enqueue_stream_hook = MagicMock()
    agent._strip_think_blocks = lambda text: text
    agent._current_api_request_id = "request-1"
    agent._hold_stream = False
    return agent


def test_emit_interim_text_delivers_only_for_the_active_request():
    agent = _agent()

    assert agent.emit_interim_text(
        "Working on it.", request_id="request-1"
    ) is True
    agent.interim_assistant_callback.assert_called_once_with(
        "Working on it.", already_streamed=False
    )

    assert agent.emit_interim_text(
        "Stale status.", request_id="request-0"
    ) is False
    assert agent.interim_assistant_callback.call_count == 1


def test_emit_interim_text_strips_private_thinking_before_delivery():
    agent = _agent()
    agent._strip_think_blocks = lambda text: text.replace("<think>secret</think>", "")

    assert agent.emit_interim_text(
        "<think>secret</think>Working on it.", request_id="request-1"
    ) is True
    agent.interim_assistant_callback.assert_called_once_with(
        "Working on it.", already_streamed=False
    )
