"""Regression tests for the safe Codex public-status fallback."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from agent.i18n import t
from agent.stream_delivery import StreamDeliveryMixin
from agent.tool_executor import _ToolCallRef, _begin_tool_execution


def _agent() -> Any:
    agent = StreamDeliveryMixin()
    agent.interim_assistant_callback = MagicMock()
    agent._enqueue_stream_hook = MagicMock()
    agent._strip_think_blocks = lambda text: text
    agent._current_api_request_id = "request-1"
    agent._hold_stream = False
    agent._public_status_fallback_sent = False
    return agent


def test_public_status_fallback_is_once_per_turn_and_request_bound():
    agent = _agent()

    assert agent.emit_public_status_fallback(request_id="request-1") is True
    assert agent.emit_public_status_fallback(request_id="request-1") is False
    assert agent.emit_public_status_fallback(request_id="stale") is False
    agent.interim_assistant_callback.assert_called_once_with(
        t("cli.stream.public_status_fallback"),
        already_streamed=False,
    )


def test_public_status_fallback_yields_to_real_commentary():
    agent = _agent()
    agent._delivered_interim_texts = {"real commentary"}

    assert agent.emit_public_status_fallback(request_id="request-1") is False
    agent.interim_assistant_callback.assert_not_called()


def test_public_status_fallback_does_not_publish_without_interim_callback():
    agent = _agent()
    agent.interim_assistant_callback = None

    assert agent.emit_public_status_fallback(request_id="request-1") is False
    assert agent._public_status_fallback_sent is False


def test_codex_responses_emits_fallback_at_tool_boundary():
    emit_status = MagicMock()
    agent = SimpleNamespace(
        api_mode="codex_responses",
        _current_api_request_id="request-1",
        emit_public_status_fallback=emit_status,
        _checkpoint_mgr=SimpleNamespace(enabled=False),
        quiet_mode=True,
        tool_progress_mode="off",
        verbose_logging=False,
        tool_progress_callback=None,
        tool_start_callback=None,
        _touch_activity=MagicMock(),
    )

    _begin_tool_execution(agent, _ToolCallRef("terminal", {}, "task-1", "call-1", []), None)

    emit_status.assert_called_once_with(request_id="request-1")


def test_codex_app_server_does_not_emit_responses_fallback():
    emit_status = MagicMock()
    agent = SimpleNamespace(
        api_mode="codex_app_server",
        _current_api_request_id="request-1",
        emit_public_status_fallback=emit_status,
        _checkpoint_mgr=SimpleNamespace(enabled=False),
        quiet_mode=True,
        tool_progress_mode="off",
        verbose_logging=False,
        tool_progress_callback=None,
        tool_start_callback=None,
        _touch_activity=MagicMock(),
    )

    _begin_tool_execution(agent, _ToolCallRef("terminal", {}, "task-1", "call-1", []), None)

    emit_status.assert_not_called()
