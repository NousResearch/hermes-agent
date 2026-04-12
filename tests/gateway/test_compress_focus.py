"""Tests for gateway /compress <focus> — focus topic on the gateway side."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/compress") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _make_runner(history: list[dict[str, str]]):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._save = MagicMock()
    return runner


@pytest.mark.asyncio
async def test_compress_focus_topic_passed_to_agent():
    """Focus topic from /compress <focus> is passed through to _compress_context."""
    history = _make_history()
    compressed = [history[0], history[-1]]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.context_compressor.protect_first_n = 0
    agent_instance.context_compressor._align_boundary_forward.return_value = 0
    agent_instance.context_compressor._find_tail_cut_by_tokens.return_value = 2
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    def _estimate(messages):
        return 100

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event("/compress database schema"))

    # Verify focus_topic was passed
    agent_instance._compress_context.assert_called_once()
    call_kwargs = agent_instance._compress_context.call_args
    assert call_kwargs.kwargs.get("focus_topic") == "database schema"

    # Verify focus is mentioned in response
    assert 'Focus: "database schema"' in result


@pytest.mark.asyncio
async def test_compress_no_focus_passes_none():
    """Bare /compress passes focus_topic=None."""
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.context_compressor.protect_first_n = 0
    agent_instance.context_compressor._align_boundary_forward.return_value = 0
    agent_instance.context_compressor._find_tail_cut_by_tokens.return_value = 2
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=100),
    ):
        result = await runner._handle_compress_command(_make_event("/compress"))

    agent_instance._compress_context.assert_called_once()
    call_kwargs = agent_instance._compress_context.call_args
    assert call_kwargs.kwargs.get("focus_topic") is None

    # No focus line in response
    assert "Focus:" not in result
