"""Regression: gateway /compress must wire SessionDB into the throwaway agent.

Without ``session_db``, ``compress_context()`` cannot rotate the session or
persist the compacted transcript (#44794). The slash command still reports
in-memory message/token deltas, but the next turn reloads the oversized
history — so Telegram runtime footers (and auto-compression) look stuck.
"""

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
        session_id="sess-parent",
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
    runner._session_db = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()
    runner._session_key_for_source = lambda source: build_session_key(source)
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=("test-model", {"api_key": "test-key"}),
    )
    return runner, session_entry


@pytest.mark.asyncio
async def test_compress_passes_gateway_session_db_to_throwaway_agent():
    history = _make_history()
    runner, _entry = _make_runner(history)
    captured: dict = {}

    class _Agent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.context_compressor = MagicMock()
            self.context_compressor.has_content_to_compress.return_value = True
            self.session_id = "sess-parent"
            self.compression_in_place = False
            self._cached_system_prompt = ""
            self.tools = None

        def _compress_context(self, *args, **kwargs):
            return (history[:2], "")

    with (
        patch("run_agent.AIAgent", _Agent),
        patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            side_effect=lambda msgs, **_kw: 100 if len(msgs) >= 4 else 40,
        ),
    ):
        await runner._handle_compress_command(_make_event())

    assert captured.get("session_db") is runner._session_db


@pytest.mark.asyncio
async def test_compress_rewrites_transcript_when_session_rotates():
    history = _make_history()
    compressed = [history[0], history[-1]]
    runner, _entry = _make_runner(history)

    agent_instance = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = False
    agent_instance.session_id = "sess-parent"

    def _rotate(*_a, **_kw):
        agent_instance.session_id = "sess-child"
        return (compressed, "")

    agent_instance._compress_context.side_effect = _rotate

    with patch("run_agent.AIAgent", return_value=agent_instance), patch(
        "agent.model_metadata.estimate_request_tokens_rough",
        side_effect=lambda msgs, **_kw: 100 if len(msgs) >= 4 else 40,
    ):
        await runner._handle_compress_command(_make_event())

    runner.session_store.rewrite_transcript.assert_called_once_with(
        "sess-child", compressed,
    )


@pytest.mark.asyncio
async def test_compress_skips_rewrite_when_rotation_and_in_place_both_fail():
    """#44794 guard: do not overwrite the live transcript on a failed rotate."""
    history = _make_history()
    runner, _entry = _make_runner(history)

    agent_instance = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = False
    agent_instance.session_id = "sess-parent"
    agent_instance._compress_context.return_value = (history[:2], "")

    with patch("run_agent.AIAgent", return_value=agent_instance), patch(
        "agent.model_metadata.estimate_request_tokens_rough",
        side_effect=lambda msgs, **_kw: 100 if len(msgs) >= 4 else 40,
    ):
        await runner._handle_compress_command(_make_event())

    runner.session_store.rewrite_transcript.assert_not_called()
