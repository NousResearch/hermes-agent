"""Regression test: /compress works with context engine plugins.

Reported by @selfhostedsoul (Discord, Apr 2026) with the LCM plugin installed:

    Compression failed: 'LCMEngine' object has no attribute '_align_boundary_forward'

Root cause: the gateway /compress handler used to reach into
ContextCompressor-specific private helpers (_align_boundary_forward,
_find_tail_cut_by_tokens) for its preflight check.  Those helpers are not
part of the generic ContextEngine ABC, so any plugin engine (LCM, etc.)
raised AttributeError.

The fix promotes the preflight into an optional ABC method
(has_content_to_compress) with a safe default of True.
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from agent.context_engine import ContextEngine
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


class _FakePluginEngine(ContextEngine):
    """Minimal ContextEngine that only implements the ABC — no private helpers.

    Mirrors the shape of a third-party context engine plugin such as LCM.
    If /compress reaches into any ContextCompressor-specific internals this
    engine will raise AttributeError, just like the real bug.
    """

    @property
    def name(self) -> str:
        return "fake-plugin"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        return None

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        # Pretend we dropped a middle turn.
        self.compression_count += 1
        if len(messages) >= 3:
            return [messages[0], messages[-1]]
        return list(messages)


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
async def test_compress_works_with_plugin_context_engine():
    """/compress must not call ContextCompressor-only private helpers.

    Uses a fake ContextEngine subclass that only implements the ABC —
    matches what a real plugin (LCM, etc.) exposes. If the gateway
    reaches into ``_align_boundary_forward`` or ``_find_tail_cut_by_tokens``
    on this engine, AttributeError propagates and the test fails with the
    exact user-visible error selfhostedsoul reported.
    """
    history = _make_history()
    compressed = [history[0], history[-1]]
    runner = _make_runner(history)

    plugin_engine = _FakePluginEngine()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    # Real plugin engine — no MagicMock auto-attributes masking missing helpers.
    agent_instance.context_compressor = plugin_engine
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=100),
    ):
        result = await runner._handle_compress_command(_make_event("/compress"))

    # No AttributeError surfaced as "Compression failed: ..."
    assert "Compression failed" not in result
    assert "_align_boundary_forward" not in result
    assert "_find_tail_cut_by_tokens" not in result
    # Happy path fired
    agent_instance._compress_context.assert_called_once()


@pytest.mark.asyncio
async def test_compress_respects_plugin_has_content_to_compress_false():
    """If a plugin reports no compressible content, gateway skips the LLM call."""

    class _EmptyEngine(_FakePluginEngine):
        def has_content_to_compress(self, messages):
            return False

    history = _make_history()
    runner = _make_runner(history)

    plugin_engine = _EmptyEngine()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance.context_compressor = plugin_engine
    agent_instance.session_id = "sess-1"

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=100),
    ):
        result = await runner._handle_compress_command(_make_event("/compress"))

    assert "Nothing to compress" in result
    agent_instance._compress_context.assert_not_called()
