"""Telegram notification-mode gating (#132516): "important" mode silences ordinary sends, but a
human-decision prompt (``is_approval_prompt``) must always push — a silently delivered prompt is
indistinguishable from "no prompt" and costs the full approvals.timeout before the command is refused.
"""

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.parametrize("mode,metadata,expected", [
    ("important", None, {"disable_notification": True}),
    ("important", {"thread_id": "t1"}, {"disable_notification": True}),
    ("important", {"notify": True}, {}),
    ("important", {"thread_id": "t1", "is_approval_prompt": True}, {}),
    ("all", None, {}),
    ("all", {"is_approval_prompt": True}, {}),
])
def test_notification_kwargs(mode, metadata, expected):
    adapter = object.__new__(TelegramAdapter)
    adapter._notifications_mode = mode
    assert adapter._notification_kwargs(metadata) == expected


def test_clarify_text_fallback_pushes_in_important_mode():
    """A failed native clarify card retries as base numbered text via adapter.send(); that
    human-decision prompt must not go out with disable_notification=True."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from gateway.config import PlatformConfig
    from gateway.run_turn_runner_clarify_delivery import text_fallback_coro
    from tools import clarify_gateway

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._bot, adapter._app = AsyncMock(), MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    adapter._notifications_mode = "important"

    # The fallback only sends for a still-pending prompt, as when the runner retries a failed card.
    clarify_gateway.register("c1", "s", "Which env?", None)
    try:
        asyncio.run(text_fallback_coro(
            adapter, chat_id="123", question="Which env?", choices=None, clarify_id="c1", session_key="s", metadata=None))
    finally:
        clarify_gateway.clear_session("s")

    assert adapter._bot.send_message.await_args.kwargs.get("disable_notification") is None
