"""Security tests for Matrix text batching."""

import asyncio
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_adapter():
    from gateway.platforms.matrix import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter.handle_message = AsyncMock()
    return adapter


def _make_event(user_id: str, text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        chat_type="group",
        user_id=user_id,
        thread_id="$thread:example.org",
    )
    return MessageEvent(text=text, source=source, message_id=f"${user_id}")


def test_matrix_text_batching_does_not_merge_different_senders():
    """Text batching must not aggregate prompts under another sender identity."""
    asyncio.run(_assert_text_batching_keeps_senders_separate())


async def _assert_text_batching_keeps_senders_separate():
    adapter = _make_adapter()

    alice = _make_event("@alice:example.org", "authorized prompt")
    mallory = _make_event("@mallory:example.org", "attacker prompt")

    assert adapter._text_batch_key(alice) != adapter._text_batch_key(mallory)

    adapter._enqueue_text_event(alice)
    adapter._enqueue_text_event(mallory)

    assert len(adapter._pending_text_batches) == 2
    assert alice.text == "authorized prompt"
    assert mallory.text == "attacker prompt"

    await asyncio.sleep(0.05)

    handled = adapter.handle_message.await_args_list
    assert len(handled) == 2
    handled_by_user = {
        call.args[0].source.user_id: call.args[0].text for call in handled
    }
    assert handled_by_user == {
        "@alice:example.org": "authorized prompt",
        "@mallory:example.org": "attacker prompt",
    }
