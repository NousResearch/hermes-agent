"""Regression tests for gateway interim commentary readability."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_adapter():
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
    adapter.MAX_MESSAGE_LENGTH = 4096
    return adapter


@pytest.mark.asyncio
async def test_commentary_send_defaults_to_dense_updates():
    """Dense mode preserves old single-newline rendering by default."""
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(adapter, "chat_123")

    await consumer._send_commentary(
        "I'll check the repo state.\n"
        "Source is clean and committed locally.\n"
        "Tests are green; I'm pushing now."
    )

    sent = adapter.send.call_args.kwargs["content"]
    assert sent == (
        "I'll check the repo state.\n"
        "Source is clean and committed locally.\n"
        "Tests are green; I'm pushing now."
    )


@pytest.mark.asyncio
async def test_commentary_send_spaces_batched_prose_updates_when_configured():
    """Spaced mode separates prose updates and adds a preserved bubble gap."""
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        config=StreamConsumerConfig(interim_assistant_spacing="spaced"),
    )

    await consumer._send_commentary(
        "I'll check the repo state.\n"
        "Memory says there were several POC commits.\n"
        "Source is clean and committed locally.\n"
        "Tests are green; I'm pushing now."
    )

    sent = adapter.send.call_args.kwargs["content"]
    assert sent == (
        "I'll check the repo state.\n\n"
        "Memory says there were several POC commits.\n\n"
        "Source is clean and committed locally.\n\n"
        "Tests are green; I'm pushing now.\n\u200b"
    )


@pytest.mark.asyncio
async def test_commentary_send_spaced_single_line_adds_preserved_message_gap():
    """Spaced mode gives separate single-line interim bubbles visual room."""
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        config=StreamConsumerConfig(interim_assistant_spacing="spaced"),
    )

    await consumer._send_commentary("I'm checking the gateway path.")

    sent = adapter.send.call_args.kwargs["content"]
    assert sent == "I'm checking the gateway path.\n\u200b"


@pytest.mark.asyncio
async def test_commentary_spacing_preserves_lists_and_code_blocks():
    """Readability spacing must not explode normal Markdown structures."""
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        config=StreamConsumerConfig(interim_assistant_spacing="spaced"),
    )

    await consumer._send_commentary(
        "Validation:\n"
        "- py_compile passed\n"
        "- pytest passed\n"
        "```bash\n"
        "python -m pytest tests/gateway -q\n"
        "```"
    )

    sent = adapter.send.call_args.kwargs["content"]
    assert sent == (
        "Validation:\n"
        "- py_compile passed\n"
        "- pytest passed\n"
        "```bash\n"
        "python -m pytest tests/gateway -q\n"
        "```\n\u200b"
    )
