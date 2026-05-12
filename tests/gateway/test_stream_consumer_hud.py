"""Stream consumer HUD-prefix integration tests.

The Telegram-only context HUD is wired into the consumer via
``StreamConsumerConfig.hud_prefix``.  These tests pin the contract:

- Outbound ``adapter.send`` / ``adapter.edit_message`` calls receive
  ``hud_prefix\\n\\nbody``.
- ``_last_sent_text`` stays HUD-free so the dedup short-circuit and
  continuation logic keep comparing raw model output.
- Setting ``hud_prefix=None`` is the legacy path — content unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _ok(message_id: str = "msg_1"):
    return SimpleNamespace(success=True, message_id=message_id, error=None)


def _fail(error: str = "boom"):
    return SimpleNamespace(success=False, message_id=None, error=error)


def _build_consumer(hud_prefix: str | None) -> tuple[GatewayStreamConsumer, MagicMock]:
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.SUPPORTS_MESSAGE_EDITING = True
    adapter.send = AsyncMock(return_value=_ok())
    adapter.edit_message = AsyncMock(return_value=_ok())
    adapter.truncate_message = MagicMock(side_effect=lambda t, n: [t])
    cfg = StreamConsumerConfig(hud_prefix=hud_prefix, cursor="")
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="c1", config=cfg)
    return consumer, adapter


@pytest.mark.asyncio
async def test_first_send_prepends_hud_prefix():
    consumer, adapter = _build_consumer("23k / 250k  9%\n[██░░░░░░░░░░░░░░░░░░]")
    body = "Hello from Hermes."
    ok = await consumer._send_or_edit(body)
    assert ok is True
    adapter.send.assert_awaited_once()
    sent_content = adapter.send.await_args.kwargs["content"]
    assert sent_content.startswith("23k / 250k  9%\n[██░░░░░░░░░░░░░░░░░░]\n\n")
    assert sent_content.endswith(body)


@pytest.mark.asyncio
async def test_no_hud_when_prefix_none():
    consumer, adapter = _build_consumer(None)
    body = "plain reply"
    await consumer._send_or_edit(body)
    sent_content = adapter.send.await_args.kwargs["content"]
    assert sent_content == body


@pytest.mark.asyncio
async def test_last_sent_text_is_hud_free():
    """The dedup short-circuit compares raw model output, not HUD-decorated bytes.

    If ``_last_sent_text`` accidentally captured the HUD prefix, the next
    edit with identical body text would not match and we would re-send.
    Worse, the continuation-prefix math would treat the HUD as part of the
    visible content.
    """
    consumer, adapter = _build_consumer("HUD-LINE")
    await consumer._send_or_edit("first text")
    assert consumer._last_sent_text == "first text"
    assert "HUD-LINE" not in consumer._last_sent_text


@pytest.mark.asyncio
async def test_edit_path_prepends_hud_prefix():
    consumer, adapter = _build_consumer("HUD")
    # First send establishes message id
    await consumer._send_or_edit("first")
    adapter.send.reset_mock()
    # Second call (now message_id is set) should edit, with HUD prefix
    await consumer._send_or_edit("first plus more")
    adapter.edit_message.assert_awaited()
    sent_content = adapter.edit_message.await_args.kwargs["content"]
    assert sent_content.startswith("HUD\n\n")
    assert sent_content.endswith("first plus more")


@pytest.mark.asyncio
async def test_empty_body_does_not_become_lone_hud():
    """An empty/whitespace body must not become a HUD-only message bubble."""
    consumer, adapter = _build_consumer("HUD")
    out = consumer._decorate_outbound("")
    assert out == ""
    out = consumer._decorate_outbound("   ")
    assert out == "   "
