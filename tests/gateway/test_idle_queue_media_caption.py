"""An idle ``/queue`` or ``/steer`` sent as the caption of a photo keeps the photo.

With no agent running, both strip the prefix and run the payload as a normal turn. A bare
caption (``/queue`` with a photo, no text) answered the usage hint and the photo never reached
the agent, while the busy ``/queue`` path accepts the same media-only message.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.test_queue_command import _make_runner, _make_source, _session_entry


async def _send(text, media):
    runner, _adapter = _make_runner(_session_entry())
    runner._handle_message_with_agent = AsyncMock(return_value="agent reply")
    event = MessageEvent(
        text=text, message_type=MessageType.PHOTO if media else MessageType.TEXT, source=_make_source(),
        message_id="m1", media_urls=["/tmp/photo.jpg"] if media else [],
        media_types=["image/jpeg"] if media else [])
    reply = await runner._handle_message(event)
    call = runner._handle_message_with_agent.await_args
    return reply, call.args[0] if call else None


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/queue", "/steer"])
async def test_bare_caption_on_a_photo_runs_the_photo(command):
    reply, sent = await _send(command, media=True)

    assert reply == "agent reply"
    assert sent.text == ""
    assert sent.media_urls == ["/tmp/photo.jpg"]
    assert sent.message_type == MessageType.PHOTO


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/queue", "/steer"])
async def test_bare_command_without_media_still_shows_usage(command):
    reply, sent = await _send(command, media=False)

    assert sent is None
    assert command in reply


@pytest.mark.asyncio
async def test_caption_with_text_still_sends_text_and_photo():
    reply, sent = await _send("/queue look at this", media=True)

    assert reply == "agent reply"
    assert (sent.text, sent.media_urls) == ("look at this", ["/tmp/photo.jpg"])
