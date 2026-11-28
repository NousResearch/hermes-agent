"""Regression for #158787: a VOICE follow-up delivered while the session is busy must run auto-TTS.

The busy follow-up observer delivers through ``session_ingress.deliver_response``, which used to
send text and attachments only. The adapter's own send path runs auto-TTS before the text, so a
sequential voice message was spoken and a queued one was silent.
"""

from types import SimpleNamespace

import pytest

from gateway.platforms import base
from gateway.session_ingress import deliver_response


class _Adapter:
    def __init__(self, *, wants_tts: bool = True, caption_carries_text: bool = False):
        self.wants_tts = wants_tts
        self.caption_carries_text = caption_carries_text
        self.calls: list[tuple] = []

    def _unwrap_ephemeral(self, response):
        return response, 0

    async def _extract_response_content(self, text, event, session_key, is_ephemeral_response=False):
        return SimpleNamespace(text_content=text, media_files=[], images=[], local_files=[])

    def _wants_auto_tts(self, event, session_key, interrupt_event, text_content, media_files):
        self.calls.append(("wants", text_content))
        return self.wants_tts

    async def _synthesize_auto_tts(self, text_content, source=None):
        self.calls.append(("synthesize", text_content))
        return ["/nonexistent/tts.wav"], None

    async def _play_tts_file(self, event, text_content, tts_path, first, metadata, record_delivery):
        self.calls.append(("play", tts_path, first))
        record_delivery(SimpleNamespace(success=True))
        return self.caption_carries_text

    async def _send_final_text(self, event, session_key, text, metadata, is_ephemeral, ttl, record_delivery):
        self.calls.append(("text", text))
        record_delivery(SimpleNamespace(success=True))

    async def _deliver_attachments(self, event, extracted, metadata, anything_sent, record_delivery):
        self.calls.append(("attachments", anything_sent))


@pytest.fixture(autouse=True)
def _no_thread_metadata(monkeypatch):
    monkeypatch.setattr(base, "_thread_metadata_for_event", lambda event: {})


def _event():
    return SimpleNamespace(source=SimpleNamespace(chat_id="chat-1"), message_id="m-1")


@pytest.mark.asyncio
async def test_busy_followup_runs_auto_tts_before_text():
    adapter = _Adapter(wants_tts=True)

    delivered = await deliver_response(adapter, _event(), "chat-1", "hello there")

    assert delivered is True
    names = [call[0] for call in adapter.calls]
    assert names == ["wants", "synthesize", "play", "text", "attachments"]


@pytest.mark.asyncio
async def test_busy_followup_without_tts_sends_text_only():
    adapter = _Adapter(wants_tts=False)

    await deliver_response(adapter, _event(), "chat-1", "hello there")

    names = [call[0] for call in adapter.calls]
    assert names == ["wants", "text", "attachments"]


@pytest.mark.asyncio
async def test_tts_caption_carries_the_text_instead_of_a_second_send():
    adapter = _Adapter(wants_tts=True, caption_carries_text=True)

    await deliver_response(adapter, _event(), "chat-1", "hello there")

    names = [call[0] for call in adapter.calls]
    assert names == ["wants", "synthesize", "play", "attachments"]
