"""A streamed turn's post-stream MEDIA delivery notifies like the non-streamed path (#144897).

The non-streamed chain marks its final content — text and attachments — with
``_mark_notify_metadata`` (``gateway/platforms/base.py``), so in ``important`` notification
mode the attachments ring. The streamed chain handed ``_deliver_media_from_response`` no
metadata at all: the function fell back to plain thread metadata, and every post-stream
``send_voice`` / ``send_document`` / image batch went out silent — a voice-only streamed
reply in ``important`` mode made no sound at all.
"""

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType, SessionSource
from gateway.run import GatewayRunner
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner

_SESSION_KEY = "agent:main:telegram:group:-1001:17585"


class _VoiceCaptureAdapter(ProgressCaptureAdapter):
    """Capture voice uploads so the metadata handed to send_voice is visible."""

    def __init__(self):
        super().__init__(platform=Platform.TELEGRAM)
        self.voices = []

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None,
                         metadata=None, is_voice=True, **kwargs) -> SendResult:
        self.voices.append({"path": audio_path, "metadata": metadata})
        return SendResult(success=True, message_id="voice-1")


async def _streamed_completion_seam(adapter, response):
    """Run an already-streamed result through ``_hmwa_deliver_turn_response`` — the seam that
    uploads the streamed reply's trailing MEDIA: files."""
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=adapter.platform, chat_id="-1001", chat_type="group", thread_id="17585",
    )
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")

    class _Entry:
        session_id = "s1"

    agent_result = {"already_sent": True}
    return await runner._hmwa_deliver_turn_response(
        event, source, _Entry(), _SESSION_KEY, None, agent_result, [], response, None, False,
    )


@pytest.mark.asyncio
async def test_streamed_media_delivery_carries_notify_metadata(tmp_path, monkeypatch):
    """In ``important`` notification mode the media completing a streamed turn must ring:
    the post-stream send carries notify metadata like the non-streamed attachments do."""
    from gateway.platforms import base as platforms_base

    voice = tmp_path / "note.ogg"
    voice.write_bytes(b"ogg")
    monkeypatch.setattr(platforms_base, "MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    adapter = _VoiceCaptureAdapter()

    result = await _streamed_completion_seam(
        adapter, f"here you go\n\nMEDIA: {voice} [[audio_as_voice]]")

    assert result is None  # the streamed body is not re-sent
    assert len(adapter.voices) == 1
    metadata = adapter.voices[0]["metadata"] or {}
    assert metadata.get("notify") is True

