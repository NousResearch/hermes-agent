"""``/voice tts all`` is a COMPOSITE delivery obligation on Telegram: the finalized text is only
released after its required voice half has landed (regression for #128151).

Before this, ``_send_voice_reply`` swallowed every failure (synthesis error, missing file,
refused ``send_voice``) and returned nothing, so ``_hmwa_deliver_turn_response`` handed the text
back to the adapter regardless. A chat pinned to ``all`` therefore received a finalized reply with
no audio whenever the configured TTS provider was unavailable.

Each test states the SAME invariant from one failure side: in exact ``all`` mode on Telegram the
seam releases the readable text if and only if the voice component was accepted. The ``off`` /
``voice_only`` rows pin that the obligation stays opt-in.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner

_SESSION_KEY = "agent:main:telegram:dm:42"
_FINAL = "The build is green on all three shards."


class _VoiceCaptureAdapter(ProgressCaptureAdapter):
    """Records the ORDER of the two halves of a composite delivery, and can refuse the voice half
    the way a real upload refusal arrives (a ``SendResult``, not an exception)."""

    def __init__(self, *, upload_ok: bool = True):
        super().__init__(platform=Platform.TELEGRAM)
        self.upload_ok = upload_ok
        self.voices: list[dict] = []

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None,
                         metadata=None, **kwargs) -> SendResult:
        self.voices.append({"chat_id": chat_id, "audio_path": audio_path, "reply_to": reply_to})
        if not self.upload_ok:
            return SendResult(success=False, error="voice upload rejected", retryable=True)
        return SendResult(success=True, message_id="voice-1")

    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        return False


def _tts_success(*_a, **_kw):
    def _impl(*, text, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake ogg opus")
        return json.dumps({"success": True, "file_path": output_path, "provider": "edge"})

    return _impl


def _tts_unavailable(*_a, **_kw):
    def _impl(*, text, output_path):
        return json.dumps({"success": False, "error": "no TTS provider configured"})

    return _impl


async def _seam(adapter, *, mode, chat_id="42", response=_FINAL):
    """Run one finalized turn through the completion seam and return the text the caller would
    send (None = nothing is released to the adapter)."""
    runner = _make_runner(adapter)
    runner._voice_mode = {f"telegram:{chat_id}": mode} if mode else {}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    event = MessageEvent(text="status?", message_type=MessageType.TEXT, source=source, message_id="7")

    class _Entry:
        session_id = "s1"

    return await runner._hmwa_deliver_turn_response(
        event, source, _Entry(), _SESSION_KEY, None, {}, [], response, None, False,
    )


@pytest.mark.asyncio
async def test_all_mode_releases_no_text_when_synthesis_fails():
    """Total synthesis failure: the configured provider is gone, so the required voice half never
    existed. Releasing the text anyway is exactly the reported symptom (the delivery looked usable
    while its required component was missing)."""
    adapter = _VoiceCaptureAdapter()

    with patch("tools.tts_tool.text_to_speech_tool", side_effect=_tts_unavailable()):
        released = await _seam(adapter, mode="all")

    assert released is None
    assert adapter.voices == []


@pytest.mark.asyncio
async def test_all_mode_releases_no_text_when_voice_upload_is_refused():
    """Synthesis succeeded but the platform refused the voice attachment. The audio never reached
    the chat, so the composite obligation is unmet even though a file existed locally -- the
    refusal was silently discarded, which let the text through as if the pair had been delivered."""
    adapter = _VoiceCaptureAdapter(upload_ok=False)

    with patch("tools.tts_tool.text_to_speech_tool", side_effect=_tts_success()):
        released = await _seam(adapter, mode="all")

    assert adapter.voices, "the upload must actually be attempted for the refusal to be visible"
    assert released is None


@pytest.mark.asyncio
async def test_all_mode_releases_text_only_after_the_voice_half_is_accepted():
    """The success half of the same invariant, and the guard against a fix that simply suppresses:
    when the voice component is accepted the text IS released -- after the audio, never before."""
    adapter = _VoiceCaptureAdapter()

    with patch("tools.tts_tool.text_to_speech_tool", side_effect=_tts_success()):
        released = await _seam(adapter, mode="all")

    assert released == _FINAL
    assert len(adapter.voices) == 1


@pytest.mark.asyncio
async def test_all_mode_still_releases_a_reply_with_nothing_to_speak():
    """A code-only reply normalizes to an empty script: there is no voice half to lose, so gating
    on it would make ``all`` mode swallow exactly the replies it cannot voice. Pinned so the gate
    cannot be tightened into a silence bug."""
    adapter = _VoiceCaptureAdapter()
    code_only = "```python\nimport hermes\nprint(hermes.__version__)\n```"

    with patch("tools.tts_tool.text_to_speech_tool", side_effect=_tts_success()):
        released = await _seam(adapter, mode="all", response=code_only)

    assert released == code_only
    assert adapter.voices == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "off", "voice_only"])
async def test_non_all_modes_keep_releasing_text_when_voice_is_unavailable(mode):
    """Isolation: the composite obligation belongs to exact ``all`` only. ``voice_only`` answers
    voice input (tested elsewhere) and must not be promoted to universal pairing, and no mode at all
    means the same best-effort behavior as before."""
    adapter = _VoiceCaptureAdapter()

    with patch("tools.tts_tool.text_to_speech_tool", side_effect=_tts_unavailable()):
        released = await _seam(adapter, mode=mode)

    assert released == _FINAL
    assert adapter.voices == []
