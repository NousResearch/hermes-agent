"""Media-attachment finals still speak in bound VCs.

Regression context: a voice-input turn whose final reply carried a MEDIA attachment
(e.g. a steered turn finishing with a generated PDF) was silent - the base
``_wants_auto_tts`` gate requires ``not media_files``, so auto-TTS was skipped
entirely with no log and no fallback. The Discord adapter now allows streaming TTS
for voice-input finals with attachments when the chat is bound to a joined VC; the
base loop still delivers the attachment as a normal file send.
"""

import asyncio

from gateway.platforms.event import MessageEvent, MessageType


def _make_adapter(*, in_vc, voice=True, streamed=False):
    """A bare object bound to the REAL adapter methods under test, with stubbed edges."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = type("AdapterStub", (), {})()
    adapter._voice_fx_cfg = {"stream_replies": {"enabled": True}}
    adapter._voice_text_channels = {123: "456"}
    adapter._auto_tts_enabled_chats = {"456"}
    adapter._auto_tts_disabled_chats = set()
    adapter._auto_tts_default = False
    adapter._streaming_tts_completed_turns = set()
    adapter._vc_memberships = {123: in_vc}

    def fake_in_vc(gid):
        return adapter._vc_memberships[gid]

    adapter.is_in_voice_channel = fake_in_vc
    adapter._stream_tts_cfg = lambda: adapter._voice_fx_cfg["stream_replies"]
    adapter._should_auto_tts_for_chat = DiscordAdapter._should_auto_tts_for_chat.__get__(adapter)
    adapter._stream_tts_enabled = DiscordAdapter._stream_tts_enabled.__get__(adapter)
    adapter._streaming_tts_turn_completed = DiscordAdapter._streaming_tts_turn_completed.__get__(adapter)
    adapter._streaming_tts_turn_key = (
        lambda session_key, turn_marker, event=None: streaming_key(session_key, turn_marker, event))
    adapter._wants_auto_tts = DiscordAdapter._wants_auto_tts.__get__(adapter)

    event = MessageEvent(
        source=type("S", (), {"platform": "discord", "chat_id": "456"})(),
        text="hello", message_type=MessageType.VOICE if voice else MessageType.TEXT)
    interrupt = asyncio.Event()
    if streamed:
        interrupt._hermes_run_generation = 1
        adapter._streaming_tts_completed_turns.add(
            streaming_key("sk", 1, event))
    return adapter, event, "sk", interrupt


def streaming_key(session_key, turn_marker, event=None):
    from gateway.platforms.base import streaming_tts_turn_key
    return streaming_tts_turn_key(session_key, turn_marker, event=event)


def test_media_final_voice_input_vc_bound_still_tts():
    adapter, event, sk, interrupt = _make_adapter(in_vc=True)
    assert adapter._wants_auto_tts(event, sk, interrupt, "Here is the PDF.", ["a.pdf"]) is True


def test_media_final_typed_input_stays_silent():
    adapter, event, sk, interrupt = _make_adapter(in_vc=True, voice=False)
    assert adapter._wants_auto_tts(event, sk, interrupt, "Here is the PDF.", ["a.pdf"]) is False


def test_media_final_vc_not_joined_stays_silent():
    adapter, event, sk, interrupt = _make_adapter(in_vc=False)
    assert adapter._wants_auto_tts(event, sk, interrupt, "Here is the PDF.", ["a.pdf"]) is False


def test_media_final_voice_input_already_streamed_stays_silent():
    adapter, event, sk, interrupt = _make_adapter(in_vc=True, streamed=True)
    assert adapter._wants_auto_tts(event, sk, interrupt, "Here is the PDF.", ["a.pdf"]) is False


def test_no_media_delegates_to_base():
    adapter, event, sk, interrupt = _make_adapter(in_vc=False)
    # No media: base semantics untouched (voice + enabled chat -> True even unbound,
    # because base falls back to file synthesis).
    assert adapter._wants_auto_tts(event, sk, interrupt, "Plain reply.", []) is True
