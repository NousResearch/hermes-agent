"""Regression tests for Element voice bubbles on Matrix (MSC3245).

Element only draws an inline audio player for an ``m.audio`` event carrying
the MSC3245 ``org.matrix.msc3245.voice`` flag. That flag is set only when
``is_voice=True`` reaches ``MatrixAdapter._upload_and_send``, and ``is_voice``
is derived from an ``[[audio_as_voice]]`` directive in the agent's response
text.

An agent that synthesises its own audio and emits its own ``MEDIA:`` tag never
got that directive (``GatewayRunner`` only auto-appends it when the response
does *not* already contain a ``MEDIA:`` tag), so its voice notes arrived as
file cards. Ogg/Opus is exactly the container MSC3245 voice messages use, so
the adapter no longer depends on the caller having inferred intent.
"""

import asyncio
from types import SimpleNamespace

import pytest


@pytest.fixture
def adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter.__new__(MatrixAdapter)


@pytest.mark.parametrize(
    "audio_path,caller_is_voice,expected_is_voice",
    [
        # The regression: the gateway could not infer intent, but the
        # container is unambiguous, so the flag must still be set.
        ("/tmp/agent_reply.ogg", False, True),
        ("/tmp/agent_reply.oga", False, True),
        ("/tmp/agent_reply.opus", False, True),
        ("/tmp/AGENT_REPLY.OGG", False, True),        # case-insensitive
        # Explicit intent is preserved.
        ("/tmp/tts_reply.ogg", True, True),
        # Non-Ogg input still honours the caller, so a plain audio attachment
        # is not silently turned into a voice note.
        ("/tmp/song.mp3", False, False),
        ("/tmp/song.wav", False, False),
    ],
)
def test_ogg_audio_is_flagged_as_voice(
    adapter, monkeypatch, audio_path, caller_is_voice, expected_is_voice
):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    captured = {}

    async def fake_send_local_file(
        self, chat_id, path, msgtype, caption=None, reply_to=None,
        file_name=None, metadata=None, is_voice=False, **kwargs
    ):
        captured["msgtype"] = msgtype
        captured["is_voice"] = is_voice
        return SimpleNamespace(success=True, error=None)

    monkeypatch.setattr(MatrixAdapter, "_send_local_file", fake_send_local_file)
    # Non-Ogg input would otherwise shell out to ffmpeg to transcode.
    monkeypatch.setattr(
        "plugins.platforms.matrix.adapter._matrix_transcode_voice_to_ogg",
        lambda _path: None,
    )

    asyncio.run(
        adapter.send_voice(
            chat_id="!room:example.org",
            audio_path=audio_path,
            is_voice=caller_is_voice,
        )
    )

    assert captured["msgtype"] == "m.audio"
    assert captured["is_voice"] is expected_is_voice


@pytest.mark.parametrize("is_voice,expect_flag", [(True, True), (False, False)])
def test_msc3245_flag_tracks_is_voice(adapter, is_voice, expect_flag):
    """The emitted event carries the voice flag iff is_voice is set.

    Pinned in both directions: without the flag Element renders a file card,
    and if it were emitted unconditionally every audio attachment would become
    a voice note.
    """
    sent = {}

    class _FakeClient:
        crypto = None
        state_store = None

        async def upload_media(self, *args, **kwargs):
            return "mxc://example.org/abc123"

        async def send_message_event(self, room_id, event_type, content, **kwargs):
            sent["content"] = content
            return "$event123"

    adapter._client = _FakeClient()
    adapter._encryption = False
    adapter._max_media_bytes = 10 * 1024 * 1024
    adapter._apply_relation_metadata = lambda content, reply_to=None, metadata=None: None

    asyncio.run(
        adapter._upload_and_send(
            room_id="!room:example.org",
            data=b"\x00" * 512,
            filename="reply.ogg",
            content_type="audio/ogg",
            msgtype="m.audio",
            is_voice=is_voice,
            voice_metadata={"duration": 4200, "waveform": [0] * 30} if is_voice else None,
        )
    )

    content = sent["content"]
    assert content["msgtype"] == "m.audio"
    assert ("org.matrix.msc3245.voice" in content) is expect_flag
    if expect_flag:
        # Element uses these to draw the scrubber; absent them the bubble
        # renders but cannot be seeked.
        assert content["info"]["duration"] == 4200
        assert len(content["org.matrix.msc1767.audio"]["waveform"]) == 30


def test_extract_media_marks_ogg_as_voice_with_directive():
    """The directive is what ultimately drives is_voice end to end."""
    from gateway.platforms.base import BasePlatformAdapter

    content = "Here you go.\n[[audio_as_voice]]\nMEDIA:/tmp/reply.ogg"
    media, cleaned = BasePlatformAdapter.extract_media(content)

    assert len(media) == 1
    path, is_voice = media[0]
    assert path.endswith("/tmp/reply.ogg")
    assert is_voice is True
    assert "[[audio_as_voice]]" not in cleaned
    assert "MEDIA:" not in cleaned


def test_extract_media_without_directive_is_not_voice():
    """Without the directive the file is a plain attachment -- the exact state
    an agent-authored ``MEDIA:`` tag used to produce."""
    from gateway.platforms.base import BasePlatformAdapter

    media, _cleaned = BasePlatformAdapter.extract_media(
        "Here you go.\nMEDIA:/tmp/reply.ogg"
    )

    assert len(media) == 1
    _path, is_voice = media[0]
    assert is_voice is False
