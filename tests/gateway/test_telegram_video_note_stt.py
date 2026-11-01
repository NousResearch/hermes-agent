"""Telegram round video notes reuse the existing STT path.

Round video messages are short voice-like clips with visual context: they stay
video attachments for visual analysis, and their mp4 audio track is also
eligible for the normal transcription pipeline. Regression for #54606.
"""

from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import _event_media_is_stt_input
from gateway.run_inbound import GatewayInboundMixin


def _video_event(metadata=None, media_type="video/mp4"):
    return MessageEvent(
        text="",
        message_type=MessageType.VIDEO,
        media_urls=["/tmp/clip.mp4"],
        media_types=[media_type],
        metadata=metadata or {},
    )


def test_telegram_video_note_video_attachment_is_stt_input():
    assert _event_media_is_stt_input(_video_event({"telegram_video_note": True}), 0) is True


def test_plain_video_attachment_is_not_automatic_stt_input():
    assert _event_media_is_stt_input(_video_event(), 0) is False


def test_video_note_flag_without_video_media_type_is_not_stt_input():
    """The marker alone must not send a non-video attachment to transcription."""
    event = _video_event({"telegram_video_note": True}, media_type="image/jpeg")
    assert _event_media_is_stt_input(event, 0) is False


def test_video_note_is_classified_as_both_video_and_stt_audio():
    """The inbound classifier must route a round video note to vision AND transcription."""
    event = _video_event({"telegram_video_note": True})

    image_paths, audio_paths, audio_file_paths, video_paths = (
        GatewayInboundMixin._classify_inbound_media(event, False)
    )

    assert image_paths == []
    assert audio_paths == ["/tmp/clip.mp4"]
    assert audio_file_paths == []
    assert video_paths == ["/tmp/clip.mp4"]


def test_plain_video_is_classified_as_video_only():
    event = _video_event()

    image_paths, audio_paths, audio_file_paths, video_paths = (
        GatewayInboundMixin._classify_inbound_media(event, False)
    )

    assert image_paths == []
    assert audio_paths == []
    assert audio_file_paths == []
    assert video_paths == ["/tmp/clip.mp4"]


def test_prepared_pending_video_note_skips_second_stt_pass():
    """A pending event already transcribed must not be transcribed again by the classifier."""
    event = _video_event({"telegram_video_note": True})

    _, audio_paths, _, video_paths = GatewayInboundMixin._classify_inbound_media(event, True)

    assert audio_paths == []
    assert video_paths == ["/tmp/clip.mp4"]
