from gateway.platforms.base import BasePlatformAdapter, MediaKind, classify_media_kind


def test_media_kind_has_four_members():
    assert {k.name for k in MediaKind} == {"IMAGE", "VIDEO", "VOICE", "DOCUMENT"}


def test_base_default_is_fail_closed_empty():
    assert BasePlatformAdapter.MEDIA_KINDS == frozenset()


def test_classify_image_video_document():
    assert classify_media_kind("/x/a.png", platform="qqbot") is MediaKind.IMAGE
    assert classify_media_kind("/x/a.mp4", platform="qqbot") is MediaKind.VIDEO
    assert classify_media_kind("/x/a.pdf", platform="qqbot") is MediaKind.DOCUMENT


def test_classify_audio_routes_to_voice_on_non_telegram():
    assert classify_media_kind("/x/a.mp3", platform="slack") is MediaKind.VOICE
    assert classify_media_kind("/x/a.ogg", is_voice=True, platform="slack") is MediaKind.VOICE


def test_classify_force_document_overrides_image():
    assert classify_media_kind("/x/a.png", platform="qqbot", force_document=True) is MediaKind.DOCUMENT
