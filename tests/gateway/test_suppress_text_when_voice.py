"""Per-platform suppression of redundant text on voice replies.

Covers the ``PlatformConfig.suppress_text_when_voice`` flag and the
``_should_suppress_text_on_voice`` helper that decides when the written text
should be dropped in favor of a voice reply.
"""
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


# --- PlatformConfig.suppress_text_when_voice ------------------------------

def test_suppress_text_when_voice_default_is_false():
    assert PlatformConfig.from_dict({}).suppress_text_when_voice is False


def test_suppress_text_when_voice_parsed_from_top_level():
    pc = PlatformConfig.from_dict({"suppress_text_when_voice": True})
    assert pc.suppress_text_when_voice is True


def test_suppress_text_when_voice_parsed_from_extra():
    pc = PlatformConfig.from_dict({"extra": {"suppress_text_when_voice": True}})
    assert pc.suppress_text_when_voice is True


def test_suppress_text_when_voice_round_trips_through_to_dict():
    pc = PlatformConfig.from_dict({"suppress_text_when_voice": True})
    assert pc.to_dict()["suppress_text_when_voice"] is True


# --- _should_suppress_text_on_voice ---------------------------------------

def _adapter(suppress_text_when_voice):
    class _ConcreteAdapter(BasePlatformAdapter):
        async def connect(self):
            pass

        async def disconnect(self):
            pass

        async def get_chat_info(self, chat_id):
            return None

        async def send(self, chat_id, content, **kwargs):
            return True

    return _ConcreteAdapter(
        config=PlatformConfig.from_dict({"suppress_text_when_voice": suppress_text_when_voice}),
        platform=Platform.SIGNAL,
    )


def test_suppress_off_never_suppresses_even_with_voice():
    ad = _adapter(False)
    assert ad._should_suppress_text_on_voice(
        voice_delivered=True, media_files=[], images=[], local_files=[]
    ) is False


def test_suppress_on_with_delivered_voice_and_no_other_media():
    ad = _adapter(True)
    assert ad._should_suppress_text_on_voice(
        voice_delivered=True, media_files=[], images=[], local_files=[]
    ) is True


def test_suppress_on_with_queued_voice_media():
    ad = _adapter(True)
    assert ad._should_suppress_text_on_voice(
        voice_delivered=False, media_files=[("/tmp/voice.ogg", True)], images=[], local_files=[]
    ) is True


def test_suppress_on_with_non_voice_media_preserves_text():
    ad = _adapter(True)
    assert ad._should_suppress_text_on_voice(
        voice_delivered=True,
        media_files=[("/tmp/pic.png", False)],
        images=["pic.png"],
        local_files=[],
    ) is False


def test_suppress_on_with_image_preserves_text():
    ad = _adapter(True)
    assert ad._should_suppress_text_on_voice(
        voice_delivered=True, media_files=[], images=["photo.png"], local_files=[]
    ) is False


def test_suppress_on_with_local_file_preserves_text():
    ad = _adapter(True)
    assert ad._should_suppress_text_on_voice(
        voice_delivered=True, media_files=[], images=[], local_files=["/tmp/doc.pdf"]
    ) is False


def test_suppress_on_but_no_voice_delivered_or_queued():
    ad = _adapter(True)
    assert ad._should_suppress_text_on_voice(
        voice_delivered=False, media_files=[], images=[], local_files=[]
    ) is False
