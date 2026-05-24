from types import SimpleNamespace

import pytest

pytest.importorskip("lark_oapi.channel")

from gateway.platforms.base import MessageType
from gateway.platforms.feishu.events_mapping import to_message_event


@pytest.mark.asyncio
async def test_post_placeholder_text_does_not_infer_resources_when_sdk_resources_empty(tmp_path):
    from lark_oapi.channel.types import Conversation, Identity, InboundMessage, PostContent

    download_called = False

    class Channel:
        bot_identity = SimpleNamespace(open_id="ou_bot")

        async def get_chat_info(self, chat_id):
            return SimpleNamespace(chat_id=chat_id, name="DM", chat_type="p2p")

        async def download_resource_to_file(
            self, file_key, *, resource_type, message_id, dest_dir, file_name=None
        ):
            nonlocal download_called
            download_called = True
            raise AssertionError("Hermes should rely on SDK resources, not placeholder text")

    msg = InboundMessage(
        id="om_test",
        create_time=0,
        conversation=Conversation(chat_id="oc_dm", chat_type="p2p"),
        sender=Identity(open_id="ou_user", display_name="User"),
        content=PostContent(post={}, raw={}),
        raw={},
        content_text="[Image: img_v3_test]\nE2E caption",
        resources=[],
    )

    event = await to_message_event(msg, channel=Channel())

    assert event.message_type == MessageType.TEXT
    assert event.media_urls == []
    assert event.media_types == []
    assert download_called is False
    assert "E2E caption" in event.text


@pytest.mark.asyncio
async def test_post_raw_payload_is_not_reparsed_when_sdk_flattening_empty(tmp_path):
    from lark_oapi.channel.types import Conversation, Identity, InboundMessage, PostContent

    post = {
        "content": [[
            {"tag": "img", "image_key": "img_v3_raw_post"},
            {"tag": "text", "text": "E2E raw caption"},
        ]]
    }
    download_called = False

    class Channel:
        bot_identity = SimpleNamespace(open_id="ou_bot")

        async def get_chat_info(self, chat_id):
            return SimpleNamespace(chat_id=chat_id, name="DM", chat_type="p2p")

        async def download_resource_to_file(
            self, file_key, *, resource_type, message_id, dest_dir, file_name=None
        ):
            nonlocal download_called
            download_called = True
            raise AssertionError("Hermes should not reparse raw post payloads")

    msg = InboundMessage(
        id="om_raw_post",
        create_time=0,
        conversation=Conversation(chat_id="oc_dm", chat_type="p2p"),
        sender=Identity(open_id="ou_user", display_name="User"),
        content=PostContent(post=post, raw=post),
        raw={"body": {"content": post}},
        content_text="",
        resources=[],
        raw_content_type="post",
    )

    event = await to_message_event(msg, channel=Channel())

    assert event.message_type == MessageType.TEXT
    assert event.media_urls == []
    assert event.media_types == []
    assert download_called is False
    assert "E2E raw caption" not in event.text


@pytest.mark.asyncio
async def test_audio_resource_falls_back_to_file_resource_type(tmp_path):
    from lark_oapi.channel.types import AudioContent, Conversation, Identity, InboundMessage

    cached_path = tmp_path / "voice.opus"
    attempts = []

    class Channel:
        bot_identity = SimpleNamespace(open_id="ou_bot")

        async def get_chat_info(self, chat_id):
            return SimpleNamespace(chat_id=chat_id, name="DM", chat_type="p2p")

        async def download_resource_to_file(
            self, file_key, *, resource_type, message_id, dest_dir, file_name=None
        ):
            attempts.append(resource_type)
            assert file_key == "file_audio_key"
            assert message_id == "om_audio"
            if resource_type == "audio":
                raise RuntimeError("download failed for audio resource_type")
            assert resource_type == "file"
            return cached_path

    msg = InboundMessage(
        id="om_audio",
        create_time=0,
        conversation=Conversation(chat_id="oc_dm", chat_type="p2p"),
        sender=Identity(open_id="ou_user", display_name="User"),
        content=AudioContent(file_key="file_audio_key", raw={"file_key": "file_audio_key"}),
        raw={},
        content_text="",
        resources=[SimpleNamespace(type="audio", file_key="file_audio_key", file_name=None)],
    )

    event = await to_message_event(msg, channel=Channel())

    assert attempts == ["audio", "file"]
    assert event.media_urls == [str(cached_path)]
    assert event.media_types[0].startswith("audio/")


@pytest.mark.asyncio
async def test_audio_resource_fallback_preserves_audio_extension(tmp_path):
    from lark_oapi.channel.types import AudioContent, Conversation, Identity, InboundMessage

    attempts = []
    requested_file_names = []

    class Channel:
        bot_identity = SimpleNamespace(open_id="ou_bot")

        async def get_chat_info(self, chat_id):
            return SimpleNamespace(chat_id=chat_id, name="DM", chat_type="p2p")

        async def download_resource_to_file(
            self, file_key, *, resource_type, message_id, dest_dir, file_name=None
        ):
            attempts.append(resource_type)
            requested_file_names.append(file_name)
            assert file_key == "file_audio_key"
            assert message_id == "om_audio"
            if resource_type == "audio":
                raise RuntimeError("download failed for audio resource_type")
            assert resource_type == "file"
            path = tmp_path / (file_name or f"{file_key}.bin")
            path.write_bytes(b"OggS mock opus payload")
            return path

    msg = InboundMessage(
        id="om_audio",
        create_time=0,
        conversation=Conversation(chat_id="oc_dm", chat_type="p2p"),
        sender=Identity(open_id="ou_user", display_name="User"),
        content=AudioContent(file_key="file_audio_key", raw={"file_key": "file_audio_key"}),
        raw={},
        content_text="",
        resources=[SimpleNamespace(type="audio", file_key="file_audio_key", file_name=None)],
    )

    event = await to_message_event(msg, channel=Channel())

    assert attempts == ["audio", "file"]
    assert requested_file_names == ["file_audio_key.ogg", "file_audio_key.ogg"]
    assert event.media_urls == [str(tmp_path / "file_audio_key.ogg")]
    assert event.media_types == ["audio/ogg"]


@pytest.mark.asyncio
async def test_file_resource_media_type_uses_filename_mime(tmp_path):
    from lark_oapi.channel.types import Conversation, FileContent, Identity, InboundMessage

    cached_path = tmp_path / "notes.txt"

    class Channel:
        bot_identity = SimpleNamespace(open_id="ou_bot")

        async def get_chat_info(self, chat_id):
            return SimpleNamespace(chat_id=chat_id, name="DM", chat_type="p2p")

        async def download_resource_to_file(
            self, file_key, *, resource_type, message_id, dest_dir, file_name=None
        ):
            assert file_key == "file_key_1"
            assert resource_type == "file"
            assert message_id == "om_file"
            assert file_name == "notes.txt"
            return cached_path

    msg = InboundMessage(
        id="om_file",
        create_time=0,
        conversation=Conversation(chat_id="oc_dm", chat_type="p2p"),
        sender=Identity(open_id="ou_user", display_name="User"),
        content=FileContent(file_key="file_key_1", file_name="notes.txt", raw={}),
        raw={},
        content_text="",
        resources=[SimpleNamespace(type="file", file_key="file_key_1", file_name="notes.txt")],
    )

    event = await to_message_event(msg, channel=Channel())

    assert event.message_type == MessageType.DOCUMENT
    assert event.media_urls == [str(cached_path)]
    assert event.media_types == ["text/plain"]


@pytest.mark.asyncio
async def test_pdf_file_resource_preserves_filename_and_pdf_mime(tmp_path):
    from lark_oapi.channel.types import Conversation, FileContent, Identity, InboundMessage

    cached_path = tmp_path / "report.pdf"

    class Channel:
        bot_identity = SimpleNamespace(open_id="ou_bot")

        async def get_chat_info(self, chat_id):
            return SimpleNamespace(chat_id=chat_id, name="DM", chat_type="p2p")

        async def download_resource_to_file(
            self, file_key, *, resource_type, message_id, dest_dir, file_name=None
        ):
            assert file_key == "file_pdf_key"
            assert resource_type == "file"
            assert message_id == "om_pdf"
            assert file_name == "report.pdf"
            cached_path.write_bytes(b"%PDF-1.7\n")
            return cached_path

    msg = InboundMessage(
        id="om_pdf",
        create_time=0,
        conversation=Conversation(chat_id="oc_dm", chat_type="p2p"),
        sender=Identity(open_id="ou_user", display_name="User"),
        content=FileContent(file_key="file_pdf_key", file_name="report.pdf", raw={}),
        raw={},
        content_text="",
        resources=[SimpleNamespace(type="file", file_key="file_pdf_key", file_name="report.pdf")],
    )

    event = await to_message_event(msg, channel=Channel())

    assert event.message_type == MessageType.DOCUMENT
    assert event.media_urls == [str(cached_path)]
    assert event.media_types == ["application/pdf"]


@pytest.mark.asyncio
async def test_pdf_file_resource_uses_content_filename_when_resource_lacks_name(tmp_path):
    from lark_oapi.channel.types import Conversation, FileContent, Identity, InboundMessage

    cached_path = tmp_path / "report.pdf"

    class Channel:
        bot_identity = SimpleNamespace(open_id="ou_bot")

        async def get_chat_info(self, chat_id):
            return SimpleNamespace(chat_id=chat_id, name="DM", chat_type="p2p")

        async def download_resource_to_file(
            self, file_key, *, resource_type, message_id, dest_dir, file_name=None
        ):
            assert file_key == "file_pdf_key"
            assert resource_type == "file"
            assert message_id == "om_pdf"
            assert file_name == "report.pdf"
            cached_path.write_bytes(b"%PDF-1.7\n")
            return cached_path

    msg = InboundMessage(
        id="om_pdf",
        create_time=0,
        conversation=Conversation(chat_id="oc_dm", chat_type="p2p"),
        sender=Identity(open_id="ou_user", display_name="User"),
        content=FileContent(file_key="file_pdf_key", file_name="report.pdf", raw={}),
        raw={},
        content_text="",
        resources=[SimpleNamespace(type="file", file_key="file_pdf_key", file_name=None)],
    )

    event = await to_message_event(msg, channel=Channel())

    assert event.message_type == MessageType.DOCUMENT
    assert event.media_urls == [str(cached_path)]
    assert event.media_types == ["application/pdf"]


@pytest.mark.asyncio
async def test_video_resource_falls_back_to_file_resource_type(tmp_path):
    from lark_oapi.channel.types import Conversation, Identity, InboundMessage, MediaContent

    attempts = []
    requested_file_names = []
    cached_path = tmp_path / "clip.mp4"

    class Channel:
        bot_identity = SimpleNamespace(open_id="ou_bot")

        async def get_chat_info(self, chat_id):
            return SimpleNamespace(chat_id=chat_id, name="DM", chat_type="p2p")

        async def download_resource_to_file(
            self, file_key, *, resource_type, message_id, dest_dir, file_name=None
        ):
            attempts.append(resource_type)
            requested_file_names.append(file_name)
            assert file_key == "video_file_key"
            assert message_id == "om_video"
            if resource_type == "video":
                raise RuntimeError("download failed for video resource_type")
            assert resource_type == "file"
            cached_path.write_bytes(b"mock mp4 payload")
            return cached_path

    msg = InboundMessage(
        id="om_video",
        create_time=0,
        conversation=Conversation(chat_id="oc_dm", chat_type="p2p"),
        sender=Identity(open_id="ou_user", display_name="User"),
        content=MediaContent(file_key="video_file_key", file_name="clip.mp4", raw={}),
        raw={},
        content_text="",
        resources=[SimpleNamespace(type="video", file_key="video_file_key", file_name="clip.mp4")],
    )

    event = await to_message_event(msg, channel=Channel())

    assert attempts == ["video", "file"]
    assert requested_file_names == ["clip.mp4", "clip.mp4"]
    assert event.message_type == MessageType.VIDEO
    assert event.media_urls == [str(cached_path)]
    assert event.media_types == ["video/mp4"]
