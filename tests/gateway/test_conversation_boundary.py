"""Transport-independent adapter interception and normalized conversation lanes."""
import pytest
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.conversation import ConversationKey
from tests.gateway.test_conversation_policy import Adapter, event, plugin_discovery


@pytest.fixture
def anyio_backend():
    return "asyncio"


class NativeAdapter(Adapter):
    """No Groupchat imports/decorators in the native transport methods."""
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="sent-1")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.sent.append((chat_id, content, finalize))
        return SendResult(success=True, message_id=message_id)

    async def send_draft(self, chat_id, draft_id, content, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True)

    async def send_document(self, chat_id, file_path, caption=None, metadata=None):
        self.sent.append((chat_id, file_path, caption))
        return SendResult(success=True, message_id="document-1")


@pytest.mark.anyio
@pytest.mark.parametrize("platform", [Platform.MATRIX, Platform.MATTERMOST, Platform.DISCORD, Platform.TELEGRAM, Platform.SLACK])
async def test_all_adapters_filter_text_edits_and_suppress_drafts(platform):
    adapter = NativeAdapter(platform, {"pingpong_guard": {"enabled": True}})
    assert adapter.conversation_middleware().buffers_output
    assert (await adapter.send("room", "ok")).success
    assert (await adapter.edit_message("room", "existing", "understood", finalize=True)).success
    assert (await adapter.send_draft("room", 1, "not yet approved")).success
    assert adapter.sent == []
    text = "A substantial complete answer that is longer than the configured filter threshold."
    assert (await adapter.send("room", text)).success
    assert adapter.sent[0][1] == text
    await adapter.disconnect()


@pytest.mark.anyio
async def test_filtered_caption_does_not_drop_attachment():
    adapter = NativeAdapter(Platform.MATTERMOST, {"pingpong_guard": {"enabled": True}})
    await adapter.send_document("room", "/tmp/report.pdf", caption="ok")
    assert adapter.sent == [("room", "/tmp/report.pdf", "")]
    await adapter.disconnect()


@pytest.mark.anyio
async def test_matrix_threads_share_room_context_but_keep_distinct_buffers():
    adapter = NativeAdapter(Platform.MATRIX, {"relevance": {"enabled": True}})
    a, b = event(Platform.MATRIX, thread="thread-a"), event(Platform.MATRIX, thread="thread-b")
    await adapter.handle_message(a)
    await adapter.handle_message(b)
    root = adapter.conversation_policy()
    first = root.for_conversation(a.source.chat_id, ConversationKey.from_source(a.source).metadata())
    second = root.for_conversation(b.source.chat_id, ConversationKey.from_source(b.source).metadata())
    assert first is not second and first is not root
    assert first.relevance._room_transcript is second.relevance._room_transcript
    assert first.relevance._pending is not second.relevance._pending
    assert first.last_inbound[a.source.chat_id] == a.text
    assert second.last_inbound[b.source.chat_id] == b.text
    await adapter.disconnect()


@pytest.mark.anyio
async def test_edit_recovers_original_normalized_thread():
    adapter = NativeAdapter(Platform.MATRIX, {"pingpong_guard": {"enabled": True, "min_chars": 1}})
    key = ConversationKey("matrix", "room", "workspace", "thread")
    await adapter.send("room", "substantive", metadata=key.metadata())
    root = adapter.conversation_policy()
    lane = root.for_conversation("room", key.metadata())
    from unittest.mock import AsyncMock
    lane.filter_output = AsyncMock(return_value=None)
    from gateway.conversation import _active_conversation
    token = _active_conversation.set(ConversationKey("matrix", "room", "workspace", "other-thread"))
    try:
        await adapter.edit_message("room", "sent-1", "another reply", finalize=True)
    finally:
        _active_conversation.reset(token)
    lane.filter_output.assert_awaited_once()
    assert len(adapter.sent) == 1
    await adapter.disconnect()


def test_platform_options_are_discovered_not_three_fixed_channels():
    from plugins.groupchat.config import available_platforms, validate_settings
    options = available_platforms()
    assert {"matrix", "mattermost", "discord", "irc", "teams"} <= set(options)
    assert validate_settings({"platforms": ["mattermost", "irc"]})["platforms"] == ["mattermost", "irc"]


@pytest.mark.anyio
async def test_standalone_output_is_checked_before_native_sender():
    from gateway.config import PlatformConfig
    from gateway.conversation import conversation_standalone
    calls = []
    @conversation_standalone
    async def native(platform, pconfig, chat_id, message, thread_id=None, media_files=None, force_document=False, args=None):
        calls.append((message, media_files, args))
        return {"success": True}
    config = PlatformConfig(extra={"groupchat": {"enabled": True, "relevance": {"enabled": False}}})
    result = await native(Platform.MATTERMOST, config, "room", "understood")
    assert result["suppressed"] and calls == []
    original = {"message": "ok", "subject": "Report"}
    await native(Platform.MATTERMOST, config, "room", "ok", media_files=["report.pdf"], args=original)
    assert calls == [("", ["report.pdf"], {"message": "", "subject": "Report"})]
    assert original["message"] == "ok"
