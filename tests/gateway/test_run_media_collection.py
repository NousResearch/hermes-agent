import json

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from gateway.run import _collect_new_media_tags_from_tool_messages


def test_tool_media_collection_ignores_media_examples_and_regexes():
    """Tool-output source code examples must not become outbound attachments."""
    tool_messages = [
        {
            "role": "tool",
            "content": """
Some inspected source text:
    MEDIA:/path/to/audio.ogg
    MEDIA:/tmp/hermes/image.png
    media_pattern = re.compile(r'''[`\"']?MEDIA:\\s*(?P<path>`[^`\\n]+`|\"[^\"\\n]+\")''')
Tool docs: include MEDIA:<path> in your response.
""",
        }
    ]

    media_tags, has_voice_directive = _collect_new_media_tags_from_tool_messages(
        tool_messages,
        history_media_paths=set(),
    )

    assert media_tags == []
    assert has_voice_directive is False


def test_tool_media_collection_keeps_existing_generated_media(tmp_path):
    """Real generated files from tool JSON remain deliverable."""
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"ogg")
    content = json.dumps(
        {
            "success": True,
            "file_path": str(audio),
            "media_tag": f"[[audio_as_voice]]\nMEDIA:{audio}",
        }
    )

    media_tags, has_voice_directive = _collect_new_media_tags_from_tool_messages(
        [{"role": "tool", "content": content}],
        history_media_paths=set(),
    )

    assert media_tags == [f"MEDIA:{audio}"]
    assert has_voice_directive is True


def test_tool_media_collection_skips_history_media(tmp_path):
    audio = tmp_path / "already-sent.ogg"
    audio.write_bytes(b"ogg")

    media_tags, _ = _collect_new_media_tags_from_tool_messages(
        [{"role": "tool", "content": f"MEDIA:{audio}"}],
        history_media_paths={str(audio)},
    )

    assert media_tags == []


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(extra={"group_sessions_per_user": False}), platform)
        self.sent: list[str] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id: str, content: str, **kwargs) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def get_chat_info(self, chat_id: str) -> dict:
        return {}

    async def _keep_typing(self, *args, **kwargs) -> None:
        return None


@pytest.mark.asyncio
async def test_linear_response_keeps_media_examples_as_text_not_attachment_comments():
    adapter = _RecordingAdapter(Platform.LINEAR)

    async def handler(_event):
        return (
            "For documentation, show these examples literally:\n"
            "- `MEDIA:/path/file.png`\n"
            "- ![sample](https://example.com/sample.png)\n"
            "- MEDIA:/tmp/voice.ogg\n"
            "Do not upload anything."
        )

    adapter.set_message_handler(handler)
    source = SessionSource(
        platform=Platform.LINEAR,
        chat_id="issue-id",
        chat_type="thread",
        user_id="anton",
        user_name="Anton",
        thread_id="DEC-18",
    )
    event = MessageEvent(
        text="what is going on here?",
        message_type=MessageType.TEXT,
        source=source,
        message_id="comment-id",
    )

    await adapter._process_message_background(event, "linear:DEC-18")

    assert len(adapter.sent) == 1
    assert "MEDIA:/path/file.png" in adapter.sent[0]
    assert "![sample](https://example.com/sample.png)" in adapter.sent[0]
    assert "MEDIA:/tmp/voice.ogg" in adapter.sent[0]
    assert not any(message.startswith(("📎 File:", "🖼️ Image:", "🔊 Audio:")) for message in adapter.sent)
