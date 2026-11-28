"""Gateway-rendered catalogs (/help, /commands) embed third-party text — skill and plugin
descriptions. A description naming a file that exists on the gateway host must stay text: it must
not be promoted to a document attachment and uploaded to whoever typed /help."""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, TextOnlyReply
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _event(text: str = "/help") -> MessageEvent:
    return MessageEvent(text=text, source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat-1", user_id="user-1", user_name="tester",
        chat_type="dm"))


def _runner():
    from gateway.run import GatewayRunner
    return object.__new__(GatewayRunner)


@pytest.fixture
def private_doc(tmp_path):
    doc = tmp_path / "PRIVATE_NOTES.md"
    doc.write_text("internal operator notes")
    return doc


@pytest.mark.asyncio
async def test_text_only_reply_keeps_existing_path_as_text(private_doc):
    adapter = _Adapter()
    reply = f"/ops - Operate the box. Source of truth: {private_doc} — read it first."
    extracted = await adapter._extract_response_content(
        reply, _event(), "s", is_ephemeral_response=False, text_only=True)
    assert extracted.local_files == []
    assert str(private_doc) in extracted.text_content


@pytest.mark.asyncio
async def test_plain_reply_still_attaches_existing_path(private_doc):
    """Control: ordinary agent replies keep bare-path delivery."""
    adapter = _Adapter()
    extracted = await adapter._extract_response_content(
        f"Here is the report: {private_doc}", _event(), "s", is_ephemeral_response=False)
    assert extracted.local_files == [str(private_doc)]


@pytest.mark.asyncio
@pytest.mark.parametrize("command, handler", [
    ("/help", "_handle_help_command"),
    ("/commands 999", "_handle_commands_command"),
])
async def test_catalog_commands_return_text_only(monkeypatch, private_doc, command, handler):
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/ops": {"description": f"Operate the box. Source of truth: {private_doc}"}},
    )
    result = await getattr(_runner(), handler)(_event(command))
    assert isinstance(result, TextOnlyReply)
    adapter = _Adapter()
    extracted = await adapter._extract_response_content(
        result, _event(command), "s", is_ephemeral_response=False,
        text_only=isinstance(result, TextOnlyReply))
    assert extracted.local_files == []
