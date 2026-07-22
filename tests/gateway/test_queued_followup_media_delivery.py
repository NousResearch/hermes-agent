import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import quote

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _runner() -> GatewayRunner:
    return object.__new__(GatewayRunner)


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        thread_id="topic-1",
    )


def _adapter():
    return SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=True, message_id="text")),
        send_multiple_images=AsyncMock(),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


class _QueuedMediaAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent = []
        self.documents = []
        self.images = []
        self.typing = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"text-{len(self.sent)}")

    async def send_document(self, chat_id, file_path, metadata=None, **kwargs):
        self.documents.append(
            {
                "chat_id": chat_id,
                "file_path": file_path,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"doc-{len(self.documents)}")

    async def send_multiple_images(self, chat_id, images, metadata=None):
        self.images.append(
            {
                "chat_id": chat_id,
                "images": images,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"images-{len(self.images)}")

    async def send_typing(self, chat_id, metadata=None):
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


class _QueuedMediaAgent:
    calls = 0
    media_path = ""

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        if type(self).calls == 1:
            return {
                "final_response": f"Done.\nMEDIA:{type(self).media_path}",
                "messages": [],
                "api_calls": 1,
            }

        return {
            "final_response": "follow-up answer",
            "messages": [],
            "api_calls": 1,
        }


def _runner_for_run_agent(adapter):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


@pytest.mark.asyncio
async def test_queued_first_response_strips_media_tag_from_text_and_sends_document(
    tmp_path, monkeypatch,
):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "quote.pdf")
    adapter = _adapter()

    await GatewayRunner._send_queued_first_response(
        _runner(),
        f"Done.\nMEDIA:{media_file}",
        _source(),
        adapter,
        {"thread_id": "topic-1"},
    )

    adapter.send.assert_awaited_once_with(
        "chat-1",
        "Done.",
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )


@pytest.mark.asyncio
async def test_queued_first_response_strips_bare_image_path_from_text_and_sends_image(
    tmp_path, monkeypatch,
):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "avatar.png")
    adapter = _adapter()

    await GatewayRunner._send_queued_first_response(
        _runner(),
        f"Here is the image:\n{media_file}",
        _source(),
        adapter,
        {"thread_id": "topic-1"},
    )

    adapter.send.assert_awaited_once_with(
        "chat-1",
        "Here is the image:",
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_multiple_images.assert_awaited_once_with(
        chat_id="chat-1",
        images=[(f"file://{quote(str(media_file))}", "")],
        metadata={"thread_id": "topic-1"},
    )


@pytest.mark.asyncio
async def test_queued_first_response_with_only_media_sends_no_empty_text(
    tmp_path, monkeypatch,
):
    media_file = _allowed_media_path(tmp_path, monkeypatch, "quote.pdf")
    adapter = _adapter()

    await GatewayRunner._send_queued_first_response(
        _runner(),
        f"MEDIA:{media_file}",
        _source(),
        adapter,
        {"thread_id": "topic-1"},
    )

    adapter.send.assert_not_awaited()
    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )


@pytest.mark.asyncio
async def test_queued_first_response_keeps_remote_image_url_visible():
    adapter = _adapter()
    response = "Mockup: https://example.com/mockup.png"

    await GatewayRunner._send_queued_first_response(
        _runner(),
        response,
        _source(),
        adapter,
        {"thread_id": "topic-1"},
    )

    adapter.send.assert_awaited_once_with(
        "chat-1",
        response,
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_multiple_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_agent_queued_followup_flushes_clean_text_and_native_media(
    tmp_path, monkeypatch,
):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    media_file = _allowed_media_path(tmp_path, monkeypatch, "handoff.pdf")
    _QueuedMediaAgent.calls = 0
    _QueuedMediaAgent.media_path = str(media_file)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _QueuedMediaAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "test"})

    adapter = _QueuedMediaAdapter()
    runner = _runner_for_run_agent(adapter)
    source = _source()
    source.chat_type = "group"
    session_key = build_session_key(source)
    adapter._pending_messages[session_key] = MessageEvent(
        text="queued follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="first prompt",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-queued-media",
        session_key=session_key,
    )

    assert result["final_response"] == "follow-up answer"
    assert adapter.sent == [
        {
            "chat_id": "chat-1",
            "content": "Done.",
            "reply_to": None,
            "metadata": {"thread_id": "topic-1"},
        }
    ]
    assert adapter.documents == [
        {
            "chat_id": "chat-1",
            "file_path": str(media_file),
            "metadata": {"thread_id": "topic-1"},
        }
    ]
