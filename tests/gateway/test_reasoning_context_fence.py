"""Regression coverage for recalled context in displayed gateway reasoning."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource


PRIVATE_CONTEXT = "PRIVATE_SENTINEL_81312_REASONING_STYLE"


class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.EMAIL)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="reasoning-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


async def _render_reasoning(monkeypatch, *, style: str, reasoning: str) -> str:
    adapter = _CaptureAdapter()
    source = SessionSource(
        platform=Platform.EMAIL,
        user_id="user@test.com",
        chat_id="user@test.com",
        user_name="testuser",
        chat_type="dm",
    )
    event = MessageEvent(text="hello", source=source, message_id="reasoning-81312")
    session_key = "agent:main:email:dm:user@test.com"

    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {Platform.EMAIL: adapter}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._set_session_env = lambda _context: None
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _generation: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=session_key,
        session_id=f"sess-81312-reasoning-{style}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.EMAIL,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.update_session = MagicMock()
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Visible reply",
            "last_reasoning": reasoning,
            "messages": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"show_reasoning": True, "reasoning_style": style}},
    )

    response = await runner._handle_message_with_agent(event, source, session_key, 1)
    assert isinstance(response, str)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("style", "prefix"),
    [("subtext", "-# Visible reasoning"), ("blockquote", "> Visible reasoning")],
)
async def test_reasoning_style_fences_raw_text_before_prefixing(
    monkeypatch, style, prefix
):
    response = await _render_reasoning(
        monkeypatch,
        style=style,
        reasoning=f"Visible reasoning\n<memory-context>\n{PRIVATE_CONTEXT}",
    )

    assert prefix in response
    assert "Visible reply" in response
    assert PRIVATE_CONTEXT not in response
    assert "memory-context" not in response


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["subtext", "blockquote"])
async def test_empty_reasoning_after_fence_does_not_add_markup(monkeypatch, style):
    response = await _render_reasoning(
        monkeypatch,
        style=style,
        reasoning=f"<memory-context>\n{PRIVATE_CONTEXT}",
    )

    assert response == "Visible reply"


@pytest.mark.asyncio
async def test_empty_composed_reasoning_keeps_safe_reply(monkeypatch):
    original_sanitizer = gateway_run._sanitize_gateway_final_response

    def empty_composed_reasoning(platform, text):
        if "💭 **Reasoning:**" in str(text):
            return ""
        return original_sanitizer(platform, text)

    monkeypatch.setattr(
        gateway_run,
        "_sanitize_gateway_final_response",
        empty_composed_reasoning,
    )

    response = await _render_reasoning(
        monkeypatch,
        style="code",
        reasoning="Visible reasoning",
    )

    assert response == "Visible reply"
