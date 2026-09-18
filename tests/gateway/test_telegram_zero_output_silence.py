import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource
from plugins.telegram_no_action_suppression import _on_pre_gateway_dispatch


class DummyTelegramAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []
        self.processing_events = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
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
        return SendResult(success=True, message_id=f"msg-{len(self.sent)}")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def _run_processing_hook(self, hook_name, event, outcome=None):
        self.processing_events.append((hook_name, outcome))

    async def _keep_typing(self, chat_id, interval=2.0, metadata=None, stop_event=None):
        await asyncio.Event().wait()


def _source(*, chat_type="group", platform=Platform.TELEGRAM):
    return SessionSource(
        platform=platform,
        user_id="user-1",
        chat_id="chat-1",
        user_name="Christine",
        chat_type=chat_type,
    )


def _event(text: str, *, chat_type="group", platform=Platform.TELEGRAM):
    return MessageEvent(
        text=text,
        source=_source(chat_type=chat_type, platform=platform),
        message_id="m1",
    )


async def _drain(adapter: DummyTelegramAdapter):
    for _ in range(50):
        if not adapter._session_tasks:
            return
        if all(task.done() for task in adapter._session_tasks.values()):
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_casual_telegram_group_message_is_hard_silent_via_pre_dispatch_plugin(monkeypatch):
    from gateway.run import GatewayRunner

    adapter = AsyncMock()
    runner = object.__new__(GatewayRunner)
    runner.session_store = MagicMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("agent should not run")
    )

    def _invoke_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            result = _on_pre_gateway_dispatch(kwargs["event"])
            return [result] if result else []
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke_hook)

    result = await runner._handle_message(
        _event("Just testing whether this casual message stays quiet.")
    )

    assert result is None
    runner._handle_message_with_agent.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_actionable_telegram_group_message_still_responds_normally():
    adapter = DummyTelegramAdapter()
    handler = AsyncMock(return_value="Ready for review.")
    adapter.set_message_handler(handler)

    await adapter.handle_message(_event("JIMMY: please review this"))
    await _drain(adapter)
    await adapter.cancel_background_tasks()

    handler.assert_called_once()
    assert [item["content"] for item in adapter.sent] == ["Ready for review."]


def test_plugin_classifier_preserves_dm_and_actionable_messages():
    assert _on_pre_gateway_dispatch(_event("hello", chat_type="dm")) is None
    assert _on_pre_gateway_dispatch(_event("hello", chat_type="private")) is None
    assert _on_pre_gateway_dispatch(_event("JIMMY: please review this")) is None
    assert _on_pre_gateway_dispatch(_event("BLOCKER: branch unclear")) is None
    assert _on_pre_gateway_dispatch(_event("hello", platform=Platform.DISCORD)) is None
    assert _on_pre_gateway_dispatch(
        _event("Just testing whether this casual message stays quiet.")
    ) == {
        "action": "skip",
        "reason": "casual/no-action Telegram group message",
    }
