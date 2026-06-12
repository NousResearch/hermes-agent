from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource


class _DirectSendAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
        self.calls = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "reply_to": reply_to, "metadata": metadata})
        return SendResult(success=True, message_id=f"m-{len(self.calls)}")

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.asyncio
async def test_discord_operational_final_response_sends_unchanged_and_writes_no_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777,888")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")

    adapter = _DirectSendAdapter()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="555",
        chat_name="provide me with the status of the statute PM worker",
        chat_type="thread",
        thread_id="999",
        user_id="888",
        user_name="Statute PM",
        is_bot=True,
    )
    event = MessageEvent(text="trigger", source=source, message_id="111")
    original = "ACTION_REQUIRED for Galt/default: approval_id: abc123; statute PM must use local DB now"

    assert not hasattr(adapter, "_send_text_response_with_routing_guard")
    result = await adapter._send_with_retry(
        chat_id=event.source.chat_id,
        content=original,
        reply_to="111",
        metadata={"thread_id": "999", "notify": True},
    )

    assert result.success is True
    assert adapter.calls == [{
        "chat_id": "555",
        "content": original,
        "reply_to": "111",
        "metadata": {"thread_id": "999", "notify": True},
    }]
    assert not (tmp_path / "logs" / "routing_guard").exists()


def test_discord_runtime_does_not_import_bot_msg_protocol():
    runtime_files = [
        Path("gateway/platforms/discord.py"),
        Path("plugins/platforms/discord/adapter.py"),
        Path("tools/send_message_tool.py"),
    ]
    offenders = [
        str(path)
        for path in runtime_files
        if "plugins.platforms.discord.bot_msg_protocol" in path.read_text(encoding="utf-8")
        or "from .bot_msg_protocol" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
