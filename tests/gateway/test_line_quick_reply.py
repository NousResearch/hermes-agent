"""Tests for the LINE platform adapter."""

import sys
import time
from pathlib import Path

import pytest


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


from plugins.platforms.line.adapter import LineAdapter


class FakeLineClient:
    def __init__(self, *, fail_reply: bool = False):
        self.fail_reply = fail_reply
        self.replies = []
        self.pushes = []

    async def reply(self, reply_token, messages):
        if self.fail_reply:
            raise RuntimeError("expired reply token")
        self.replies.append((reply_token, messages))

    async def push(self, chat_id, messages):
        self.pushes.append((chat_id, messages))


def _make_adapter(client=None, *, language="en"):
    adapter = object.__new__(LineAdapter)
    adapter._client = client or FakeLineClient()
    adapter._reply_tokens = {}
    adapter.language = language
    return adapter


def _quick_reply_texts(message):
    return [
        item["action"]["text"]
        for item in message["quickReply"]["items"]
    ]


@pytest.mark.asyncio
async def test_send_exec_approval_uses_reply_token_quick_replies():
    client = FakeLineClient()
    adapter = _make_adapter(client)
    adapter._reply_tokens["U123"] = ("reply-token", time.time() + 60)

    result = await adapter.send_exec_approval(
        chat_id="U123",
        command="rm -rf /important",
        session_key="agent:main:line:dm:U123",
        description="dangerous deletion",
    )

    assert result.success is True
    assert result.message_id == "reply-token"
    assert client.pushes == []
    assert len(client.replies) == 1

    token, messages = client.replies[0]
    assert token == "reply-token"
    message = messages[0]
    assert "rm -rf /important" in message["text"]
    assert "dangerous deletion" in message["text"]
    assert _quick_reply_texts(message) == [
        "/approve",
        "/approve session",
        "/approve always",
        "/deny",
    ]


@pytest.mark.asyncio
async def test_send_slash_confirm_pushes_quick_replies_without_reply_token():
    client = FakeLineClient()
    adapter = _make_adapter(client)

    result = await adapter.send_slash_confirm(
        chat_id="U123",
        title="Confirm",
        message="Reset this session?",
        session_key="agent:main:line:dm:U123",
        confirm_id="1",
    )

    assert result.success is True
    assert client.replies == []
    assert len(client.pushes) == 1

    chat_id, messages = client.pushes[0]
    assert chat_id == "U123"
    assert messages[0]["text"] == "Reset this session?"
    assert _quick_reply_texts(messages[0]) == ["/approve", "/always", "/cancel"]


@pytest.mark.asyncio
async def test_quick_reply_send_falls_back_to_push_when_reply_token_fails():
    client = FakeLineClient(fail_reply=True)
    adapter = _make_adapter(client)
    adapter._reply_tokens["U123"] = ("expired-token", time.time() + 60)

    result = await adapter.send_slash_confirm(
        chat_id="U123",
        title="Confirm",
        message="Reload MCP servers?",
        session_key="agent:main:line:dm:U123",
        confirm_id="2",
    )

    assert result.success is True
    assert client.replies == []
    assert len(client.pushes) == 1
    assert client.pushes[0][0] == "U123"


@pytest.mark.asyncio
async def test_quick_reply_labels_use_adapter_language():
    client = FakeLineClient()
    adapter = _make_adapter(client, language="zh-hant")

    result = await adapter.send_exec_approval(
        chat_id="U123",
        command="ls",
        session_key="agent:main:line:dm:U123",
    )

    assert result.success is True
    message = client.pushes[0][1][0]
    assert [
        item["action"]["label"]
        for item in message["quickReply"]["items"]
    ] == ["批准一次", "本次工作階段", "永遠批准", "拒絕"]
    assert _quick_reply_texts(message) == [
        "/approve",
        "/approve session",
        "/approve always",
        "/deny",
    ]
