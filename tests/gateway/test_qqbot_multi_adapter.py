"""Tests for multi-account QQBot adapter support."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from gateway.platforms.qqbot import QQMultiAdapter, collect_qq_credentials


_QQ_ENV_PREFIXES = ("QQ_APP_ID", "QQ_CLIENT_SECRET")


def _clear_qq_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key == "QQ_APP_ID" or key == "QQ_CLIENT_SECRET":
            monkeypatch.delenv(key, raising=False)
        elif key.startswith("QQ_APP_ID_") or key.startswith("QQ_CLIENT_SECRET_"):
            monkeypatch.delenv(key, raising=False)


def test_collect_qq_credentials_reads_indexed_env_pairs(monkeypatch):
    _clear_qq_env(monkeypatch)
    monkeypatch.setenv("QQ_APP_ID_1", "app-one")
    monkeypatch.setenv("QQ_CLIENT_SECRET_1", "secret-one")
    monkeypatch.setenv("QQ_APP_ID_0", "app-zero")
    monkeypatch.setenv("QQ_CLIENT_SECRET_0", "secret-zero")
    monkeypatch.setenv("QQ_APP_ID_2", "incomplete")

    creds = collect_qq_credentials(PlatformConfig(enabled=True))

    assert [(label, app_id) for label, app_id, _secret in creds] == [
        ("env:0", "app-zero"),
        ("env:1", "app-one"),
    ]


def test_gateway_connected_platforms_accepts_indexed_qq_env(monkeypatch):
    _clear_qq_env(monkeypatch)
    monkeypatch.setenv("QQ_APP_ID_0", "app-zero")
    monkeypatch.setenv("QQ_CLIENT_SECRET_0", "secret-zero")

    cfg = GatewayConfig(platforms={Platform.QQBOT: PlatformConfig(enabled=True)})

    assert Platform.QQBOT in cfg.get_connected_platforms()


def test_multi_adapter_routes_reply_through_child_that_received_chat(monkeypatch):
    _clear_qq_env(monkeypatch)
    monkeypatch.setenv("QQ_APP_ID_0", "app-zero")
    monkeypatch.setenv("QQ_CLIENT_SECRET_0", "secret-zero")
    monkeypatch.setenv("QQ_APP_ID_1", "app-one")
    monkeypatch.setenv("QQ_CLIENT_SECRET_1", "secret-one")

    adapter = QQMultiAdapter(PlatformConfig(enabled=True))
    assert len(adapter._children) == 2
    child0, child1 = adapter._children
    child0.send = AsyncMock(return_value=SendResult(success=True, message_id="m0"))
    child1.send = AsyncMock(return_value=SendResult(success=True, message_id="m1"))

    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.QQBOT,
            chat_id="chat-a",
            chat_type="dm",
            user_id="user-a",
        ),
        raw_message={"id": "raw"},
        message_id="msg-a",
    )

    awaitable = adapter.handle_child_message(child1, event)
    import asyncio

    asyncio.run(awaitable)
    result = asyncio.run(adapter.send("chat-a", "reply"))

    assert result.message_id == "m1"
    child1.send.assert_awaited_once()
    child0.send.assert_not_awaited()


def test_gateway_runner_uses_multi_adapter_for_qqbot(monkeypatch):
    _clear_qq_env(monkeypatch)
    monkeypatch.setenv("QQ_APP_ID_0", "app-zero")
    monkeypatch.setenv("QQ_CLIENT_SECRET_0", "secret-zero")

    from gateway import platforms as _unused  # noqa: F401 - ensure package import path exists
    import gateway.platforms.qqbot as qqbot_pkg
    from gateway.run import GatewayRunner

    monkeypatch.setattr(qqbot_pkg, "check_qq_requirements", lambda: True)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()

    adapter = GatewayRunner._create_adapter(
        runner,
        Platform.QQBOT,
        PlatformConfig(enabled=True),
    )

    assert isinstance(adapter, QQMultiAdapter)
    assert len(adapter._children) == 1
