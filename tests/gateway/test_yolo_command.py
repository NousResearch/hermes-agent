"""Tests for gateway /yolo session scoping."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, is_session_principal_isolated
from tools import approval as approval_module
from tools import approval_context
from tools.approval import disable_session_yolo, is_session_yolo_enabled


@pytest.fixture(autouse=True)
def _clean_yolo_state(monkeypatch):
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")
    yield
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _make_event(
    chat_id: str,
    *,
    platform: Platform = Platform.TELEGRAM,
    user_id: str | None = None,
    chat_type: str = "dm",
    thread_id: str | None = None,
) -> MessageEvent:
    source = SessionSource(
        platform=platform,
        user_id=user_id or f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type=chat_type,
        thread_id=thread_id,
    )
    return MessageEvent(text="/yolo", source=source)


@pytest.mark.asyncio
async def test_yolo_command_toggles_only_current_session(monkeypatch):
    runner = _make_runner()

    event_a = _make_event("chat-a")
    session_a = runner._session_key_for_source(event_a.source)
    session_b = runner._session_key_for_source(_make_event("chat-b").source)

    result_on = await runner._handle_yolo_command(event_a)

    assert "ON" in result_on
    assert is_session_yolo_enabled(session_a) is True
    assert is_session_yolo_enabled(session_b) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None

    result_off = await runner._handle_yolo_command(event_a)

    assert "OFF" in result_off
    assert is_session_yolo_enabled(session_a) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_type", "thread_id", "group_sessions_per_user", "thread_sessions_per_user"),
    [
        ("group", None, False, False),
        ("thread", "topic-1", True, False),
    ],
)
async def test_shared_session_cannot_enable_yolo_for_another_sender(
    monkeypatch, chat_type, thread_id, group_sessions_per_user, thread_sessions_per_user,
):
    runner = _make_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                extra={"allow_admin_from": ["admin"], "user_allowed_commands": ["yolo"]},
            )
        },
        group_sessions_per_user=group_sessions_per_user,
        thread_sessions_per_user=thread_sessions_per_user,
    )
    admin = _make_event(
        "shared-chat", platform=Platform.DISCORD, user_id="admin",
        chat_type=chat_type, thread_id=thread_id,
    )
    non_admin = _make_event(
        "shared-chat", platform=Platform.DISCORD, user_id="member",
        chat_type=chat_type, thread_id=thread_id,
    )
    session_key = runner._session_key_for_source(admin.source)
    assert runner._session_key_for_source(non_admin.source) == session_key
    disable_session_yolo(session_key)

    result = await runner._handle_yolo_command(admin)

    assert "shared" in result.lower()
    assert is_session_yolo_enabled(session_key) is False

    denied = MagicMock(return_value="deny")
    token = approval_context.set_current_session_key(session_key)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    try:
        decision = approval_module.check_all_command_guards(
            "rm -rf /tmp/member-scratch", "local", approval_callback=denied,
        )
    finally:
        approval_context.reset_current_session_key(token)
        disable_session_yolo(session_key)
    assert decision["approved"] is False
    denied.assert_called_once()


def test_session_yolo_scope_distinguishes_paired_and_broadcast_dms():
    paired = _make_event("private-chat", user_id="admin").source
    broadcast = SimpleNamespace(
        platform=SimpleNamespace(value="ntfy"),
        chat_id="shared-topic",
        chat_type="dm",
    )

    assert is_session_principal_isolated(paired) is True
    assert is_session_principal_isolated(broadcast) is False
