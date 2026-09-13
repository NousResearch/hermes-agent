"""Tests for gateway /yolo session scoping."""

import os
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tools.approval import check_dangerous_command, disable_session_yolo, is_session_yolo_enabled
from tools.approval_context import reset_current_session_key, set_current_session_key


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
    user_id: str | None = None,
    chat_type: str = "dm",
    thread_id: str | None = None,
) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
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
async def test_yolo_command_rechecks_admin_at_side_effect_boundary():
    runner = _make_runner()
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(
                extra={
                    "allow_admin_from": ["admin-1"],
                    "user_allowed_commands": ["yolo"],
                }
            )
        }
    )

    user_event = _make_event("chat-a", user_id="user-1")
    user_session = runner._session_key_for_source(user_event.source)
    assert runner._check_slash_access(user_event.source, "yolo") is None

    denied = await runner._handle_yolo_command(user_event)

    assert "admin" in denied.lower()
    assert is_session_yolo_enabled(user_session) is False

    admin_event = _make_event("chat-b", user_id="admin-1")
    admin_session = runner._session_key_for_source(admin_event.source)
    enabled = await runner._handle_yolo_command(admin_event)

    assert "ON" in enabled
    assert is_session_yolo_enabled(admin_session) is True


@pytest.mark.parametrize(
    ("chat_type", "thread_id", "group_per_user", "thread_per_user"),
    [
        ("group", None, False, False),
        ("thread", "topic-1", True, False),
        ("thread", "topic-1", False, True),
    ],
)
@pytest.mark.asyncio
async def test_yolo_command_cannot_enable_bypass_for_shared_sessions(
    monkeypatch, chat_type, thread_id, group_per_user, thread_per_user
):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    runner = _make_runner()
    runner.config = SimpleNamespace(
        group_sessions_per_user=group_per_user,
        thread_sessions_per_user=thread_per_user,
        platforms={
            Platform.TELEGRAM: SimpleNamespace(
                extra={
                    "group_allow_admin_from": ["admin-1"],
                    "group_user_allowed_commands": ["yolo"],
                }
            )
        },
    )
    admin_event = _make_event(
        "shared-chat", user_id="admin-1", chat_type=chat_type, thread_id=thread_id
    )
    member_event = _make_event(
        "shared-chat", user_id="member-1", chat_type=chat_type, thread_id=thread_id
    )
    shared_session = runner._session_key_for_source(admin_event.source)
    assert runner._session_key_for_source(member_event.source) == shared_session
    assert runner._check_slash_access(member_event.source, "yolo") is None

    try:
        response = await runner._handle_yolo_command(admin_event)

        assert "shared" in response.lower()
        assert is_session_yolo_enabled(shared_session) is False

        token = set_current_session_key(shared_session)
        try:
            denied = check_dangerous_command(
                "rm -rf /tmp/hermes-yolo-shared-session",
                "local",
                approval_callback=lambda *args: "deny",
            )
            assert denied["approved"] is False
        finally:
            reset_current_session_key(token)
    finally:
        disable_session_yolo(shared_session)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["ntfy", "raft", "a2a"])
async def test_yolo_command_rejects_broadcast_sources_that_report_dm(platform):
    dynamic_platform = Platform(platform)
    runner = _make_runner()
    runner.config = SimpleNamespace(
        platforms={
            dynamic_platform: SimpleNamespace(extra={"allow_admin_from": ["admin-1"]})
        }
    )
    event = MessageEvent(
        text="/yolo",
        source=SessionSource(
            platform=dynamic_platform,
            user_id="admin-1",
            chat_id="shared-topic",
            chat_type="dm",
        ),
    )
    session_key = runner._session_key_for_source(event.source)

    try:
        response = await runner._handle_yolo_command(event)

        assert "shared" in response.lower()
        assert is_session_yolo_enabled(session_key) is False
    finally:
        disable_session_yolo(session_key)
