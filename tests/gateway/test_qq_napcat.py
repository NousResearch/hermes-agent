"""Tests for the QQ (NapCat) platform adapter."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType


def _group_payload(*, message_id, user_id, group_id=987654321, text, nickname, card=None, self_id=999001, segments=None):
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": self_id,
        "message_id": message_id,
        "user_id": user_id,
        "group_id": group_id,
        "raw_message": text,
        "message": segments or [{"type": "text", "data": {"text": text}}],
        "sender": {"nickname": nickname, "card": card or nickname},
    }


@pytest.mark.asyncio
async def test_build_message_event_from_private_payload():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    event = adapter._build_message_event(
        {
            "post_type": "message",
            "message_type": "private",
            "message_id": 123,
            "user_id": 456789,
            "raw_message": "hello from qq",
            "sender": {"nickname": "Alice"},
        }
    )

    assert event.text == "hello from qq"
    assert event.message_type == MessageType.TEXT
    assert event.message_id == "123"
    assert event.source.platform == getattr(Platform, "QQ_NAPCAT")
    assert event.source.chat_id == "456789"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "456789"
    assert event.source.user_name == "Alice"


@pytest.mark.asyncio
async def test_build_message_event_from_group_payload():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    event = adapter._build_message_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 124,
            "user_id": 456789,
            "group_id": 987654321,
            "raw_message": "hello group",
            "sender": {"nickname": "Alice", "card": "AliceCard"},
        }
    )

    assert event.text == "hello group"
    assert event.message_id == "124"
    assert event.source.chat_id == "987654321"
    assert event.source.chat_type == "group"
    assert event.source.user_id == "456789"
    assert event.source.user_name == "AliceCard"


@pytest.mark.asyncio
async def test_group_message_requires_mention_when_enabled():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        {
            "post_type": "message",
            "message_type": "group",
            "self_id": 999001,
            "message_id": 125,
            "user_id": 456789,
            "group_id": 987654321,
            "raw_message": "hello group",
            "message": [{"type": "text", "data": {"text": "hello group"}}],
            "sender": {"nickname": "Alice", "card": "AliceCard"},
        }
    )

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_message_with_bot_mention_is_processed_and_cleaned():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        {
            "post_type": "message",
            "message_type": "group",
            "self_id": 999001,
            "message_id": 126,
            "user_id": 456789,
            "group_id": 987654321,
            "raw_message": "@999001 hello group",
            "message": [
                {"type": "at", "data": {"qq": "999001"}},
                {"type": "text", "data": {"text": " hello group"}},
            ],
            "sender": {"nickname": "Alice", "card": "AliceCard"},
        }
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello group"
    assert event.source.chat_id == "987654321"


@pytest.mark.asyncio
async def test_group_message_matching_wake_word_pattern_is_processed():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
                "mention_patterns": [r"^\s*马噶\b"],
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        {
            "post_type": "message",
            "message_type": "group",
            "self_id": 999001,
            "message_id": 127,
            "user_id": 456789,
            "group_id": 987654321,
            "raw_message": "马噶 看看这个",
            "message": [{"type": "text", "data": {"text": "马噶 看看这个"}}],
            "sender": {"nickname": "Alice", "card": "AliceCard"},
        }
    )

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_group_follow_up_window_allows_same_user_without_repeat_mention():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    adapter._record_successful_response_context(
        MessageEvent(
            text="hello group",
            source=SessionSource(
                platform=Platform.QQ_NAPCAT,
                chat_id="987654321",
                chat_type="group",
                user_id="456789",
                user_name="Alice",
            ),
        ),
        ["bot-msg-1"],
    )

    await adapter._handle_payload(
        {
            "post_type": "message",
            "message_type": "group",
            "self_id": 999001,
            "message_id": 128,
            "user_id": 456789,
            "group_id": 987654321,
            "raw_message": "继续说",
            "message": [{"type": "text", "data": {"text": "继续说"}}],
            "sender": {"nickname": "Alice", "card": "AliceCard"},
        }
    )

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_group_follow_up_window_does_not_open_for_other_users():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    adapter._record_successful_response_context(
        MessageEvent(
            text="hello group",
            source=SessionSource(
                platform=Platform.QQ_NAPCAT,
                chat_id="987654321",
                chat_type="group",
                user_id="456789",
                user_name="Alice",
            ),
        ),
        ["bot-msg-2"],
    )

    await adapter._handle_payload(
        {
            "post_type": "message",
            "message_type": "group",
            "self_id": 999001,
            "message_id": 129,
            "user_id": 111222,
            "group_id": 987654321,
            "raw_message": "我也问一句",
            "message": [{"type": "text", "data": {"text": "我也问一句"}}],
            "sender": {"nickname": "Bob", "card": "BobCard"},
        }
    )

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_reply_to_recent_bot_message_is_processed():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    adapter._record_successful_response_context(
        MessageEvent(
            text="hello group",
            source=SessionSource(
                platform=Platform.QQ_NAPCAT,
                chat_id="987654321",
                chat_type="group",
                user_id="456789",
                user_name="Alice",
            ),
        ),
        ["bot-msg-3"],
    )

    await adapter._handle_payload(
        {
            "post_type": "message",
            "message_type": "group",
            "self_id": 999001,
            "message_id": 130,
            "user_id": 111222,
            "group_id": 987654321,
            "raw_message": "接着这条问",
            "message": [
                {"type": "reply", "data": {"id": "bot-msg-3"}},
                {"type": "text", "data": {"text": "接着这条问"}},
            ],
            "sender": {"nickname": "Bob", "card": "BobCard"},
        }
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_message_id == "bot-msg-3"


@pytest.mark.asyncio
async def test_recent_same_user_group_session_allows_follow_up_after_restart():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.session import SessionEntry

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
                "followup_window_seconds": 900,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    payload = _group_payload(
        message_id=131,
        user_id=456789,
        text="继续说下去",
        nickname="Alice",
        card="AliceCard",
    )
    session_key = adapter._session_key_for_source(adapter._build_message_event(payload).source)
    adapter.set_session_store(
        SimpleNamespace(
            config=None,
            list_sessions=MagicMock(
                return_value=[
                    SessionEntry(
                        session_key=session_key,
                        session_id="sess-1",
                        created_at=datetime.now() - timedelta(minutes=5),
                        updated_at=datetime.now() - timedelta(seconds=30),
                        platform=Platform.QQ_NAPCAT,
                        chat_type="group",
                    )
                ]
            ),
        )
    )

    await adapter._handle_payload(payload)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_recent_project_group_session_allows_cross_user_follow_up_after_restart():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.session import SessionEntry

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
                "project_group_mode": True,
                "group_batch_debounce_seconds": 0.01,
                "followup_window_seconds": 900,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    payload = _group_payload(
        message_id=132,
        user_id=111222,
        text="那接下来怎么安排",
        nickname="Bob",
        card="BobCard",
    )
    session_key = adapter._session_key_for_source(adapter._build_message_event(payload).source)
    adapter.set_session_store(
        SimpleNamespace(
            config=None,
            list_sessions=MagicMock(
                return_value=[
                    SessionEntry(
                        session_key=session_key,
                        session_id="sess-2",
                        created_at=datetime.now() - timedelta(minutes=5),
                        updated_at=datetime.now() - timedelta(seconds=20),
                        platform=Platform.QQ_NAPCAT,
                        chat_type="group",
                    )
                ]
            ),
        )
    )

    await adapter._handle_payload(payload)
    await asyncio.sleep(0.03)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "BobCard: 那接下来怎么安排" in event.text


@pytest.mark.asyncio
async def test_project_group_mode_merges_observed_messages_before_trigger_dispatch():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
                "project_group_mode": True,
                "group_batch_debounce_seconds": 0.01,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        _group_payload(
            message_id=201,
            user_id=456789,
            text="今天天气真好",
            nickname="Alice",
            card="AliceCard",
        )
    )
    adapter.handle_message.assert_not_awaited()

    await adapter._handle_payload(
        _group_payload(
            message_id=202,
            user_id=111222,
            text="@999001 不知道马噶那边怎么样",
            nickname="Bob",
            card="BobCard",
            segments=[
                {"type": "at", "data": {"qq": "999001"}},
                {"type": "text", "data": {"text": " 不知道马噶那边怎么样"}},
            ],
        )
    )

    await asyncio.sleep(0.03)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.message_id == "202"
    assert "AliceCard: 今天天气真好" in event.text
    assert "BobCard: 不知道马噶那边怎么样" in event.text


@pytest.mark.asyncio
async def test_project_group_mode_enforces_min_interval_and_merges_cooldown_messages():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
                "project_group_mode": True,
                "group_batch_debounce_seconds": 0.01,
                "group_min_model_interval_seconds": 0.08,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        _group_payload(
            message_id=203,
            user_id=456789,
            text="@999001 第一问",
            nickname="Alice",
            card="AliceCard",
            segments=[
                {"type": "at", "data": {"qq": "999001"}},
                {"type": "text", "data": {"text": " 第一问"}},
            ],
        )
    )
    await asyncio.sleep(0.03)
    assert adapter.handle_message.await_count == 1

    await adapter._handle_payload(
        _group_payload(
            message_id=204,
            user_id=111222,
            text="@999001 第二问",
            nickname="Bob",
            card="BobCard",
            segments=[
                {"type": "at", "data": {"qq": "999001"}},
                {"type": "text", "data": {"text": " 第二问"}},
            ],
        )
    )
    await asyncio.sleep(0.02)
    await adapter._handle_payload(
        _group_payload(
            message_id=205,
            user_id=333444,
            text="补一句上下文",
            nickname="Carol",
            card="CarolCard",
        )
    )
    await asyncio.sleep(0.03)

    assert adapter.handle_message.await_count == 1

    await asyncio.sleep(0.06)

    assert adapter.handle_message.await_count == 2
    event = adapter.handle_message.await_args_list[1].args[0]
    assert "BobCard: 第二问" in event.text
    assert "CarolCard: 补一句上下文" in event.text


@pytest.mark.asyncio
async def test_project_group_mode_waits_for_active_session_before_flushing():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
                "project_group_mode": True,
                "group_sessions_per_user": False,
                "group_batch_debounce_seconds": 0.01,
                "group_min_model_interval_seconds": 0.01,
                "group_batch_retry_seconds": 0.01,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    source = adapter._build_message_event(
        _group_payload(
            message_id=206,
            user_id=456789,
            text="@999001 先别打断",
            nickname="Alice",
            card="AliceCard",
            segments=[
                {"type": "at", "data": {"qq": "999001"}},
                {"type": "text", "data": {"text": " 先别打断"}},
            ],
        )
    ).source
    session_key = adapter._session_key_for_source(source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter._handle_payload(
        _group_payload(
            message_id=206,
            user_id=456789,
            text="@999001 先别打断",
            nickname="Alice",
            card="AliceCard",
            segments=[
                {"type": "at", "data": {"qq": "999001"}},
                {"type": "text", "data": {"text": " 先别打断"}},
            ],
        )
    )

    await asyncio.sleep(0.04)
    adapter.handle_message.assert_not_awaited()

    adapter._active_sessions.pop(session_key, None)
    await asyncio.sleep(0.04)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_project_group_mode_follow_up_window_is_group_shared():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
                "project_group_mode": True,
                "group_batch_debounce_seconds": 0.01,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    adapter._record_successful_response_context(
        MessageEvent(
            text="hello group",
            source=SessionSource(
                platform=Platform.QQ_NAPCAT,
                chat_id="987654321",
                chat_type="group",
                user_id="456789",
                user_name="Alice",
            ),
        ),
        ["bot-msg-9"],
    )

    await adapter._handle_payload(
        _group_payload(
            message_id=207,
            user_id=111222,
            text="那接下来怎么做",
            nickname="Bob",
            card="BobCard",
        )
    )
    await asyncio.sleep(0.03)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "BobCard: 那接下来怎么做" in event.text


@pytest.mark.asyncio
async def test_send_uses_private_and_group_actions():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter._call_api = AsyncMock(return_value={"message_id": 999})

    adapter._chat_types["456789"] = "private"
    result = await adapter.send("456789", "hello dm")

    assert result.success is True
    assert adapter._call_api.await_args.args[0] == "send_private_msg"
    assert adapter._call_api.await_args.args[1]["user_id"] == 456789

    adapter._call_api.reset_mock()

    result = await adapter.send("group:987654321", "hello group")

    assert result.success is True
    assert adapter._call_api.await_args.args[0] == "send_group_msg"
    assert adapter._call_api.await_args.args[1]["group_id"] == 987654321


@pytest.mark.asyncio
async def test_upload_group_file_calls_napcat_upload_api(tmp_path):
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    upload_path = tmp_path / "群文件.txt"
    upload_path.write_text("hello")

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter._call_api = AsyncMock(return_value={"file_id": "fid-1"})

    data = await adapter.upload_group_file(
        "group:987654321",
        str(upload_path),
        folder_id="/docs",
    )

    assert data == {"file_id": "fid-1"}
    assert adapter._call_api.await_args.args == (
        "upload_group_file",
        {
            "group_id": 987654321,
            "file": str(upload_path.resolve()),
            "name": "群文件.txt",
            "folder": "/docs",
        },
    )


@pytest.mark.asyncio
async def test_upload_group_file_omits_folder_for_root(tmp_path):
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    upload_path = tmp_path / "根目录文件.txt"
    upload_path.write_text("hello")

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter._call_api = AsyncMock(return_value={"file_id": "fid-root"})

    data = await adapter.upload_group_file(
        "group:987654321",
        str(upload_path),
        folder_id="/",
    )

    assert data == {"file_id": "fid-root"}
    assert adapter._call_api.await_args.args == (
        "upload_group_file",
        {
            "group_id": 987654321,
            "file": str(upload_path.resolve()),
            "name": "根目录文件.txt",
        },
    )


@pytest.mark.asyncio
async def test_group_file_listing_wrappers_call_expected_actions():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter._call_api = AsyncMock(return_value={"files": [], "folders": []})

    root_listing = await adapter.get_group_root_files("group:987654321")
    folder_listing = await adapter.get_group_files_by_folder("group:987654321", "/docs")

    assert root_listing == {"files": [], "folders": []}
    assert folder_listing == {"files": [], "folders": []}
    assert adapter._call_api.await_args_list[0].args == (
        "get_group_root_files",
        {"group_id": 987654321},
    )
    assert adapter._call_api.await_args_list[1].args == (
        "get_group_files_by_folder",
        {"group_id": 987654321, "folder_id": "/docs"},
    )


@pytest.mark.asyncio
async def test_delete_group_file_calls_napcat_delete_api():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter._call_api = AsyncMock(return_value={})

    data = await adapter.delete_group_file("group:987654321", "fid-1", 102)

    assert data == {}
    assert adapter._call_api.await_args.args == (
        "delete_group_file",
        {"group_id": 987654321, "file_id": "fid-1", "busid": 102},
    )


@pytest.mark.asyncio
async def test_create_group_file_folder_calls_napcat_create_api():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter._call_api = AsyncMock(return_value={"folder_id": "/docs"})

    data = await adapter.create_group_file_folder("group:987654321", "文档")

    assert data == {"folder_id": "/docs"}
    assert adapter._call_api.await_args.args == (
        "create_group_file_folder",
        {"group_id": 987654321, "name": "文档", "parent_id": "/"},
    )


@pytest.mark.asyncio
async def test_delete_group_folder_calls_napcat_delete_api():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter._call_api = AsyncMock(return_value={})

    data = await adapter.delete_group_folder("group:987654321", "/docs")

    assert data == {}
    assert adapter._call_api.await_args.args == (
        "delete_group_folder",
        {"group_id": 987654321, "folder_id": "/docs"},
    )


@pytest.mark.asyncio
async def test_group_file_info_and_url_wrappers_call_expected_actions():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter._call_api = AsyncMock(side_effect=[
        {"total_space": 1},
        {"url": "https://files.example.com/fid-1"},
    ])

    info = await adapter.get_group_file_system_info("group:987654321")
    url = await adapter.get_group_file_url("group:987654321", "fid-1", 102)

    assert info == {"total_space": 1}
    assert url == {"url": "https://files.example.com/fid-1"}
    assert adapter._call_api.await_args_list[0].args == (
        "get_group_file_system_info",
        {"group_id": 987654321},
    )
    assert adapter._call_api.await_args_list[1].args == (
        "get_group_file_url",
        {"group_id": 987654321, "file_id": "fid-1", "busid": 102},
    )


@pytest.mark.asyncio
async def test_move_rename_and_forward_group_file_call_expected_actions():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter._call_api = AsyncMock(side_effect=[{}, {}, {}])

    moved = await adapter.move_group_file("group:987654321", "fid-1", "/docs")
    renamed = await adapter.rename_group_file("group:987654321", "fid-1", "/", "新文件.pdf")
    forwarded = await adapter.trans_group_file("group:987654321", "fid-1", "group:123456")

    assert moved == {}
    assert renamed == {}
    assert forwarded == {}
    assert adapter._call_api.await_args_list[0].args == (
        "move_group_file",
        {"group_id": 987654321, "file_id": "fid-1", "target_dir": "/docs"},
    )
    assert adapter._call_api.await_args_list[1].args == (
        "rename_group_file",
        {
            "group_id": 987654321,
            "file_id": "fid-1",
            "current_parent_directory": "/",
            "new_name": "新文件.pdf",
        },
    )
    assert adapter._call_api.await_args_list[2].args == (
        "trans_group_file",
        {"group_id": 987654321, "file_id": "fid-1", "target_group_id": 123456},
    )


@pytest.mark.asyncio
async def test_connect_preserves_ws_path_and_query_when_access_token_present():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    class FakeSession:
        def __init__(self):
            self.url = None

        async def ws_connect(self, url, heartbeat=None):
            self.url = url
            return object()

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001/ws?client=hermes",
                "access_token": "abc+def&ghi=",
            },
        )
    )
    adapter._session = FakeSession()

    await adapter._connect_websocket()

    assert adapter._session.url == (
        "ws://127.0.0.1:3001/ws?client=hermes&access_token=abc%2Bdef%26ghi%3D"
    )


@pytest.mark.asyncio
async def test_cache_segment_media_decodes_file_uri_with_spaces_and_unicode(tmp_path):
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    media_path = tmp_path / "截图 1.png"
    media_path.write_bytes(b"png")

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    cached_path, mime_type = await adapter._cache_segment_media(
        "image",
        {"file": media_path.as_uri()},
    )

    assert cached_path == str(media_path)
    assert mime_type == "image/png"


@pytest.mark.asyncio
async def test_cache_segment_media_ignores_query_when_guessing_image_extension(tmp_path, monkeypatch):
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", cache_dir)

    fake_response = MagicMock()
    fake_response.content = b"\x89PNG\r\n\x1a\nfake"
    fake_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("tools.url_safety.is_safe_url", return_value=True):
        cached_path, mime_type = await adapter._cache_segment_media(
            "image",
            {"url": "https://cdn.example.com/a.png?token=abc"},
        )

    assert Path(cached_path).suffix == ".png"
    assert mime_type == "image/png"


@pytest.mark.asyncio
async def test_cache_segment_media_ignores_query_when_guessing_audio_extension(tmp_path, monkeypatch):
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    cache_dir = tmp_path / "audio"
    monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", cache_dir)

    fake_response = MagicMock()
    fake_response.content = b"OggSfake"
    fake_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=fake_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("tools.url_safety.is_safe_url", return_value=True):
        cached_path, mime_type = await adapter._cache_segment_media(
            "record",
            {"url": "https://cdn.example.com/v.ogg?sig=xyz"},
        )

    assert Path(cached_path).suffix == ".ogg"
    assert mime_type == "audio/ogg"
