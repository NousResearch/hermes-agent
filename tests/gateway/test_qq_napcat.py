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
        PlatformConfig(
            enabled=True,
            extra={"ws_url": "ws://127.0.0.1:3001", "allow_all_groups": True},
        )
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
async def test_connect_records_runtime_missing_status_after_local_connect_failure():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    class _FailingSession:
        async def ws_connect(self, _url, heartbeat=None):
            raise ConnectionRefusedError("boom")

        async def close(self):
            return None

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={"ws_url": "ws://127.0.0.1:3001", "allow_all_groups": True},
        )
    )

    with patch("gateway.platforms.qq_napcat.aiohttp.ClientSession", return_value=_FailingSession()), \
         patch(
             "gateway.platforms.qq_napcat.diagnose_local_qq_napcat_endpoint",
             return_value={
                 "code": "qq_napcat_runtime_missing",
                 "message": "QQ NapCat local runtime is missing",
             },
         ), \
         patch("gateway.status.write_runtime_status") as status_mock:
        connected = await adapter.connect()

    assert connected is False
    assert status_mock.call_args_list[-1].kwargs["platform_state"] == "unavailable"
    assert status_mock.call_args_list[-1].kwargs["error_code"] == "qq_napcat_runtime_missing"
    assert status_mock.call_args_list[-1].kwargs["error_message"] == "QQ NapCat local runtime is missing"


@pytest.mark.asyncio
async def test_build_message_event_from_group_payload():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={"ws_url": "ws://127.0.0.1:3001", "allow_all_groups": True},
        )
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
async def test_build_message_event_from_group_payload_includes_trigger_reason_metadata():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": False,
            },
        )
    )

    event = adapter._build_message_event(
        _group_payload(
            message_id=1242,
            user_id=456789,
            text="@马嘎 ok",
            nickname="Alice",
            card="AliceCard",
            segments=[
                {"type": "at", "data": {"qq": "999001", "name": "马嘎"}},
                {"type": "text", "data": {"text": " ok"}},
            ],
        )
    )

    assert event.metadata["group_trigger_reason"] == "require_mention_disabled"
    assert event.metadata["explicit_group_trigger"] is True
    assert event.metadata["explicit_group_trigger_reason"] == "bot_mention"
    assert event.metadata["explicit_addressed"] is True
    assert event.metadata["address_reason"] == "bot_mention"
    assert event.metadata["requires_reply"] is True


@pytest.mark.asyncio
async def test_build_message_event_strips_cq_image_markup_from_text():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    event = adapter._build_message_event(
        {
            "post_type": "message",
            "message_type": "private",
            "message_id": 1241,
            "user_id": 456789,
            "raw_message": (
                "[CQ:image,file=7CDEAA1F045EF8013AC8631FF3708901.png,sub_type=0,"
                "url=https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=abc]"
                "看看这个"
            ),
            "message": [
                {
                    "type": "image",
                    "data": {
                        "file": "7CDEAA1F045EF8013AC8631FF3708901.png",
                        "url": "https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=abc",
                    },
                },
                {"type": "text", "data": {"text": "看看这个"}},
            ],
            "sender": {"nickname": "Alice"},
        }
    )

    assert event.text == "看看这个"
    assert "CQ:image" not in event.text
    assert "7CDEAA1F045EF8013AC8631FF3708901" not in event.text
    assert event.message_type == MessageType.PHOTO


@pytest.mark.asyncio
async def test_request_payload_is_persisted_without_dispatching_message():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_social_requests import QqSocialRequestStore

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        {
            "post_type": "request",
            "request_type": "group",
            "sub_type": "invite",
            "group_id": 987654321,
            "user_id": 179033731,
            "comment": "来群里聊项目",
            "flag": "group-flag-qqnapcat",
            "time": 1713012345,
        }
    )

    adapter.handle_message.assert_not_awaited()
    stored = QqSocialRequestStore().get_request("group:group-flag-qqnapcat")
    assert stored is not None
    assert stored["status"] == "pending"
    assert stored["sub_type"] == "invite"


@pytest.mark.asyncio
async def test_friend_request_is_auto_approved_when_policy_enabled():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_social_policy import set_social_policy
    from gateway.qq_social_requests import QqSocialRequestStore

    set_social_policy(
        auto_approve_friend_requests=True,
        notify_target="qq_napcat:dm:179033731",
        updated_by="test",
    )
    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter.handle_message = AsyncMock()

    with patch.object(adapter, "_call_api", new=AsyncMock(return_value={"ok": True})) as call_mock, \
         patch.object(adapter, "send", new=AsyncMock(return_value=SimpleNamespace(success=True, error=None))) as send_mock:
        await adapter._handle_payload(
            {
                "post_type": "request",
                "request_type": "friend",
                "user_id": 456789,
                "comment": "加一下",
                "flag": "friend-auto-1",
                "time": 1713012345,
            }
        )

    stored = QqSocialRequestStore().get_request("friend:friend-auto-1")
    assert stored is not None
    assert stored["status"] == "approved"
    assert stored["handled_by"] == "qq_napcat:auto_social_policy"
    assert stored["handled_via"] == "auto_social_policy"
    assert stored["decision_note"] == "按社交自动处理策略自动通过好友请求。"
    call_mock.assert_awaited_once_with(
        "set_friend_add_request",
        {"flag": "friend-auto-1", "approve": True},
    )
    send_mock.assert_awaited()
    notice_text = send_mock.await_args.args[1]
    assert "当前状态：approved" in notice_text
    assert "处理来源：auto_social_policy" in notice_text


@pytest.mark.asyncio
async def test_group_invite_is_auto_approved_when_policy_enabled():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_social_policy import set_social_policy
    from gateway.qq_social_requests import QqSocialRequestStore

    set_social_policy(
        auto_approve_group_invites=True,
        notify_target="qq_napcat:dm:179033731",
        updated_by="test",
    )
    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    with patch.object(adapter, "_call_api", new=AsyncMock(return_value={"ok": True})) as call_mock, \
         patch.object(adapter, "send", new=AsyncMock(return_value=SimpleNamespace(success=True, error=None))) as send_mock:
        await adapter._handle_payload(
            {
                "post_type": "request",
                "request_type": "group",
                "sub_type": "invite",
                "group_id": 987654321,
                "user_id": 179033731,
                "comment": "来群里聊项目",
                "flag": "group-auto-1",
                "time": 1713012345,
            }
        )

    stored = QqSocialRequestStore().get_request("group:group-auto-1")
    assert stored is not None
    assert stored["status"] == "approved"
    assert stored["handled_via"] == "auto_social_policy"
    assert stored["decision_note"] == "按社交自动处理策略自动通过加群邀请。"
    call_mock.assert_awaited_once_with(
        "set_group_add_request",
        {"flag": "group-auto-1", "sub_type": "invite", "approve": True},
    )
    send_mock.assert_awaited()
    notice_text = send_mock.await_args.args[1]
    assert "当前状态：approved" in notice_text
    assert "处理来源：auto_social_policy" in notice_text


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
async def test_group_message_matching_default_maga_alias_is_processed():
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
            "message_id": 1271,
            "user_id": 456789,
            "group_id": 987654321,
            "raw_message": "马嘎 看看这个",
            "message": [{"type": "text", "data": {"text": "马嘎 看看这个"}}],
            "sender": {"nickname": "Alice", "card": "AliceCard"},
        }
    )

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_active_intel_worker_group_archives_without_dispatching_even_without_static_allowlist():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_group_archive import QqGroupArchiveStore
    from gateway.qq_intel_assignments import hire_intel_worker

    hire_intel_worker(
        worker_name="钢镚",
        target_group_ref="group:987654321",
        objective="去刺探情报",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:dm:179033731",
        notify_target="qq_napcat:dm:179033731",
        updated_by="test",
        joined_groups=[{"group_id": "987654321", "group_name": "目标群"}],
    )

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": False,
                "allowed_groups": [],
                "require_mention": True,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        _group_payload(
            message_id=1272,
            user_id=456789,
            text="今天群里有动静",
            nickname="Alice",
            card="AliceCard",
        )
    )

    adapter.handle_message.assert_not_awaited()
    store = QqGroupArchiveStore()
    archived = store.list_recent_messages(group_id="987654321", limit=5)
    assert len(archived) == 1
    assert archived[0]["text"] == "今天群里有动静"


@pytest.mark.asyncio
async def test_duplicate_private_message_id_is_ignored():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )
    adapter.handle_message = AsyncMock()

    payload = {
        "post_type": "message",
        "message_type": "private",
        "message_id": 1272,
        "user_id": 456789,
        "raw_message": "hello again",
        "message": [{"type": "text", "data": {"text": "hello again"}}],
        "sender": {"nickname": "Alice"},
    }

    await adapter._handle_payload(payload)
    await adapter._handle_payload(dict(payload))

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
async def test_synthetic_fallback_response_does_not_open_group_follow_up_window():
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
            text="刚才这轮接口空转了",
            source=SessionSource(
                platform=Platform.QQ_NAPCAT,
                chat_id="987654321",
                chat_type="group",
                user_id="456789",
                user_name="Alice",
            ),
            metadata={"skip_successful_response_context": True},
        ),
        ["bot-msg-fallback"],
    )

    await adapter._handle_payload(
        {
            "post_type": "message",
            "message_type": "group",
            "self_id": 999001,
            "message_id": 1281,
            "user_id": 456789,
            "group_id": 987654321,
            "raw_message": "继续说",
            "message": [{"type": "text", "data": {"text": "继续说"}}],
            "sender": {"nickname": "Alice", "card": "AliceCard"},
        }
    )

    adapter.handle_message.assert_not_awaited()


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
                    SimpleNamespace(
                        session_key=session_key,
                        session_id="sess-1",
                        created_at=datetime.now() - timedelta(minutes=5),
                        updated_at=datetime.now() - timedelta(seconds=30),
                        last_visible_reply_at=datetime.now() - timedelta(seconds=30),
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
                    SimpleNamespace(
                        session_key=session_key,
                        session_id="sess-2",
                        created_at=datetime.now() - timedelta(minutes=5),
                        updated_at=datetime.now() - timedelta(seconds=20),
                        last_visible_reply_at=datetime.now() - timedelta(seconds=20),
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
async def test_recent_group_session_without_visible_reply_does_not_restore_follow_up_after_restart():
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
        message_id=133,
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
                    SimpleNamespace(
                        session_key=session_key,
                        session_id="sess-3",
                        created_at=datetime.now() - timedelta(minutes=5),
                        updated_at=datetime.now() - timedelta(seconds=30),
                        last_visible_reply_at=None,
                        platform=Platform.QQ_NAPCAT,
                        chat_type="group",
                    )
                ]
            ),
        )
    )

    await adapter._handle_payload(payload)

    adapter.handle_message.assert_not_awaited()


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
async def test_project_group_mode_skips_low_signal_single_message_without_explicit_trigger():
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
            message_id=2021,
            user_id=456789,
            text="今天天气真好",
            nickname="Alice",
            card="AliceCard",
        )
    )

    await asyncio.sleep(0.03)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_project_group_mode_dispatches_explicit_request_without_mention():
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
            message_id=2022,
            user_id=456789,
            text="马噶那边啥情况，看看这个怎么安排？",
            nickname="Alice",
            card="AliceCard",
        )
    )

    await asyncio.sleep(0.03)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "AliceCard: 马噶那边啥情况，看看这个怎么安排？" in event.text


@pytest.mark.asyncio
async def test_project_group_mode_dispatches_admin_message_without_mention():
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
                "admin_users": ["179033731"],
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        _group_payload(
            message_id=2023,
            user_id=179033731,
            text="看看这个事情怎么推进",
            nickname="發發發",
            card="發發發",
        )
    )

    await asyncio.sleep(0.03)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.raw_message["latest_is_admin"] is True
    assert event.raw_message["latest_user_id"] == "179033731"
    assert "[最新一条来自管理员，请优先直接响应这条消息" not in event.text
    assert "發發發: 看看这个事情怎么推进" in event.text


@pytest.mark.asyncio
async def test_collect_only_group_archives_without_dispatching_or_loading_media():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_group_archive import QqGroupArchiveStore
    from gateway.qq_group_policies import set_group_policy

    set_group_policy("987654321", mode="collect_only", updated_by="test")

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
    adapter._populate_media = AsyncMock()

    await adapter._handle_payload(
        _group_payload(
            message_id=2024,
            user_id=456789,
            text="卖冬虫夏草，私聊我",
            nickname="广告哥",
            card="广告哥",
            segments=[
                {"type": "text", "data": {"text": "卖冬虫夏草，私聊我"}},
                {"type": "image", "data": {"url": "https://example.com/ad.jpg"}},
            ],
        )
    )

    adapter.handle_message.assert_not_awaited()
    adapter._populate_media.assert_not_awaited()

    archived = QqGroupArchiveStore().list_recent_messages(group_id="987654321", limit=5)
    assert len(archived) == 1
    assert archived[0]["message_id"] == "2024"
    assert archived[0]["text"] == "卖冬虫夏草，私聊我"
    assert archived[0]["has_media"] is True


@pytest.mark.asyncio
async def test_project_group_batch_is_dropped_if_group_switches_to_collect_only_before_flush():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_group_policies import set_group_policy

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
                "project_group_mode": True,
                "group_batch_debounce_seconds": 0.05,
                "group_min_model_interval_seconds": 0.0,
                "admin_users": ["179033731"],
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        _group_payload(
            message_id=20241,
            user_id=179033731,
            text="这条不要再打到主模型",
            nickname="發發發",
            card="發發發",
        )
    )

    set_group_policy("987654321", mode="collect_only", updated_by="test")
    await asyncio.sleep(0.08)

    adapter.handle_message.assert_not_awaited()
    assert adapter._group_pending_batches.get("987654321") is None


@pytest.mark.asyncio
async def test_project_group_batch_preserves_explicit_trigger_metadata_from_omitted_message():
    from gateway.platforms.qq_napcat import QqNapCatAdapter, _BufferedGroupMessage

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": False,
                "project_group_mode": True,
                "group_batch_max_messages": 2,
            },
        )
    )

    payloads = [
        _group_payload(
            message_id=30201,
            user_id=179033731,
            text="@马嘎 看下这个",
            nickname="發發發",
            card="發發發",
            segments=[
                {"type": "at", "data": {"qq": "999001", "name": "马嘎"}},
                {"type": "text", "data": {"text": " 看下这个"}},
            ],
        ),
        _group_payload(
            message_id=30202,
            user_id=30002,
            text="我补一句上下文",
            nickname="同事A",
            card="同事A",
        ),
        _group_payload(
            message_id=30203,
            user_id=30003,
            text="再补一句结果",
            nickname="同事B",
            card="同事B",
        ),
    ]
    batch = []
    for index, payload in enumerate(payloads, start=1):
        event = adapter._build_message_event(payload)
        event.text = adapter._clean_bot_mention_text(event.text, payload)
        batch.append(
            _BufferedGroupMessage(
                event=event,
                payload=payload,
                observed_at=float(index),
            )
        )

    merged = await adapter._build_group_batch_event("987654321", batch)

    assert "@马嘎" not in merged.text
    assert "省略 1 条更早消息" in merged.text
    assert merged.metadata["group_trigger_reason"] == "require_mention_disabled"
    assert merged.metadata["explicit_group_trigger"] is True
    assert merged.metadata["explicit_group_trigger_reason"] == "bot_mention"
    assert merged.metadata["explicit_addressed"] is True
    assert merged.metadata["requires_reply"] is True


@pytest.mark.asyncio
async def test_active_intel_overlay_effective_policy_exposes_report_targets():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_intel_assignments import hire_intel_worker

    hire_intel_worker(
        worker_name="钢镚",
        target_group_ref="group:987654321",
        objective="去刺探情报",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:dm:179033731",
        notify_target="qq_napcat:dm:179033731",
        updated_by="test",
        joined_groups=[{"group_id": "987654321", "group_name": "目标群"}],
    )

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    effective = adapter._effective_group_policy("987654321")

    assert effective["mode"] == "collect_only"
    assert effective["archive_enabled"] is True
    assert effective["daily_report_enabled"] is True
    assert effective["daily_report_target"] == "qq_napcat:dm:179033731"
    assert effective["manual_report_target"] == "qq_napcat:dm:179033731"
    assert effective["daily_report_targets"] == ["qq_napcat:dm:179033731"]
    assert effective["manual_report_targets"] == ["qq_napcat:dm:179033731"]
    assert effective["notify_targets"] == ["qq_napcat:dm:179033731"]


@pytest.mark.asyncio
async def test_disabled_group_ignores_message_without_archiving():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_group_archive import QqGroupArchiveStore
    from gateway.qq_group_policies import set_group_policy

    set_group_policy("987654321", mode="disabled", updated_by="test")

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": False,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        _group_payload(
            message_id=2025,
            user_id=456789,
            text="这条不该被处理",
            nickname="Alice",
            card="AliceCard",
        )
    )

    adapter.handle_message.assert_not_awaited()
    assert QqGroupArchiveStore().list_recent_messages(group_id="987654321", limit=5) == []


@pytest.mark.asyncio
async def test_group_policy_can_enable_listening_even_when_static_group_allowlist_is_closed():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_group_archive import QqGroupArchiveStore
    from gateway.qq_group_policies import set_group_policy

    set_group_policy("987654321", mode="collect_only", updated_by="test")

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": False,
                "allowed_groups": [],
                "require_mention": True,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        _group_payload(
            message_id=20251,
            user_id=456789,
            text="这个群现在开始监听",
            nickname="Alice",
            card="AliceCard",
        )
    )

    adapter.handle_message.assert_not_awaited()
    archived = QqGroupArchiveStore().list_recent_messages(group_id="987654321", limit=5)
    assert len(archived) == 1
    assert archived[0]["message_id"] == "20251"


@pytest.mark.asyncio
async def test_group_policy_can_force_project_mode_when_global_mode_is_off():
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.qq_group_policies import set_group_policy

    set_group_policy("987654321", mode="project_mode", updated_by="test")

    adapter = QqNapCatAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "ws_url": "ws://127.0.0.1:3001",
                "allow_all_groups": True,
                "require_mention": True,
                "group_batch_debounce_seconds": 0.01,
                "admin_users": ["179033731"],
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_payload(
        _group_payload(
            message_id=2026,
            user_id=179033731,
            text="看看这个事情怎么推进",
            nickname="發發發",
            card="發發發",
        )
    )

    await asyncio.sleep(0.03)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert "發發發: 看看这个事情怎么推进" in event.text


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
                "group_min_model_interval_seconds": 0.12,
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

    await asyncio.sleep(0.10)

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


def test_mime_for_segment_detects_gif_images():
    from gateway.platforms.qq_napcat import QqNapCatAdapter

    assert QqNapCatAdapter._mime_for_segment("image", "/tmp/animated.gif") == "image/gif"


@pytest.mark.asyncio
async def test_populate_media_preserves_remote_image_source(monkeypatch):
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    async def _fake_cache(seg_type, data):
        assert seg_type == "image"
        assert data["url"] == "https://cdn.example.com/a.png"
        return "/tmp/cached-a.png", "image/png"

    monkeypatch.setattr(adapter, "_cache_segment_media", _fake_cache)

    event = MessageEvent(
        text="看图",
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1"),
    )
    payload = {
        "message": [
            {
                "type": "image",
                "data": {"url": "https://cdn.example.com/a.png"},
            }
        ]
    }

    await adapter._populate_media(event, payload)

    assert event.media_urls == ["/tmp/cached-a.png"]
    assert event.media_sources == ["https://cdn.example.com/a.png"]
    assert event.media_types == ["image/png"]


@pytest.mark.asyncio
async def test_populate_media_skips_low_value_gif_and_sticker_segments(monkeypatch):
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    cache_mock = AsyncMock(return_value=("/tmp/should-not-exist", "image/png"))
    monkeypatch.setattr(adapter, "_cache_segment_media", cache_mock)

    event = MessageEvent(
        text="看这个表情包",
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1"),
    )
    payload = {
        "message": [
            {
                "type": "image",
                "data": {"url": "https://cdn.example.com/animated.gif"},
            },
            {
                "type": "image",
                "data": {
                    "url": "https://multimedia.nt.qq.com.cn/download/qq-sticker.webp?fileid=abc"
                },
            },
        ]
    }

    await adapter._populate_media(event, payload)

    assert event.media_urls == []
    assert event.media_sources == []
    assert event.media_types == []
    cache_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_populate_media_preserves_remote_source_for_qq_signed_images(monkeypatch):
    from gateway.platforms.qq_napcat import QqNapCatAdapter
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    adapter = QqNapCatAdapter(
        PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:3001"})
    )

    async def _fake_cache(seg_type, data):
        assert seg_type == "image"
        return "/tmp/localized-qq-image.jpg", "image/jpeg"

    monkeypatch.setattr(adapter, "_cache_segment_media", _fake_cache)

    event = MessageEvent(
        text="看图",
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1"),
    )
    payload = {
        "message": [
            {
                "type": "image",
                "data": {
                    "url": (
                        "https://multimedia.nt.qq.com.cn/download"
                        "?appid=1406&fileid=abc&rkey=def"
                    )
                },
            }
        ]
    }

    await adapter._populate_media(event, payload)

    assert event.media_urls == ["/tmp/localized-qq-image.jpg"]
    assert event.media_sources == [
        "https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=abc&rkey=def"
    ]
    assert event.media_types == ["image/jpeg"]
