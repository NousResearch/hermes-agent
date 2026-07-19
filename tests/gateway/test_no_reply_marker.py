"""Tests for the gateway's silent reply marker contract."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _make_runner(platform):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=f"agent:main:{platform.value}:group:chat-1:user-1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._update_prompt_pending = {}
    runner.pairing_store = MagicMock()
    runner._set_session_env = lambda _context: None
    runner._is_user_authorized = lambda _source: True
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._deliver_media_from_response = AsyncMock()
    runner._run_process_watcher = AsyncMock()
    runner._configured_admin_user_ids = lambda _platform: []
    runner._is_admin_user = lambda _source: False
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "[[NO_REPLY]]",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner


@pytest.mark.asyncio
async def test_no_reply_marker_suppresses_outbound_message_but_still_persists_session():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    event = MessageEvent(
        text="只是路过说一句",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        message_id="m1",
    )

    result = await runner._handle_message(event)

    # Upstream uses empty string (not None) as the stable "no delivery" signal.
    assert not result
    assert runner.session_store.append_to_transcript.called
    runner._send_voice_reply.assert_not_awaited()
    runner._deliver_media_from_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_qq_group_trigger_injects_reply_required_note_into_context_prompt():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "收到，你继续说。",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="ok",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        metadata={
            "group_trigger_reason": "require_mention_disabled",
            "explicit_group_trigger": True,
            "explicit_group_trigger_reason": "bot_mention",
        },
        message_id="m1-explicit",
    )

    result = await runner._handle_message(event)

    assert result == "收到，你继续说。"
    # Volatile group-reply note rides the per-turn sidecar (api_content), not
    # the pinned session context prompt — so turn1→turn2 system bytes stay stable.
    session_key = runner.session_store.get_or_create_session.return_value.session_key
    notes = "\n\n".join(
        getattr(runner, "_pending_turn_sidecar_notes", {}).get(session_key, [])
    )
    assert "explicitly addressed you" in notes
    assert "Do not return [[NO_REPLY]]" in notes
    assert "empty response" in notes
    assert "bot_mention" in notes
    context_prompt = runner._run_agent.await_args.kwargs["context_prompt"]
    assert "explicitly addressed you" not in context_prompt


@pytest.mark.asyncio
async def test_no_reply_marker_in_explicit_qq_group_trigger_never_stays_silent():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    event = MessageEvent(
        text="ok",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        metadata={
            "group_trigger_reason": "require_mention_disabled",
            "explicit_group_trigger": True,
            "explicit_group_trigger_reason": "bot_mention",
        },
        message_id="m1-explicit-no-reply",
    )

    result = await runner._handle_message(event)

    assert result == "收到，你继续说。"


@pytest.mark.asyncio
async def test_empty_placeholder_in_explicit_qq_group_trigger_returns_visible_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="ok",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        metadata={
            "group_trigger_reason": "require_mention_disabled",
            "explicit_group_trigger": True,
            "explicit_group_trigger_reason": "bot_mention",
        },
        message_id="m1-explicit-empty",
    )

    result = await runner._handle_message(event)

    assert result == "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"


@pytest.mark.asyncio
async def test_empty_placeholder_in_explicit_qq_group_trigger_with_empty_body_still_replies():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        metadata={
            "group_trigger_reason": "require_mention_disabled",
            "explicit_group_trigger": True,
            "explicit_group_trigger_reason": "bot_mention",
        },
        message_id="m1-explicit-empty-body",
    )

    result = await runner._handle_message(event)

    assert result == "我在，你继续说。"


@pytest.mark.asyncio
async def test_generic_explicit_group_metadata_gets_visible_fallback_outside_qq():
    platform = getattr(Platform, "WEIXIN")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="在吗",
        source=SessionSource(
            platform=platform,
            user_id="wxid-user",
            chat_id="project@chatroom",
            user_name="tester",
            chat_type="group",
        ),
        metadata={
            "explicit_addressed": True,
            "requires_reply": True,
            "address_reason": "reply_to_bot",
        },
        message_id="wx-explicit-empty",
    )

    result = await runner._handle_message(event)

    assert result == "刚才这轮没吐出正文，但消息我收到了。你再发一遍，或者我继续接着干。"
    session_key = runner.session_store.get_or_create_session.return_value.session_key
    notes = "\n\n".join(
        getattr(runner, "_pending_turn_sidecar_notes", {}).get(session_key, [])
    )
    assert "explicitly addressed you" in notes
    assert "reply_to_bot" in notes
    context_prompt = runner._run_agent.await_args.kwargs["context_prompt"]
    assert "explicitly addressed you" not in context_prompt


@pytest.mark.asyncio
async def test_empty_placeholder_suppresses_outbound_message_but_still_persists_session():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="发张图",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        message_id="m2",
    )

    result = await runner._handle_message(event)

    # Upstream uses empty string (not None) as the stable "no delivery" signal.
    assert not result
    assert runner.session_store.append_to_transcript.called
    runner._send_voice_reply.assert_not_awaited()
    runner._deliver_media_from_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_placeholder_in_dm_returns_fallback_reply():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="你在吗",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="dm",
        ),
        message_id="m3",
    )

    result = await runner._handle_message(event)

    assert result == "刚才接口抽了，没吐出正文。你再发一条，或者我继续接着刚才的话题说。"
    assert runner.session_store.append_to_transcript.called
    runner._send_voice_reply.assert_not_awaited()
    runner._deliver_media_from_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_placeholder_fallback_rewrites_persisted_assistant_message():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [
                {"role": "user", "content": "你在吗"},
                {"role": "assistant", "content": "(empty)"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="你在吗",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="dm",
        ),
        message_id="m3b",
    )

    result = await runner._handle_message(event)

    assert result == "刚才接口抽了，没吐出正文。你再发一条，或者我继续接着刚才的话题说。"
    transcript_entries = [
        call.args[1]
        for call in runner.session_store.append_to_transcript.call_args_list
        if len(call.args) >= 2
    ]
    assistant_entries = [entry for entry in transcript_entries if entry.get("role") == "assistant"]
    assert assistant_entries
    assert assistant_entries[-1]["content"] == result


@pytest.mark.asyncio
async def test_empty_placeholder_in_qq_group_with_explicit_maga_mention_returns_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="@马嘎 cursor 风格的官方网站 快落实",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        message_id="m4",
    )

    result = await runner._handle_message(event)

    assert result == "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"
    assert runner.session_store.append_to_transcript.called
    runner._send_voice_reply.assert_not_awaited()
    runner._deliver_media_from_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_placeholder_in_qq_group_with_explicit_maga_alias_returns_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="@马噶 还在不在",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        message_id="m5",
    )

    result = await runner._handle_message(event)

    assert result == "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"


@pytest.mark.asyncio
async def test_empty_placeholder_in_qq_group_from_admin_without_alias_returns_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._configured_admin_user_ids = lambda _platform: ["123456"]
    runner._is_admin_user = lambda _source: True
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="这个网关现在有问题，我重构还没完",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        message_id="m5a",
    )

    result = await runner._handle_message(event)

    assert result == "刚才这轮接口空转了，消息我收到了。你再发一遍，或者我继续接着干。"


@pytest.mark.asyncio
async def test_empty_placeholder_in_qq_group_request_without_alias_returns_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="应该如何应对这种情况？",
        source=SessionSource(
            platform=platform,
            user_id="555666",
            chat_id="987654",
            user_name="tester2",
            chat_type="group",
        ),
        message_id="m5b",
    )

    result = await runner._handle_message(event)

    assert result == "刚才这轮接口空转了，但消息我收到了。你再发一遍，或者我继续接着干。"


@pytest.mark.asyncio
async def test_empty_placeholder_in_qq_group_media_message_returns_media_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="[QQ项目群合并消息，共 1 条]\n测试用户: [图片]",
        source=SessionSource(
            platform=platform,
            user_id="777888",
            chat_id="987654",
            user_name="tester3",
            chat_type="group",
        ),
        message_id="m5c",
    )

    result = await runner._handle_message(event)

    assert result == "刚才这条带图/附件的消息我这轮没读出来。你再发一次，或者补一句文字我继续接。"


@pytest.mark.asyncio
async def test_no_reply_marker_in_qq_group_with_explicit_maga_mention_returns_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    event = MessageEvent(
        text="@马嘎 在?",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        message_id="m6",
    )

    result = await runner._handle_message(event)

    assert result == "刚才我这轮空转了，但消息我收到了。你再发一遍，或者我继续接着干。"


@pytest.mark.asyncio
async def test_no_reply_marker_in_qq_group_runtime_short_query_without_alias_returns_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    event = MessageEvent(
        text="还在吗",
        source=SessionSource(
            platform=platform,
            user_id="555666",
            chat_id="987654",
            user_name="tester-runtime",
            chat_type="group",
        ),
        message_id="m6-runtime",
    )

    result = await runner._handle_message(event)

    assert result == "刚才我这轮空转了，但我还在。你再发一遍，或者我继续接着干。"


@pytest.mark.asyncio
async def test_no_reply_marker_in_qq_group_from_admin_request_without_alias_returns_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._configured_admin_user_ids = lambda _platform: ["123456"]
    runner._is_admin_user = lambda _source: True
    event = MessageEvent(
        text="这个网关现在怎么修？",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester-admin",
            chat_type="group",
        ),
        message_id="m6-admin-request",
    )

    result = await runner._handle_message(event)

    assert result == "刚才这轮接口空转了，消息我收到了。你再发一遍，或者我继续接着干。"


@pytest.mark.asyncio
async def test_no_reply_marker_in_qq_group_from_admin_without_alias_stays_silent():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._configured_admin_user_ids = lambda _platform: ["123456"]
    runner._is_admin_user = lambda _source: True
    event = MessageEvent(
        text="这个讨论先继续，你们先说",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        message_id="m6a",
    )

    result = await runner._handle_message(event)

    # Upstream uses empty string (not None) as the stable "no delivery" signal.
    assert not result


@pytest.mark.asyncio
async def test_no_reply_marker_in_admin_group_batch_short_command_returns_fallback():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._configured_admin_user_ids = lambda _platform: ["123456"]
    runner._is_admin_user = lambda _source: True
    event = MessageEvent(
        text="[QQ项目群合并消息，共 1 条]\n發發發: 拿下她",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="發發發",
            chat_type="group",
        ),
        raw_message={
            "qq_group_batch": True,
            "latest_is_admin": True,
            "latest_user_id": "123456",
        },
        message_id="m6-admin-batch-short",
    )

    result = await runner._handle_message(event)

    assert result == "刚才这轮接口空转了，消息我收到了。你再发一遍，或者我继续接着干。"
