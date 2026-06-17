"""Feishu active group context and user-centered reply gating tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from tests.gateway.feishu_helpers import (
    install_dedup_state,
    make_adapter_skeleton,
    make_message,
    make_sender,
)


def _mention(
    *,
    key: str = "@_user_1",
    name: str = "水晶",
    open_id: str = "ou_crystal",
    user_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        key=key,
        name=name,
        id=SimpleNamespace(open_id=open_id, user_id=user_id),
    )


def _text_message(
    *,
    message_id: str,
    text: str,
    chat_id: str = "oc_group",
    chat_type: str = "group",
    mentions: list | None = None,
) -> SimpleNamespace:
    message = make_message(
        message_id=message_id,
        chat_id=chat_id,
        chat_type=chat_type,
        mentions=mentions or [],
    )
    message.content = json.dumps({"text": text}, ensure_ascii=False)
    message.message_type = "text"
    message.parent_id = None
    message.upper_message_id = None
    message.root_id = None
    message.thread_id = None
    return message


def _wire_runtime(adapter, tmp_path) -> None:
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter.platform = Platform.FEISHU
    adapter._recent_context_state_path = tmp_path / "feishu_recent_context.json"


def test_human_to_human_mentions_are_observe_only_when_free_reply_enabled():
    adapter = make_adapter_skeleton(
        bot_open_id="ou_bot",
        require_mention=False,
        group_policy="open",
    )
    message = _text_message(
        message_id="om_mention",
        text="@_user_1 你看一下这个",
        mentions=[_mention()],
    )

    assert adapter._admit(make_sender(open_id="ou_jack"), message) == "human_to_human_mention"


@pytest.mark.asyncio
async def test_human_to_human_mentions_are_recorded_without_dispatch(tmp_path):
    adapter = make_adapter_skeleton(
        bot_open_id="ou_bot",
        require_mention=False,
        group_policy="open",
    )
    _wire_runtime(adapter, tmp_path)
    install_dedup_state(adapter)
    adapter._process_inbound_message = AsyncMock()

    message = _text_message(
        message_id="om_context_only",
        text="@_user_1 这条只给水晶看，机器人不要抢答",
        mentions=[_mention()],
    )
    data = SimpleNamespace(
        event=SimpleNamespace(
            sender=make_sender(open_id="ou_jack"),
            message=message,
        )
    )

    await adapter._handle_message_event_data(data)

    adapter._process_inbound_message.assert_not_called()
    context = adapter._build_recent_channel_context(
        "oc_group",
        current_message_id="om_followup",
    )
    assert "[Observed Feishu recent chat context" in context
    assert "水晶" in context
    assert "机器人不要抢答" in context


@pytest.mark.asyncio
async def test_admitted_followup_includes_recent_channel_context(tmp_path):
    adapter = make_adapter_skeleton(
        bot_open_id="ou_bot",
        require_mention=False,
        group_policy="open",
    )
    _wire_runtime(adapter, tmp_path)
    adapter.get_chat_info = AsyncMock(return_value={"name": "忠赢"})
    adapter._resolve_sender_profile = AsyncMock(
        return_value={
            "user_id": "ou_jack",
            "user_id_alt": "on_jack",
            "user_name": "袁焊忠",
        }
    )
    adapter._fetch_message_text = AsyncMock(return_value=None)
    adapter._dispatch_inbound_event = AsyncMock()
    adapter._remember_recent_message(
        chat_id="oc_group",
        message_id="om_old",
        sender_label="水晶",
        text="前面已经说过预算别超过三千",
    )

    message = _text_message(
        message_id="om_new",
        text="那你结合刚才的话给建议",
    )

    await adapter._process_inbound_message(
        data=message,
        message=message,
        sender_id=make_sender(open_id="ou_jack").sender_id,
        chat_type="group",
        message_id="om_new",
        is_bot=False,
    )

    event = adapter._dispatch_inbound_event.call_args.args[0]
    assert event.message_type == MessageType.TEXT
    assert event.text == "那你结合刚才的话给建议"
    assert event.channel_context
    assert "前面已经说过预算别超过三千" in event.channel_context
    assert "那你结合刚才的话给建议" not in event.channel_context


def test_recent_context_state_file_can_be_loaded(tmp_path):
    adapter = make_adapter_skeleton(
        bot_open_id="ou_bot",
        require_mention=False,
        group_policy="open",
    )
    _wire_runtime(adapter, tmp_path)
    adapter._remember_recent_message(
        chat_id="oc_group",
        message_id="om_old",
        sender_label="水晶",
        text="重启后也要记得这条背景",
    )

    restored = make_adapter_skeleton(
        bot_open_id="ou_bot",
        require_mention=False,
        group_policy="open",
    )
    _wire_runtime(restored, tmp_path)
    restored._load_recent_context_state()

    context = restored._build_recent_channel_context(
        "oc_group",
        current_message_id="om_new",
    )
    assert "重启后也要记得这条背景" in context


def test_recent_context_state_file_records_latest_purpose(tmp_path):
    adapter = make_adapter_skeleton(
        bot_open_id="ou_bot",
        require_mention=False,
        group_policy="open",
    )
    _wire_runtime(adapter, tmp_path)

    adapter._remember_recent_message(
        chat_id="oc_group",
        message_id="om_latest",
        sender_label="水晶",
        text="帮我总结刚才的方案",
    )

    payload = json.loads(adapter._recent_context_state_path.read_text(encoding="utf-8"))
    latest = payload["latest_by_chat"]["oc_group"]
    assert latest["latest_message_id"] == "om_latest"
    assert latest["latest_sender_label"] == "水晶"
    assert latest["latest_purpose"] == "帮我总结刚才的方案"
    assert latest["updated_at"] > 0
