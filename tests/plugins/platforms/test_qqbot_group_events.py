"""Mention-only QQ intake in both delivery modes (#63765, #50780)."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.qqbot.adapter import QQAdapter


def _adapter(**extra):
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={
        "app_id": "123456", "client_secret": "test-only", "group_policy": "open",
        "dm_policy": "allowlist", "allow_from": ["dm-user"], **extra,
    }))
    adapter.handle_message = AsyncMock()
    return adapter


def _group(**changes):
    return {
        "id": "message-1", "group_openid": "group-1",
        "author": {"member_openid": "sender-1"},
        "content": "<@self> hello <@other-member>",
        "mentions": [{"member_openid": "self", "id": "self", "bot": True, "is_you": True}],
        **changes,
    }


async def _deliver(adapter, events):
    tasks = []

    def spawn(coro):
        task = asyncio.get_running_loop().create_task(coro)
        tasks.append(task)
        return task

    with patch("plugins.platforms.qqbot.adapter.asyncio.create_task", side_effect=spawn):
        for seq, (event_type, message) in enumerate(events, 1):
            adapter._dispatch_payload({"op": 0, "t": event_type, "s": seq, "d": message})
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("content,mentions,policy,allowed", [
    ("<@self> hello <@other-member>", [{"member_openid": "self", "bot": True, "is_you": True}], "open", True),
    ("@Alice hello <@!self>", [{"id": "self", "bot": True, "is_you": True}], "open", True),
    ("hello", [{"member_openid": "self", "bot": True, "is_you": True}], "allowlist", True),
    ("<@self> hello", [{"id": "self", "bot": True, "is_you": True}], "disabled", False),
    ("<@self> hello", [{"id": "self", "bot": True, "is_you": True}], "pairing", False),
    ("<@self> hello", [{"id": "self", "bot": True, "is_you": True}], "unlisted", False),
    ("<@self> hello", [{"id": "self", "bot": True}], "open", False),
    ("<@other> hello", [{"id": "other", "bot": True, "is_you": False}], "open", False),
    ("<@123456> hello", [{"id": "123456", "bot": True}], "open", False),
    ("<@self> hello", [{"id": "self", "bot": True, "is_you": "true"}], "open", False),
    ("<@self> hello", [{"id": "self", "bot": True, "is_you": 1}], "open", False),
    ("<@self> hello", [{"id": "self", "bot": "true", "is_you": True}], "open", False),
    ("<@self> hello", [{"id": "self", "bot": 1, "is_you": True}], "open", False),
    ("<@self> hello", [{"id": "self", "bot": False, "is_you": True}], "open", False),
    ("hello", [{"bot": True, "is_you": True}], "open", False),
    ("<@all> hello", [{"id": "all", "bot": True, "is_you": True}], "open", False),
    ("hello", [{"id": "bad>id", "bot": True, "is_you": True}], "open", False),
    ("hello", [{"id": 123, "bot": True, "is_you": True}], "open", False),
    ("hello", None, "open", False),
    ("hello", [], "open", False),
    ("hello", [None, "self", {"bot": True}], "open", False),
    ("hello", {"id": "self", "bot": True, "is_you": True}, "open", False),
])
async def test_full_group_intake_requires_acl_and_the_receiving_bot(content, mentions, policy, allowed):
    adapter = _adapter(
        group_policy="allowlist" if policy == "unlisted" else policy,
        group_allow_from=["another-group" if policy == "unlisted" else "group-1"],
    )
    await _deliver(adapter, [("GROUP_MESSAGE_CREATE", _group(content=content, mentions=mentions))])
    assert adapter.handle_message.await_count == int(allowed)
    if allowed:
        event = adapter.handle_message.call_args.args[0]
        assert event.source.chat_id == "group-1"
        assert event.source.user_id == "sender-1"
        assert event.source.chat_type == "group"
        assert "<@self>" not in event.text and "<@!self>" not in event.text
        assert ("<@other-member>" in event.text) == ("<@other-member>" in content)
        if content.startswith("@Alice"):
            assert event.text == "@Alice hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("events,expected", [
    ([
        ("GROUP_MESSAGE_CREATE", _group()),
        ("GROUP_AT_MESSAGE_CREATE", _group(content="@Bot legacy", mentions=[])),
    ], [("group", "hello <@other-member>")]),
    ([
        ("GROUP_AT_MESSAGE_CREATE", _group(content="@Bot legacy", mentions=[])),
        ("GROUP_MESSAGE_CREATE", _group()),
    ], [("group", "legacy")]),
    ([
        ("GROUP_MESSAGE_CREATE", _group(content="hello", mentions=[])),
        ("GROUP_AT_MESSAGE_CREATE", _group(content="@Bot legacy", mentions=[])),
    ], [("group", "legacy")]),
    ([
        ("GROUP_MESSAGE_CREATE", _group()),
        ("GROUP_MESSAGE_CREATE", _group()),
    ], [("group", "hello <@other-member>")]),
    ([
        ("GROUP_AT_MESSAGE_CREATE", _group(content="@Bot legacy", mentions=[])),
        ("C2C_MESSAGE_CREATE", {"id": "private-1", "author": {"user_openid": "dm-user"}, "content": "hi"}),
    ], [("group", "legacy"), ("dm", "hi")]),
    ([
        ("READY", {"user": {"id": "ready-self-id", "bot": True}}),
        ("GROUP_MESSAGE_CREATE", _group()),
        ("GROUP_MESSAGE_CREATE", _group(
            id="message-2", group_openid="group-2", content="<@second-self> next <@self>",
            mentions=[{"id": "second-self", "bot": True, "is_you": True}, {"id": "self", "bot": True, "is_you": False}],
        )),
        ("GROUP_MESSAGE_CREATE", _group(id="message-3", group_openid="group-2", mentions=[{"id": "self", "bot": True, "is_you": False}])),
    ], [("group", "hello <@other-member>"), ("group", "next <@self>")]),
])
async def test_group_delivery_deduplicates_without_changing_legacy_or_private_intake(events, expected):
    adapter = _adapter()
    await _deliver(adapter, events)
    assert [(c.args[0].source.chat_type, c.args[0].text) for c in adapter.handle_message.call_args_list] == expected
