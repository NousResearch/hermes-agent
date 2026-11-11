"""Reactions on our own messages pass the same sender gates as text (#105157)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig


def _react_to_own_message(reactor_open_id, *, chat_type="group", extra=None):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig(extra=extra or {}))
    adapter._app_id = "cli_self_app"
    adapter._bot_open_id = "ou_self_bot"
    # GET im/v1/messages reports sender.id=app_id for bot messages: this is our own message.
    msg = SimpleNamespace(
        sender=SimpleNamespace(sender_type="app", id="cli_self_app", id_type="app_id"),
        chat_id="oc_chat", chat_type=chat_type,
    )
    response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=[msg]))
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(get=Mock(return_value=response))))
    )
    adapter._build_get_message_request = Mock(return_value=object())
    adapter._handle_message_with_guards = AsyncMock()
    adapter._resolve_sender_profile = AsyncMock(
        return_value={"user_id": reactor_open_id, "user_name": "Human", "user_id_alt": None}
    )
    adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
    event = SimpleNamespace(
        message_id="om_own_msg",
        user_id=SimpleNamespace(open_id=reactor_open_id, user_id=None, union_id=None),
        reaction_type=SimpleNamespace(emoji_type="THUMBSUP"),
    )
    asyncio.run(adapter._handle_reaction_event("im.message.reaction.created_v1", SimpleNamespace(event=event)))
    return adapter._handle_message_with_guards


@pytest.fixture(autouse=True)
def _owner_allowlist(monkeypatch):
    for key in ("FEISHU_GROUP_POLICY", "FEISHU_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "FEISHU_ADMINS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_owner")


def test_group_reaction_from_non_allowlisted_user_is_dropped():
    # Their text is dropped by _admit, so their reaction must be too.
    _react_to_own_message("ou_stranger").assert_not_awaited()


def test_group_reaction_from_allowlisted_user_is_routed_without_mention():
    guards = _react_to_own_message("ou_owner")
    guards.assert_awaited_once()
    assert guards.await_args.args[0].text == "reaction:added:THUMBSUP"


def test_group_rule_disabled_drops_reactions_from_allowlisted_users():
    extra = {"group_rules": {"oc_chat": {"policy": "disabled"}}}
    _react_to_own_message("ou_owner", extra=extra).assert_not_awaited()


def test_dm_reaction_follows_the_dm_allowlist():
    _react_to_own_message("ou_stranger", chat_type="p2p").assert_not_awaited()
    _react_to_own_message("ou_owner", chat_type="p2p").assert_awaited_once()
