"""Telegram intake authorization is the documented union, not the first matching list.

``allow_from`` / ``group_allow_from`` used to be the SOLE authority at the intake prefilter, so a
global user posting in a group, a member of a ``group_allowed_chats`` chat, or a DM user listed only
in ``TELEGRAM_ALLOWED_USERS`` was dropped before the runner's OR chain ran.
"""

from types import SimpleNamespace

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter

_EXTRA = {"allow_from": ["111"], "group_allow_from": ["222"], "group_allowed_chats": ["-1001"],
          "unauthorized_dm_behavior": "ignore"}


def _wired_adapter(monkeypatch):
    """Real adapter + the real gateway auth callback, wired as ``GatewayRunner`` does."""
    from hermes_agent.gateway.pairing import PairingStore
    from hermes_agent.gateway.run import GatewayRunner

    for key in ("GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS",
                "TELEGRAM_GROUP_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_CHATS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111,333")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra=dict(_EXTRA)))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.config.platforms = {Platform.TELEGRAM: adapter.config}
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {"default": runner.pairing_store}
    runner._primary_profile_name = "default"
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))
    return adapter


def _message(user_id, chat_id, chat_type):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id, full_name=f"U{user_id}", first_name=f"U{user_id}", is_bot=False),
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="G", full_name=None, is_forum=False),
        message_thread_id=None, is_topic_message=False, sender_chat=None, direct_messages_topic=None)


@pytest.mark.parametrize("user_id, chat_id, chat_type, admitted", [
    (111, -1002, "supergroup", True),   # global allow_from user in any group
    (444, -1001, "supergroup", True),   # any member of a group_allowed_chats chat
    (222, -1002, "supergroup", True),   # group_allow_from user (control)
    (555, -1002, "supergroup", False),  # nobody (control)
    (333, 333, "private", True),        # TELEGRAM_ALLOWED_USERS-only user in a DM
    (666, 666, "private", False),       # nobody in a DM (control)
])
def test_intake_admits_the_documented_union_and_nothing_more(monkeypatch, user_id, chat_id, chat_type, admitted):
    adapter = _wired_adapter(monkeypatch)
    assert adapter._is_user_authorized_from_message(_message(user_id, chat_id, chat_type)) is admitted
