"""Telegram ``allow_from`` is global (DMs + groups), on a multiplexed secondary profile as on default.

telegram.md documents ``allow_from`` as "Global access (DMs + groups)". The default profile got that
through the YAML-to-env bridge (``TELEGRAM_ALLOWED_USERS``); a secondary profile running its own bot
gets no bridge, so its group verdict came from ``group_allow_from`` alone and an ``allow_from`` user
was dropped in groups. Two real profiles under a temp ``HERMES_HOME``, real config loader, real
adapter wired to the real ``GatewayRunner`` auth callback.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_agent.gateway.config import Platform

_YAML = ("telegram:\n  allow_from: ['111']\n  group_allow_from: ['222']\n"
         "  unauthorized_dm_behavior: ignore\n")
_AUTH_ENV = ("TELEGRAM_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_CHATS",
             "TELEGRAM_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS")


@pytest.fixture
def two_profiles(monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    secondary = home / "profiles" / "bot2"
    secondary.mkdir(parents=True)
    (home / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n" + _YAML, encoding="utf-8")
    (home / ".env").write_text("TELEGRAM_BOT_TOKEN=111:default\n", encoding="utf-8")
    (secondary / "config.yaml").write_text(_YAML, encoding="utf-8")
    (secondary / ".env").write_text("TELEGRAM_BOT_TOKEN=222:bot2\n", encoding="utf-8")
    for key in _AUTH_ENV:  # set-then-delete so the default profile's bridge writes are undone too
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    from hermes_agent.agent import secret_scope
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    yield home, secondary
    secret_scope.set_multiplex_active(prev)


def _wire(home, secondary, profile):
    from hermes_agent.gateway.config import GatewayConfig, load_gateway_config
    from hermes_agent.gateway.pairing import PairingStore
    from hermes_agent.gateway.run import GatewayRunner, _profile_runtime_scope
    from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {"default": runner.pairing_store, "bot2": PairingStore(profile="bot2")}
    runner._primary_profile_name = "default"
    if profile == "default":
        runner.config = load_gateway_config()
        adapter = TelegramAdapter(runner.config.platforms[Platform.TELEGRAM])
        runner.adapters, runner._profile_adapters = {Platform.TELEGRAM: adapter}, {}
        adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))
        return adapter
    runner.config = GatewayConfig(multiplex_profiles=True)
    with _profile_runtime_scope(secondary):  # as _load_secondary_profile_config loads it
        cfg = load_gateway_config()
    assert not any(os.environ.get(key) for key in _AUTH_ENV), "secondary YAML leaked into the env bridge"
    adapter = TelegramAdapter(cfg.platforms[Platform.TELEGRAM])
    adapter._hermes_profile_name = "bot2"
    runner.adapters, runner._profile_adapters = {}, {"bot2": {Platform.TELEGRAM: adapter}}
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM, profile_name="bot2"))
    return adapter


def _message(user_id, chat_id, chat_type):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id, full_name=f"U{user_id}", first_name=f"U{user_id}", is_bot=False),
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="G", full_name=None, is_forum=False),
        message_thread_id=None, is_topic_message=False, sender_chat=None, direct_messages_topic=None)


@pytest.mark.parametrize("profile", ["default", "bot2"])
@pytest.mark.parametrize("user_id, chat_id, chat_type, admitted", [
    (111, -1002, "supergroup", True),   # allow_from user in a group: the ruling
    (111, 111, "private", True),        # allow_from user in a DM
    (222, -1002, "supergroup", True),   # group_allow_from user in a group
    (222, 222, "private", False),       # group_allow_from grants no DM access
    (555, -1002, "supergroup", False),  # on no list: denied in a group
    (666, 666, "private", False),       # on no list: denied in a DM
])
def test_allow_from_is_global_on_every_profile(two_profiles, profile, user_id, chat_id, chat_type, admitted):
    adapter = _wire(*two_profiles, profile)
    assert adapter._is_user_authorized_from_message(_message(user_id, chat_id, chat_type)) is admitted
    # The runner's verdict (what the message path re-checks after intake) agrees.
    assert adapter._is_sender_authorized(
        str(user_id), chat_type="dm" if chat_type == "private" else "group", chat_id=str(chat_id)) is admitted
