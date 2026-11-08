"""A Discord allowlist written only in config.yaml (``discord.allow_from``) under multiplex.

Multiplexed authorization reads each profile's own secret scope. The default profile's
``DISCORD_ALLOWED_USERS`` (bridged from ``allow_from``) reaches it, but a secondary profile running
its own bot has its list only in its adapter's config. Either way the user must be admitted wherever
the adapter's own intake admits them: guild channels, threads and component clicks. Real
``GatewayRunner`` auth callback and real adapter; the default-profile cases also use the real config
loader.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform

USER = "202965502740398090"
STRANGER = "111111111111111111"
CHANNEL = "1501061216548683885"
THREAD = "1554296921646960760"


@pytest.fixture
def config_only_discord(monkeypatch):
    """Multiplexed default profile; the Discord allowlist lives only in config.yaml."""
    from agent import secret_scope

    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n"
        f"discord:\n  require_mention: true\n  allow_from: {USER}\n",
        encoding="utf-8",
    )
    (home / ".env").write_text("DISCORD_BOT_TOKEN=test-token\n", encoding="utf-8")
    monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    yield home
    secret_scope.set_multiplex_active(prev)


def _wire(home):
    from gateway.config import load_gateway_config
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner
    from plugins.platforms.discord.adapter import DiscordAdapter

    runner = object.__new__(GatewayRunner)
    runner.config = load_gateway_config()
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {"default": runner.pairing_store}
    runner._primary_profile_name = "default"
    runner._profile_adapters = {}
    adapter = DiscordAdapter(runner.config.platforms[Platform.DISCORD])
    runner.adapters = {Platform.DISCORD: adapter}
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.DISCORD))
    return runner, adapter


def _click(user_id, channel):
    return SimpleNamespace(
        user=SimpleNamespace(id=int(user_id), roles=[]),
        channel_id=int(getattr(channel, "id", THREAD)),
        guild=SimpleNamespace(id=1),
        channel=channel,
    )


def _thread():
    import discord

    thread = object.__new__(discord.Thread)
    return thread


def test_button_click_in_a_thread_follows_the_thread_message_verdict(config_only_discord):
    """Guard: a thread click is checked like the thread's messages, so an allowlisted user who can
    chat in a thread is not told "This bot is private" when clicking Allow Once there."""
    from plugins.platforms.discord.adapter_component_auth import _component_check_auth

    _runner, adapter = _wire(config_only_discord)
    snapshot = adapter._get_allowed_users()
    assert USER in snapshot

    click = _click(USER, _thread())
    assert _component_check_auth(click, snapshot, set(), live_auth=adapter._component_live_auth) is True
    stranger = _click(STRANGER, _thread())
    assert _component_check_auth(stranger, snapshot, set(), live_auth=adapter._component_live_auth) is False


@pytest.mark.parametrize("user_id,expected", [(USER, True), (STRANGER, False)])
def test_guild_channel_message_and_button_honor_config_allow_from(config_only_discord, user_id, expected):
    """Guard: ``allow_from`` is Discord's only user allowlist, so it governs guild channels too:
    the default bot admits the allowlisted user's channel messages and button clicks."""
    from plugins.platforms.discord.adapter_component_auth import _component_check_auth

    runner, adapter = _wire(config_only_discord)
    source = adapter.build_source(
        chat_id=CHANNEL, chat_type="group", user_id=user_id, user_name="u", guild_id="1")
    runner._canonicalize(source, primary_home=config_only_discord)
    assert runner._is_user_authorized_for_source(source) is expected

    click = _click(user_id, SimpleNamespace(id=int(CHANNEL)))
    snapshot = adapter._get_allowed_users()
    assert _component_check_auth(click, snapshot, set(), live_auth=adapter._component_live_auth) is expected


def test_telegram_group_still_requires_group_allow_from(monkeypatch):
    """Guard: Telegram's ``allow_from`` is DM-only and must not admit the user in a group."""
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.telegram.adapter import TelegramAdapter

    for key in ("TELEGRAM_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_USERS", "GATEWAY_ALLOWED_USERS",
                "GATEWAY_ALLOW_ALL_USERS", "TELEGRAM_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, extra={"allow_from": "777"})
    runner.config.platforms = {Platform.TELEGRAM: adapter.config}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.pairing_store = PairingStore(profile="default")

    def source(chat_type):
        return SessionSource(platform=Platform.TELEGRAM, chat_id="-100555", chat_type=chat_type,
                             user_id="777")

    assert runner._is_user_authorized(source("dm")) is True
    assert runner._is_user_authorized(source("group")) is False


def test_button_click_in_a_thread_resolves_a_thread_scoped_route():
    """A click carries the thread id, so the live verdict resolves a ``thread_id`` profile route
    the same way the thread's messages do (here: the routed profile's pairing store)."""
    from agent import secret_scope
    from gateway.pairing import PairingStore

    home = Path(os.environ["HERMES_HOME"])
    (home / "profiles" / "kody").mkdir(parents=True)
    (home / "profiles" / "kody" / ".env").write_text("", encoding="utf-8")
    (home / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\ndiscord:\n  require_mention: true\n"
        f"profile_routes:\n  - name: kody-thread\n    platform: discord\n    profile: kody\n"
        f"    thread_id: \"{THREAD}\"\n",
        encoding="utf-8",
    )
    (home / ".env").write_text("DISCORD_BOT_TOKEN=test-token\nDISCORD_ALLOWED_USERS=999\n", encoding="utf-8")
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        runner, adapter = _wire(home)
        kody = PairingStore(profile="kody")
        kody._save_json(kody._approved_path("discord"), {USER: {}})
        runner.pairing_stores["kody"] = kody

        message = adapter.build_source(chat_id=THREAD, chat_type="thread", user_id=USER,
                                       user_name="u", thread_id=THREAD, guild_id="1")
        runner._canonicalize(message, primary_home=home)
        assert message.profile == "kody"
        assert runner._is_user_authorized_for_source(message) is True
        assert adapter._component_live_auth(_click(USER, _thread())) is True
    finally:
        secret_scope.set_multiplex_active(prev)


@pytest.mark.parametrize("user_id,expected", [(USER, True), (STRANGER, False)])
def test_secondary_profiles_own_bot_honors_its_config_allow_from_in_guild_channels(
    monkeypatch, user_id, expected
):
    """A profile running its own Discord bot keeps its allowlist in its own config.yaml.

    Under multiplex only the default profile's list reaches the env, so the guild-channel verdict for
    the secondary's bot can come only from its adapter's ``allow_from`` (``PlatformConfig.extra``,
    where the config loader puts it), as its DMs already do.
    """
    from agent import secret_scope
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner
    from plugins.platforms.discord.adapter import DiscordAdapter

    home = Path(os.environ["HERMES_HOME"])
    (home / "profiles" / "kody").mkdir(parents=True)
    (home / ".env").write_text("DISCORD_BOT_TOKEN=test-token\nDISCORD_ALLOWED_USERS=999\n", encoding="utf-8")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "999")  # the default profile's live env
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.pairing_store = PairingStore(profile="default")
        runner.pairing_stores = {"default": runner.pairing_store, "kody": PairingStore(profile="kody")}
        runner._primary_profile_name = "default"
        runner.adapters = {}
        adapter = DiscordAdapter(PlatformConfig(
            enabled=True, token="kody-token", extra={"allow_from": USER, "require_mention": True}))
        adapter._hermes_profile_name = "kody"
        runner._profile_adapters = {"kody": {Platform.DISCORD: adapter}}
        adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.DISCORD, profile_name="kody"))

        assert adapter._is_sender_authorized(user_id, "dm", CHANNEL) is expected
        assert adapter._is_sender_authorized(user_id, "group", CHANNEL) is expected
    finally:
        secret_scope.set_multiplex_active(prev)
