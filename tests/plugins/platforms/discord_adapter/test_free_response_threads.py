"""Tests for ``discord.free_response_thread_mode`` and ``discord.free_response_overrides``.

``free_response_channels`` alone makes every thread under a listed channel mention-free (the
thread inherits its parent). ``free_response_thread_mode`` (inherit | never | always) and
``free_response_overrides`` (ID -> bool) let an operator opt a thread out, or in.
"""

import json
import os
import sys
from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
    discord_mod.Message = type("Message", (), {})
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord import free_response as discord_free_response
from plugins.platforms.discord.free_response import FreeResponse, FreeResponseSettings, parse_overrides

PARENT_ID = 1000
THREAD_ID = 2000
OTHER_THREAD_ID = 2001


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeTextChannel:
    def __init__(self, channel_id: int = PARENT_ID, name: str = "shared"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name="Hermes Server")
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


class FakeThread:
    def __init__(self, channel_id: int = THREAD_ID, name: str = "lane", parent=None, owner_id=None):
        self.id = channel_id
        self.name = name
        self.owner_id = owner_id
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name="Hermes Server")
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


_ENV_VARS = (
    "DISCORD_REQUIRE_MENTION",
    "DISCORD_THREAD_REQUIRE_MENTION",
    "DISCORD_FREE_RESPONSE_CHANNELS",
    "DISCORD_FREE_RESPONSE_THREAD_MODE",
    "DISCORD_FREE_RESPONSE_OVERRIDES",
    "DISCORD_FREE_RESPONSE_AUTO_THREAD",
    "DISCORD_AUTO_THREAD",
    "DISCORD_NO_THREAD_CHANNELS",
    "DISCORD_ALLOWED_CHANNELS",
    "DISCORD_IGNORED_CHANNELS",
    "DISCORD_HISTORY_BACKFILL",
    "DISCORD_ALLOW_BOTS",
    "DISCORD_IGNORE_NO_MENTION",
    "DISCORD_ALLOW_ALL_USERS",
)


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    config = PlatformConfig(enabled=True, token="fake-token", extra=dict(extra))
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.fixture
def make_adapter(monkeypatch):
    return lambda **extra: _make_adapter(monkeypatch, **extra)


def _message(channel, content="hello"):
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=[],
        attachments=[],
        reference=None,
        created_at=datetime.now(UTC),
        channel=channel,
        author=SimpleNamespace(id=42, display_name="Jezza", name="Jezza", bot=False),
        type=discord_platform.discord.MessageType.default,
        guild=SimpleNamespace(id=1),
    )


def _thread(thread_id=THREAD_ID, parent_id=PARENT_ID, name="lane", parent_name="shared"):
    return FakeThread(thread_id, name, parent=FakeTextChannel(parent_id, parent_name))


def _free(adapter, channel) -> bool:
    parent_id = adapter._get_parent_channel_id(channel) if isinstance(channel, FakeThread) else None
    return adapter._discord_free_response(_message(channel), parent_id) is FreeResponse.FREE


# --- resolution -------------------------------------------------------------------------------


def test_inherit_is_the_default_and_threads_follow_the_parent(make_adapter):
    adapter = make_adapter(free_response_channels=str(PARENT_ID))
    assert _free(adapter, FakeTextChannel(PARENT_ID)) is True
    assert _free(adapter, _thread()) is True
    assert _free(adapter, _thread(parent_id=555)) is False
    assert _free(adapter, FakeTextChannel(555)) is False


def test_inherit_matches_parent_by_name_as_before(make_adapter):
    adapter = make_adapter(free_response_channels="#shared")
    assert _free(adapter, _thread()) is True
    assert _free(adapter, _thread(parent_name="other")) is False


def test_wildcard_keeps_working(make_adapter):
    adapter = make_adapter(free_response_channels="*")
    assert _free(adapter, FakeTextChannel(555)) is True
    assert _free(adapter, _thread(parent_id=555)) is True


def test_never_requires_the_thread_itself_to_be_listed(make_adapter):
    adapter = make_adapter(free_response_channels=f"{PARENT_ID},{OTHER_THREAD_ID}",
                           free_response_thread_mode="never")
    assert _free(adapter, FakeTextChannel(PARENT_ID)) is True
    assert _free(adapter, _thread(THREAD_ID)) is False
    assert _free(adapter, _thread(OTHER_THREAD_ID)) is True


def test_never_matches_a_listed_thread_by_name(make_adapter):
    adapter = make_adapter(free_response_channels=f"{PARENT_ID},#lane", free_response_thread_mode="never")
    assert _free(adapter, _thread(name="lane")) is True
    assert _free(adapter, _thread(name="other")) is False


def test_never_does_not_inherit_the_wildcard(make_adapter):
    adapter = make_adapter(free_response_channels="*", free_response_thread_mode="never")
    assert _free(adapter, FakeTextChannel(555)) is True
    assert _free(adapter, _thread(parent_id=555)) is False


def test_always_makes_every_thread_free_but_not_plain_channels(make_adapter):
    adapter = make_adapter(free_response_channels="", free_response_thread_mode="always")
    assert _free(adapter, _thread(parent_id=555)) is True
    assert _free(adapter, FakeTextChannel(555)) is False


def test_thread_override_true_beats_never(make_adapter):
    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never",
                           free_response_overrides={str(THREAD_ID): True})
    assert _free(adapter, _thread(THREAD_ID)) is True
    assert _free(adapter, _thread(OTHER_THREAD_ID)) is False


def test_thread_override_false_beats_inherit_always_and_listing(make_adapter):
    for mode in ("inherit", "always"):
        adapter = make_adapter(free_response_channels=f"{PARENT_ID},{THREAD_ID}", free_response_thread_mode=mode,
                               free_response_overrides={str(THREAD_ID): False})
        assert _free(adapter, _thread(THREAD_ID)) is False
        assert _free(adapter, _thread(OTHER_THREAD_ID)) is True


def test_parent_override_applies_to_inheriting_threads_only(make_adapter):
    adapter = make_adapter(free_response_channels=str(PARENT_ID),
                           free_response_overrides={str(PARENT_ID): False})
    assert _free(adapter, FakeTextChannel(PARENT_ID)) is False
    assert _free(adapter, _thread()) is False  # inherit: the parent's override
    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never",
                           free_response_overrides={str(PARENT_ID): True})
    assert _free(adapter, FakeTextChannel(PARENT_ID)) is True
    assert _free(adapter, _thread()) is False  # never: the parent is not consulted


def test_parent_override_true_frees_threads_of_an_unlisted_parent(make_adapter):
    adapter = make_adapter(free_response_overrides={str(PARENT_ID): True})
    assert _free(adapter, _thread()) is True


def test_channel_override_beats_listing_and_wildcard(make_adapter):
    adapter = make_adapter(free_response_channels="*", free_response_overrides={"555": False})
    assert _free(adapter, FakeTextChannel(555)) is False
    assert _free(adapter, FakeTextChannel(556)) is True


def test_override_by_name_and_id_precedence(make_adapter):
    adapter = make_adapter(free_response_overrides={"#shared": False, str(PARENT_ID): True})
    assert _free(adapter, FakeTextChannel(PARENT_ID)) is True  # the ID is checked before the name


# --- parsing ------------------------------------------------------------------------------------


def test_env_mirrors_are_read_when_config_is_unset(make_adapter, monkeypatch):
    adapter = make_adapter()  # clears these vars, so set them afterwards
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", str(PARENT_ID))
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_THREAD_MODE", " Never ")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_OVERRIDES", json.dumps({str(OTHER_THREAD_ID): True}))
    assert _free(adapter, _thread(OTHER_THREAD_ID)) is True
    assert _free(adapter, _thread(THREAD_ID)) is False


def test_config_wins_over_env_like_free_response_channels(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_THREAD_MODE", "always")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_OVERRIDES", json.dumps({str(THREAD_ID): True}))
    adapter = make_adapter(free_response_thread_mode="never", free_response_overrides={str(THREAD_ID): False})
    assert adapter._free_response_settings() == FreeResponseSettings("never", {str(THREAD_ID): False})


def test_overrides_env_is_json_only(caplog):
    assert parse_overrides('{"2000": true, "2001": "off"}') == {"2000": True, "2001": False}
    assert parse_overrides("2000=true") == {}
    assert "not valid JSON" in caplog.text


def test_overrides_skip_bad_values_and_warn(caplog):
    assert parse_overrides({"2000": True, "2001": "maybe"}) == {"2000": True}
    assert "2001" in caplog.text and "allowed values: true, false" in caplog.text


def test_overrides_yaml_list_warns_and_is_ignored(make_adapter, caplog):
    adapter = make_adapter(free_response_overrides=["2000", "2001"])
    assert adapter._free_response_settings().overrides == {}
    assert "expected a map" in caplog.text


def test_yaml_int_keys_and_bool_values(make_adapter):
    assert parse_overrides({THREAD_ID: True, "2001": "false"}) == {str(THREAD_ID): True, "2001": False}
    adapter = make_adapter(free_response_overrides={THREAD_ID: False}, free_response_channels=str(PARENT_ID))
    assert _free(adapter, _thread()) is False


def test_invalid_mode_warns_once_naming_key_and_allowed_values(make_adapter, caplog):
    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="sometimes")
    for _ in range(3):
        assert _free(adapter, _thread()) is True  # falls back to inherit
    warnings = [r for r in caplog.records if "free_response_thread_mode" in r.getMessage()]
    assert len(warnings) == 1
    assert "allowed: inherit, never, always" in warnings[0].getMessage()


def test_settings_are_parsed_once_not_per_message(make_adapter, monkeypatch):
    adapter = make_adapter(free_response_channels=str(PARENT_ID))
    calls = []
    real = discord_free_response.derive_settings
    monkeypatch.setattr(discord_free_response, "derive_settings", lambda *a, **k: calls.append(1) or real(*a, **k))
    for _ in range(5):
        _free(adapter, _thread())
    assert len(calls) == 1


def test_live_reload_applies_new_values(make_adapter):
    adapter = make_adapter(free_response_channels=str(PARENT_ID))
    assert _free(adapter, _thread()) is True
    adapter.config.extra["free_response_thread_mode"] = "never"
    assert _free(adapter, _thread()) is True  # unchanged until a reload re-derives
    adapter._rederive_live_gates()
    assert _free(adapter, _thread()) is False
    adapter.config.extra["free_response_overrides"] = {str(THREAD_ID): True}
    adapter._rederive_live_gates()
    assert _free(adapter, _thread()) is True


def test_connect_gates_derive_the_settings(make_adapter):
    adapter = make_adapter(free_response_thread_mode="never")
    adapter._mark_live_gate_baseline = lambda: None
    adapter._derive_connect_gates()
    assert adapter._free_response_settings().thread_mode == "never"


def test_new_keys_follow_config_yaml_live():
    from plugins.platforms.discord.adapter_live_gates import _LIVE_GATE_EXTRA_KEYS
    assert {"free_response_thread_mode", "free_response_overrides"} <= set(_LIVE_GATE_EXTRA_KEYS)


def test_settings_are_per_adapter_and_use_the_adapters_env_snapshot(make_adapter, monkeypatch):
    """Two profiles on one process: each adapter reads its own config and its own gate-env snapshot."""
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_THREAD_MODE", "always")  # ambient env: another profile's
    one = make_adapter(free_response_channels=str(PARENT_ID))
    one._gate_env_snapshot = {"DISCORD_FREE_RESPONSE_THREAD_MODE": "never"}
    two = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="inherit")
    assert one._free_response_settings().thread_mode == "never"
    assert two._free_response_settings().thread_mode == "inherit"
    assert _free(one, _thread()) is False
    assert _free(two, _thread()) is True


def test_yaml_bridge_seeds_extra_and_env(monkeypatch):
    for name in ("DISCORD_FREE_RESPONSE_THREAD_MODE", "DISCORD_FREE_RESPONSE_OVERRIDES"):
        # Register the absent state so the bridge's env writes are undone at teardown.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    extra = discord_platform._apply_yaml_config(
        {}, {"free_response_thread_mode": "Never", "free_response_overrides": {2000: True}})
    assert extra["free_response_thread_mode"] == "never"
    assert extra["free_response_overrides"] == {2000: True}
    assert os.environ["DISCORD_FREE_RESPONSE_THREAD_MODE"] == "never"
    assert json.loads(os.environ["DISCORD_FREE_RESPONSE_OVERRIDES"]) == {"2000": True}


# --- decision paths -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_ingress_inherit_unchanged_and_never_requires_mention(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_channels=str(PARENT_ID))
    await adapter._handle_message(_message(_thread()))
    adapter.handle_message.assert_awaited_once()

    # Answering joins the thread (persisted), so the opt-out half uses a different, unjoined thread.
    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never")
    assert await adapter._handle_message(_message(_thread(OTHER_THREAD_ID))) is False
    adapter.handle_message.assert_not_awaited()
    # The parent channel itself is still free-response.
    await adapter._handle_message(_message(FakeTextChannel(PARENT_ID)))
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_ingress_listed_thread_is_free_under_never(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_channels=f"{PARENT_ID},{THREAD_ID}", free_response_thread_mode="never")
    await adapter._handle_message(_message(_thread()))
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_ingress_mention_still_works_under_never(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never")
    message = _message(_thread(), content="<@999> hi")
    message.mentions = [adapter._client.user]
    await adapter._handle_message(message)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_thread_require_mention_with_never_gates_a_joined_thread(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never",
                           thread_require_mention=True)
    await adapter._threads.mark_async(str(THREAD_ID))
    assert await adapter._handle_message(_message(_thread())) is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_path_honors_the_setting(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never")
    adapter._handle_message = AsyncMock(return_value=True)
    assert await adapter._dispatch_recovered_message(_message(_thread())) is False
    adapter._handle_message.assert_not_awaited()
    assert await adapter._dispatch_recovered_message(_message(_thread(OTHER_THREAD_ID))) is False

    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never",
                           free_response_overrides={str(THREAD_ID): True})
    adapter._handle_message = AsyncMock(return_value=True)
    assert await adapter._dispatch_recovered_message(_message(_thread())) is True
    adapter._handle_message.assert_awaited_once()

    adapter = make_adapter(free_response_channels=str(PARENT_ID))
    adapter._handle_message = AsyncMock(return_value=True)
    assert await adapter._dispatch_recovered_message(_message(_thread())) is True


def test_admission_third_party_mention_in_unjoined_thread_follows_the_setting(make_adapter, monkeypatch):
    monkeypatch.delenv("DISCORD_IGNORE_NO_MENTION", raising=False)

    def third_party(channel):
        message = _message(channel, content="<@820> hi")
        message.mentions = [SimpleNamespace(id=820, bot=False)]
        return message

    adapter = make_adapter(free_response_channels=str(PARENT_ID))
    adapter._is_allowed_user = MagicMock(return_value=True)
    assert adapter._discord_message_admission(third_party(_thread()), claim=False)[0] is True

    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never")
    adapter._is_allowed_user = MagicMock(return_value=True)
    assert adapter._discord_message_admission(third_party(_thread()), claim=False)[0] is False


def test_allowed_and_ignored_channel_keys_still_include_the_parent(make_adapter):
    adapter = make_adapter(free_response_thread_mode="never")
    keys = adapter._discord_channel_keys(_message(_thread()), str(PARENT_ID))
    assert {str(THREAD_ID), "lane", "#lane", str(PARENT_ID), "shared", "#shared"} <= keys


# --- joined threads -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_never_silences_a_thread_the_bot_answered_in_before_the_switch(make_adapter, monkeypatch):
    """The motivating case: a bot answered in a shared channel's threads under ``inherit``, then the
    operator sets ``never``. Participation is persisted, so without the opt-out it would keep
    answering every message in those threads forever."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_channels=str(PARENT_ID))
    assert await adapter._handle_message(_message(_thread())) is True  # joins THREAD_ID

    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never")
    assert str(THREAD_ID) in adapter._threads
    assert await adapter._handle_message(_message(_thread())) is False
    assert await adapter._dispatch_recovered_message(_message(_thread())) is False
    message = _message(_thread(), content="<@999> now you")
    message.mentions = [adapter._client.user]
    assert await adapter._handle_message(message) is True  # a mention still works...
    assert await adapter._handle_message(_message(_thread())) is False  # ...one message at a time


@pytest.mark.asyncio
async def test_thread_override_false_beats_participation(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_overrides={str(THREAD_ID): False})
    await adapter._threads.mark_async(str(THREAD_ID))
    assert await adapter._handle_message(_message(_thread())) is False


@pytest.mark.asyncio
async def test_joined_threads_outside_free_channels_keep_the_bypass(make_adapter, monkeypatch):
    """An auto-thread from an @mention in an ordinary channel, and a thread the bot itself created
    under a free channel (auto-thread, ``/thread``), still continue without a mention."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_channels=str(PARENT_ID), free_response_thread_mode="never")
    await adapter._threads.mark_async(str(OTHER_THREAD_ID))
    assert await adapter._handle_message(_message(_thread(OTHER_THREAD_ID, parent_id=555))) is True
    await adapter._threads.mark_async(str(THREAD_ID))
    own = FakeThread(THREAD_ID, "lane", parent=FakeTextChannel(PARENT_ID, "shared"), owner_id=999)
    assert await adapter._handle_message(_message(own)) is True


# --- auto-threading -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_override_true_makes_a_channel_free_so_it_skips_auto_thread(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_overrides={"555": True})
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)  # default true
    adapter._auto_create_thread = AsyncMock()
    await adapter._handle_message(_message(FakeTextChannel(555)))
    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_override_true_with_free_response_auto_thread_threads_the_message(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_overrides={"555": True})
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_AUTO_THREAD", "true")
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    adapter._auto_create_thread = AsyncMock(return_value=FakeThread(456, "auto"))
    message = _message(FakeTextChannel(555))
    await adapter._handle_message(message)
    adapter._auto_create_thread.assert_awaited_once_with(message)


@pytest.mark.asyncio
async def test_override_false_on_a_listed_channel_restores_mention_gating(make_adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter = make_adapter(free_response_channels="555", free_response_overrides={"555": False})
    assert await adapter._handle_message(_message(FakeTextChannel(555))) is False
    adapter.handle_message.assert_not_awaited()
