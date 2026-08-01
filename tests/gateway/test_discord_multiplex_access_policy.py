"""Regression tests for Discord authorization isolation in multiplex mode."""

from __future__ import annotations

import os
import weakref
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import secret_scope as ss
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
import plugins.platforms.discord.adapter as discord_adapter_module
from plugins.platforms.discord.adapter import (
    ChoicePickerView,
    ClarifyChoiceView,
    DiscordAdapter,
    ExecApprovalView,
    ModelPickerView,
    SlashConfirmView,
    UpdatePromptView,
    _apply_yaml_config,
    discord,
)


@pytest.fixture(autouse=True)
def _isolate_multiplex_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    for name in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_ALLOW_ALL_USERS",
        "DISCORD_ALLOW_BOTS",
        "DISCORD_BOTS_REQUIRE_INLINE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_THREAD_REQUIRE_MENTION",
        "DISCORD_MISSED_MESSAGE_BACKFILL",
        "DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS",
        "DISCORD_MISSED_MESSAGE_BACKFILL_WINDOW_SECONDS",
        "DISCORD_MISSED_MESSAGE_BACKFILL_LIMIT",
        "DISCORD_MISSED_MESSAGE_BACKFILL_MAX_DISPATCHES",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _adapter_in_scope(
    secrets: dict[str, str], *, extra: dict | None = None
) -> DiscordAdapter:
    token = ss.set_secret_scope(secrets)
    try:
        return DiscordAdapter(
            PlatformConfig(enabled=True, token="test-token", extra=extra or {})
        )
    finally:
        ss.reset_secret_scope(token)


def _slash_interaction(user_id: str, channel_id: str) -> SimpleNamespace:
    guild = SimpleNamespace(
        id=400000000000000001,
        get_member=lambda _uid: None,
    )
    return SimpleNamespace(
        user=SimpleNamespace(id=int(user_id), roles=[]),
        guild=guild,
        guild_id=guild.id,
        channel_id=int(channel_id),
        channel=SimpleNamespace(id=int(channel_id)),
    )


def _message(user_id: str, channel_id: str) -> SimpleNamespace:
    guild = SimpleNamespace(
        id=400000000000000001,
        name="Synthetic Guild",
        get_member=lambda _uid: None,
    )
    channel = SimpleNamespace(
        id=int(channel_id),
        name=f"channel-{channel_id}",
        guild=guild,
        topic=None,
    )
    return SimpleNamespace(
        id=500000000000000001,
        content="hello",
        mentions=[],
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        guild=guild,
        channel=channel,
        author=SimpleNamespace(
            id=int(user_id),
            display_name="Synthetic User",
            name="synthetic-user",
            roles=[],
            guild=guild,
            bot=False,
        ),
    )


def _prepare_message_adapter(adapter: DiscordAdapter) -> AsyncMock:
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=900000000000000001, name="Synthetic Bot")
    )
    adapter._text_batch_delay_seconds = 0
    handler = AsyncMock()
    adapter.handle_message = handler
    return handler


def test_profile_user_allowlists_are_isolated_in_either_startup_order():
    """Each adapter must snapshot its own profile policy, never process globals."""
    ss.set_multiplex_active(True)
    profile_a = {"DISCORD_ALLOWED_USERS": "100000000000000001"}
    profile_b = {"DISCORD_ALLOWED_USERS": "100000000000000002"}

    for first, second in ((profile_a, profile_b), (profile_b, profile_a)):
        first_adapter = _adapter_in_scope(first)
        second_adapter = _adapter_in_scope(second)

        first_user = first["DISCORD_ALLOWED_USERS"]
        second_user = second["DISCORD_ALLOWED_USERS"]
        assert first_adapter._is_allowed_user(first_user) is True
        assert first_adapter._is_allowed_user(second_user) is False
        assert second_adapter._is_allowed_user(second_user) is True
        assert second_adapter._is_allowed_user(first_user) is False


def test_profile_role_channel_and_allow_all_policies_are_isolated():
    """Every Discord admission input must remain adapter-local."""
    ss.set_multiplex_active(True)
    profile_a = {
        "DISCORD_ALLOWED_ROLES": "200000000000000001",
        "DISCORD_ALLOWED_CHANNELS": "300000000000000001",
        "DISCORD_FREE_RESPONSE_CHANNELS": "300000000000000021",
    }
    profile_b = {
        "DISCORD_ALLOWED_ROLES": "200000000000000002",
        "DISCORD_ALLOWED_CHANNELS": "300000000000000002",
        "DISCORD_FREE_RESPONSE_CHANNELS": "300000000000000022",
    }

    adapter_a = _adapter_in_scope(profile_a)
    adapter_b = _adapter_in_scope(profile_b)
    guild = SimpleNamespace(id=400000000000000001, get_member=lambda _uid: None)
    member_a = SimpleNamespace(
        id=100000000000000001,
        guild=guild,
        roles=[SimpleNamespace(id=200000000000000001)],
    )
    member_b = SimpleNamespace(
        id=100000000000000002,
        guild=guild,
        roles=[SimpleNamespace(id=200000000000000002)],
    )

    assert (
        adapter_a._is_allowed_user(str(member_a.id), author=member_a, guild=guild)
        is True
    )
    assert (
        adapter_a._is_allowed_user(str(member_b.id), author=member_b, guild=guild)
        is False
    )
    assert (
        adapter_b._is_allowed_user(str(member_b.id), author=member_b, guild=guild)
        is True
    )
    assert (
        adapter_b._is_allowed_user(str(member_a.id), author=member_a, guild=guild)
        is False
    )
    assert (
        adapter_a._discord_channel_ids_allowed({profile_a["DISCORD_ALLOWED_CHANNELS"]})
        is True
    )
    assert (
        adapter_a._discord_channel_ids_allowed({profile_b["DISCORD_ALLOWED_CHANNELS"]})
        is False
    )
    assert (
        adapter_b._discord_channel_ids_allowed({profile_b["DISCORD_ALLOWED_CHANNELS"]})
        is True
    )
    assert (
        adapter_b._discord_channel_ids_allowed({profile_a["DISCORD_ALLOWED_CHANNELS"]})
        is False
    )
    assert adapter_a._discord_free_response_channels() == {
        profile_a["DISCORD_FREE_RESPONSE_CHANNELS"]
    }
    assert adapter_b._discord_free_response_channels() == {
        profile_b["DISCORD_FREE_RESPONSE_CHANNELS"]
    }

    discord_open = _adapter_in_scope({"DISCORD_ALLOW_ALL_USERS": "true"})
    gateway_open = _adapter_in_scope({"GATEWAY_ALLOW_ALL_USERS": "yes"})
    closed = _adapter_in_scope({})
    assert discord_open._is_allowed_user("100000000000000099") is True
    assert gateway_open._is_allowed_user("100000000000000099") is True
    assert closed._is_allowed_user("100000000000000099") is False


def test_yaml_policy_does_not_mutate_shared_env_in_multiplex(monkeypatch):
    """Scoped/multiplex config loading must hand off via extras only."""
    policy_env = {
        "DISCORD_ALLOWED_USERS": "100000000000000001",
        "DISCORD_FREE_RESPONSE_CHANNELS": "300000000000000021",
        "DISCORD_ALLOWED_CHANNELS": "300000000000000001",
        "DISCORD_IGNORED_CHANNELS": "300000000000000011",
        "DISCORD_REQUIRE_MENTION": "false",
        "DISCORD_THREAD_REQUIRE_MENTION": "true",
        "DISCORD_BOTS_REQUIRE_INLINE_MENTION": "yes",
    }
    for name in policy_env:
        monkeypatch.delenv(name, raising=False)

    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({})
    try:
        extra = _apply_yaml_config(
            {},
            {
                "allow_from": [policy_env["DISCORD_ALLOWED_USERS"]],
                "free_response_channels": [
                    policy_env["DISCORD_FREE_RESPONSE_CHANNELS"]
                ],
                "allowed_channels": [policy_env["DISCORD_ALLOWED_CHANNELS"]],
                "ignored_channels": [policy_env["DISCORD_IGNORED_CHANNELS"]],
                "require_mention": False,
                "thread_require_mention": True,
                "bots_require_inline_mention": True,
            },
        )
    finally:
        ss.reset_secret_scope(token)

    assert extra is not None
    assert all(name not in os.environ for name in policy_env)

    ss.set_multiplex_active(False)
    _apply_yaml_config({}, {"allow_from": ["100000000000000099"]})
    assert os.environ["DISCORD_ALLOWED_USERS"] == "100000000000000099"


def test_mention_history_and_approval_controls_are_profile_local(monkeypatch):
    """Non-user Discord admission/history controls must not leak through env."""
    env_names = (
        "DISCORD_IGNORE_NO_MENTION",
        "DISCORD_HISTORY_BACKFILL",
        "DISCORD_HISTORY_BACKFILL_LIMIT",
        "DISCORD_APPROVAL_MENTIONS",
    )
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({})
    try:
        extra_a = _apply_yaml_config(
            {},
            {
                "allow_from": ["100000000000000001"],
                "ignore_no_mention": True,
                "history_backfill": True,
                "history_backfill_limit": 17,
                "approval_mentions": True,
            },
        )
        extra_b = _apply_yaml_config(
            {},
            {
                "allow_from": ["100000000000000002"],
                "ignore_no_mention": False,
                "history_backfill": False,
                "history_backfill_limit": 3,
                "approval_mentions": False,
            },
        )
    finally:
        ss.reset_secret_scope(token)

    assert all(name not in os.environ for name in env_names)
    assert extra_a is not None
    assert extra_b is not None

    adapter_a = _adapter_in_scope({}, extra=extra_a)
    adapter_b = _adapter_in_scope({}, extra=extra_b)
    assert adapter_a._discord_ignore_no_mention() is True
    assert adapter_b._discord_ignore_no_mention() is False
    assert adapter_a._discord_history_backfill() is True
    assert adapter_b._discord_history_backfill() is False
    assert adapter_a._discord_history_backfill_limit() == 17
    assert adapter_b._discord_history_backfill_limit() == 3
    assert adapter_a._approval_mention_content() == "<@100000000000000001>"
    assert adapter_b._approval_mention_content() is None


def test_scoped_env_policy_survives_unscoped_primary_adapter_creation(monkeypatch):
    """Primary startup must retain scoped .env policy after config loading."""
    scoped_policy = {
        "DISCORD_ALLOWED_USERS": "100000000000000001",
        "DISCORD_ALLOWED_ROLES": "200000000000000001",
        "DISCORD_ALLOWED_CHANNELS": "300000000000000001",
        "DISCORD_IGNORED_CHANNELS": "300000000000000011",
        "DISCORD_FREE_RESPONSE_CHANNELS": "300000000000000021",
        "DISCORD_ALLOW_ALL_USERS": "false",
        "DISCORD_ALLOW_BOTS": "none",
        "DISCORD_BOTS_REQUIRE_INLINE_MENTION": "yes",
        "DISCORD_REQUIRE_MENTION": "false",
        "DISCORD_THREAD_REQUIRE_MENTION": "true",
        "GATEWAY_ALLOWED_USERS": "100000000000000009",
        "GATEWAY_ALLOW_ALL_USERS": "false",
        "DISCORD_MISSED_MESSAGE_BACKFILL": "true",
        "DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS": "300000000000000031",
        "DISCORD_MISSED_MESSAGE_BACKFILL_WINDOW_SECONDS": "120",
        "DISCORD_MISSED_MESSAGE_BACKFILL_LIMIT": "7",
        "DISCORD_MISSED_MESSAGE_BACKFILL_MAX_DISPATCHES": "3",
    }
    token = ss.set_secret_scope(scoped_policy)
    try:
        seeded_extra = _apply_yaml_config(
            {},
            {
                "allow_from": ["100000000000000099"],
                "allowed_roles": ["200000000000000099"],
                "allowed_channels": ["300000000000000099"],
                "ignored_channels": ["300000000000000098"],
                "free_response_channels": ["300000000000000097"],
                "allow_all_users": True,
                "allow_bots": "all",
                "bots_require_inline_mention": False,
                "require_mention": True,
                "thread_require_mention": False,
            },
        )
    finally:
        ss.reset_secret_scope(token)

    ss.set_multiplex_active(True)
    adapter = DiscordAdapter(
        PlatformConfig(enabled=True, token="test-token", extra=seeded_extra or {})
    )

    assert adapter._access_policy.allowed_user_ids == {
        scoped_policy["DISCORD_ALLOWED_USERS"]
    }
    assert adapter._access_policy.allowed_role_ids == {
        int(scoped_policy["DISCORD_ALLOWED_ROLES"])
    }
    assert adapter._access_policy.allowed_channel_keys == {
        scoped_policy["DISCORD_ALLOWED_CHANNELS"]
    }
    assert adapter._access_policy.ignored_channel_keys == {
        scoped_policy["DISCORD_IGNORED_CHANNELS"]
    }
    assert adapter._access_policy.free_response_channel_keys == {
        scoped_policy["DISCORD_FREE_RESPONSE_CHANNELS"]
    }
    assert adapter._access_policy.gateway_allowed_user_ids == {
        scoped_policy["GATEWAY_ALLOWED_USERS"]
    }
    assert adapter._access_policy.allow_all_users is False
    assert adapter._access_policy.gateway_allow_all_users is False
    assert adapter._access_policy.allow_bots == "none"
    assert adapter._access_policy.bots_require_inline_mention is True
    assert adapter._discord_require_mention() is False
    assert adapter._discord_thread_require_mention() is True
    assert adapter._missed_message_backfill_enabled() is True
    assert adapter._missed_message_backfill_channels() == {"300000000000000031"}
    assert adapter._missed_message_backfill_window_seconds() == 120
    assert adapter._missed_message_backfill_limit() == 7
    assert adapter._missed_message_backfill_max_dispatches() == 3


def test_scoped_explicit_empty_policy_shadows_process_globals(monkeypatch):
    """An explicit empty profile value must not inherit a permissive global."""
    for name in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.setenv(name, "*")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    scoped_policy = {
        "DISCORD_ALLOWED_USERS": "",
        "DISCORD_ALLOWED_ROLES": "",
        "DISCORD_ALLOWED_CHANNELS": "",
        "DISCORD_IGNORED_CHANNELS": "",
        "DISCORD_FREE_RESPONSE_CHANNELS": "",
        "DISCORD_ALLOW_ALL_USERS": "false",
        "GATEWAY_ALLOWED_USERS": "",
        "GATEWAY_ALLOW_ALL_USERS": "false",
    }
    token = ss.set_secret_scope(scoped_policy)
    try:
        seeded_extra = _apply_yaml_config(
            {},
            {
                "allow_from": ["100000000000000099"],
                "allowed_channels": ["300000000000000099"],
            },
        )
    finally:
        ss.reset_secret_scope(token)

    ss.set_multiplex_active(True)
    adapter = DiscordAdapter(
        PlatformConfig(enabled=True, token="test-token", extra=seeded_extra or {})
    )

    assert adapter._access_policy.allowed_user_ids == set()
    assert adapter._access_policy.allowed_role_ids == set()
    assert adapter._access_policy.allowed_channel_keys == set()
    assert adapter._access_policy.ignored_channel_keys == set()
    assert adapter._access_policy.free_response_channel_keys == set()
    assert adapter._access_policy.gateway_allowed_user_ids == set()
    assert adapter._access_policy.allow_all_users is False
    assert adapter._access_policy.gateway_allow_all_users is False


def test_bot_identity_survives_session_source_round_trip():
    """Persistence must not downgrade a Discord bot to a human principal."""
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="300000000000000001",
        user_id="100000000000000001",
        is_bot=True,
        authorization_channel_keys=[
            "300000000000000001",
            "synthetic-room",
            "#synthetic-room",
        ],
    )

    payload = source.to_dict()

    assert payload["is_bot"] is True
    restored = SessionSource.from_dict(payload)
    assert restored.is_bot is True
    assert set(restored.authorization_channel_keys) == {
        "300000000000000001",
        "synthetic-room",
        "#synthetic-room",
    }
    payload.pop("is_bot")
    assert SessionSource.from_dict(payload).is_bot is False


def test_restored_role_only_source_revalidates_against_owning_adapter():
    """Restoration must check current membership instead of persisting a grant."""
    from gateway.run import GatewayRunner

    ss.set_multiplex_active(True)
    adapter = _adapter_in_scope({"DISCORD_ALLOWED_ROLES": "200000000000000001"})
    member = SimpleNamespace(
        roles=[SimpleNamespace(id=200000000000000001)]
    )
    guild = SimpleNamespace(
        get_member=lambda user_id: (
            member if user_id == 100000000000000001 else None
        )
    )
    adapter._client = SimpleNamespace(
        get_guild=lambda guild_id: (
            guild if guild_id == 300000000000000001 else None
        )
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {"profile-one": {Platform.DISCORD: adapter}}
    runner._active_profile_name = lambda: "default"
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_stores = {}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="300000000000000001",
        chat_type="group",
        user_id="100000000000000001",
        scope_id="300000000000000001",
        role_authorized=True,
        transport_profile="profile-one",
    )
    payload = source.to_dict()
    assert "role_authorized" not in payload
    restored = SessionSource.from_dict(payload)
    assert restored.role_authorized is False
    assert runner._is_user_authorized(restored) is True

    member.roles = []
    assert runner._is_user_authorized(restored) is False


def test_restored_secondary_dm_role_uses_owning_profile_guild():
    """Restored DM role checks must not read the active profile's guild config."""
    from gateway.run import GatewayRunner

    ss.set_multiplex_active(True)
    role_id = 200000000000000001
    owner_guild_id = 300000000000000001
    active_guild_id = 300000000000000099
    adapter = _adapter_in_scope(
        {"DISCORD_ALLOWED_ROLES": str(role_id)},
        extra={"dm_role_auth_guild": owner_guild_id},
    )
    owner_member = SimpleNamespace(roles=[SimpleNamespace(id=role_id)])
    owner_guild = SimpleNamespace(
        get_member=lambda user_id: (
            owner_member if user_id == 100000000000000001 else None
        )
    )
    active_guild = SimpleNamespace(get_member=lambda _user_id: None)
    adapter._client = SimpleNamespace(
        get_guild=lambda guild_id: {
            owner_guild_id: owner_guild,
            active_guild_id: active_guild,
        }.get(guild_id)
    )

    hermes_home = Path(os.environ["HERMES_HOME"])
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        f"discord:\n  dm_role_auth_guild: {active_guild_id}\n"
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {"profile-one": {Platform.DISCORD: adapter}}
    runner._active_profile_name = lambda: "default"
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_stores = {}
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="100000000000000001",
        chat_type="dm",
        user_id="100000000000000001",
        role_authorized=True,
        transport_profile="profile-one",
    )

    restored = SessionSource.from_dict(source.to_dict())
    assert restored.role_authorized is False
    assert runner._is_user_authorized(restored) is True


def test_single_profile_dm_role_guild_refreshes_live_config(monkeypatch):
    """Legacy DM role checks must observe guild changes without a restart."""
    role_id = 200000000000000001
    old_guild_id = 300000000000000001
    new_guild_id = 300000000000000002
    old_user_id = 100000000000000001
    new_user_id = 100000000000000002
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", str(role_id))
    hermes_home = Path(os.environ["HERMES_HOME"])
    hermes_home.mkdir(parents=True, exist_ok=True)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        f"discord:\n  dm_role_auth_guild: {old_guild_id}\n",
        encoding="utf-8",
    )
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"dm_role_auth_guild": old_guild_id},
        )
    )
    role = SimpleNamespace(id=role_id)
    old_guild = SimpleNamespace(
        get_member=lambda user_id: (
            SimpleNamespace(roles=[role]) if user_id == old_user_id else None
        )
    )
    new_guild = SimpleNamespace(
        get_member=lambda user_id: (
            SimpleNamespace(roles=[role]) if user_id == new_user_id else None
        )
    )
    adapter._client = SimpleNamespace(
        get_guild=lambda guild_id: {
            old_guild_id: old_guild,
            new_guild_id: new_guild,
        }.get(guild_id)
    )

    assert adapter._has_allowed_role(str(old_user_id), is_dm=True) is True
    config_path.write_text(
        f"discord:\n  dm_role_auth_guild: {new_guild_id}\n",
        encoding="utf-8",
    )

    assert adapter._has_allowed_role(str(old_user_id), is_dm=True) is False
    assert adapter._has_allowed_role(str(new_user_id), is_dm=True) is True


def test_gateway_auth_uses_the_secondary_discord_adapter_policy(monkeypatch):
    """The downstream gateway gate must not reopen process-global Discord auth."""
    from gateway.run import GatewayRunner

    ss.set_multiplex_active(True)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100000000000000099")
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "100000000000000098")

    adapter = _adapter_in_scope({
        "DISCORD_ALLOWED_USERS": "100000000000000001",
        "GATEWAY_ALLOWED_USERS": "100000000000000009",
    })
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {"profile-one": {Platform.DISCORD: adapter}}
    runner._active_profile_name = lambda: "default"
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_stores = {}

    own_source = SessionSource(
        platform=Platform.DISCORD,
        user_id="100000000000000001",
        chat_id="300000000000000001",
        chat_type="group",
        profile="profile-one",
    )
    foreign_source = SessionSource(
        platform=Platform.DISCORD,
        user_id="100000000000000099",
        chat_id="300000000000000001",
        chat_type="group",
        profile="profile-one",
    )

    assert runner._is_user_authorized(own_source) is True
    assert runner._is_user_authorized(foreign_source) is False


def test_real_profile_loader_and_factory_isolate_two_discord_policies(tmp_path):
    """Real config loading and adapter creation must retain each profile's policy."""
    from gateway.config import load_gateway_config
    from gateway.run import GatewayRunner, _profile_runtime_scope

    ss.set_multiplex_active(True)
    adapters = []
    profile_specs = (
        (
            "profile-one",
            "100000000000000001",
            "300000000000000001",
            400000000000000001,
        ),
        (
            "profile-two",
            "100000000000000002",
            "300000000000000002",
            400000000000000002,
        ),
    )
    for profile_name, user_id, channel_id, dm_guild_id in profile_specs:
        home = tmp_path / profile_name
        home.mkdir(parents=True)
        (home / ".env").write_text(
            f"DISCORD_BOT_TOKEN=synthetic-{profile_name}-token\n",
            encoding="utf-8",
        )
        (home / "config.yaml").write_text(
            "gateway:\n"
            "  multiplex_profiles: true\n"
            "discord:\n"
            "  enabled: true\n"
            f"  allow_from: '{user_id}'\n"
            f"  allowed_channels: '{channel_id}'\n"
            f"  dm_role_auth_guild: {dm_guild_id}\n",
            encoding="utf-8",
        )

        with _profile_runtime_scope(home):
            config = load_gateway_config()
            runner = object.__new__(GatewayRunner)
            runner.config = config
            adapter = runner._create_adapter(
                Platform.DISCORD,
                config.platforms[Platform.DISCORD],
            )
            assert adapter is not None
            assert adapter.platform == Platform.DISCORD
            assert callable(getattr(adapter, "_authorization_policy_allows", None))
            adapters.append(adapter)

    first, second = adapters
    assert first._is_allowed_user("100000000000000001", is_dm=True) is True
    assert first._is_allowed_user("100000000000000002", is_dm=True) is False
    assert second._is_allowed_user("100000000000000002", is_dm=True) is True
    assert second._is_allowed_user("100000000000000001", is_dm=True) is False
    assert first._access_policy.allowed_channel_keys == {"300000000000000001"}
    assert second._access_policy.allowed_channel_keys == {"300000000000000002"}
    assert first._access_policy.dm_role_auth_guild_id == 400000000000000001
    assert second._access_policy.dm_role_auth_guild_id == 400000000000000002
    assert "DISCORD_ALLOWED_USERS" not in os.environ


@pytest.mark.asyncio
async def test_interactive_view_producers_pass_the_owning_policy(monkeypatch):
    """Every Discord send_* producer must construct its view with local policy."""
    import plugins.platforms.discord.adapter as discord_adapter_module

    ss.set_multiplex_active(True)
    adapter = _adapter_in_scope({"DISCORD_ALLOWED_USERS": "100000000000000001"})
    channel = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(id=300000000000000099))
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _channel_id: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )

    captured = {}

    def install_spy(class_name):
        def factory(*args, **kwargs):
            captured[class_name] = {"args": args, **kwargs}
            return SimpleNamespace(_message=None)

        monkeypatch.setattr(discord_adapter_module, class_name, factory)

    for class_name in (
        "ExecApprovalView",
        "SlashConfirmView",
        "ClarifyChoiceView",
        "UpdatePromptView",
        "ModelPickerView",
        "ChoicePickerView",
    ):
        install_spy(class_name)

    assert (
        await adapter.send_exec_approval("300000000000000001", "true", "session-a")
    ).success
    assert (
        await adapter.send_slash_confirm(
            "300000000000000001",
            "Confirm",
            "Continue?",
            "session-a",
            "confirm-a",
        )
    ).success
    assert (
        await adapter.send_clarify(
            "300000000000000001",
            "Choose",
            ["A"],
            "clarify-a",
            "session-a",
        )
    ).success
    assert (
        await adapter.send_update_prompt(
            "300000000000000001", "Update?", session_key="session-a"
        )
    ).success
    assert (
        await adapter.send_model_picker(
            "300000000000000001",
            [{"slug": "synthetic", "name": "Synthetic", "models": []}],
            "synthetic/model",
            "synthetic",
            "session-a",
            AsyncMock(),
        )
    ).success
    assert (
        await adapter.send_choice_picker(
            "300000000000000001",
            "Choose",
            [{"label": "A", "value": "a"}],
            "session-a",
            AsyncMock(),
        )
    ).success

    assert set(captured) == {
        "ExecApprovalView",
        "SlashConfirmView",
        "ClarifyChoiceView",
        "UpdatePromptView",
        "ModelPickerView",
        "ChoicePickerView",
    }
    assert all(
        constructor["access_policy"] is adapter._access_policy
        for constructor in captured.values()
    )


def test_multiplex_missing_mention_settings_use_profile_safe_defaults(monkeypatch):
    """Absent profile mention settings must not inherit process globals."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_THREAD_REQUIRE_MENTION", "true")
    ss.set_multiplex_active(True)

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    assert adapter._discord_require_mention() is True
    assert adapter._discord_thread_require_mention() is False


def test_multiplex_missing_backfill_settings_ignore_process_globals(monkeypatch):
    """Absent recovery config must use local defaults, not another profile's env."""
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS", "300000000000000099")
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_WINDOW_SECONDS", "60")
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_LIMIT", "1")
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_MAX_DISPATCHES", "1")
    ss.set_multiplex_active(True)
    adapter = _adapter_in_scope({
        "DISCORD_ALLOWED_CHANNELS": "300000000000000001",
        "DISCORD_FREE_RESPONSE_CHANNELS": "300000000000000021",
    })

    assert adapter._missed_message_backfill_enabled() is False
    assert adapter._missed_message_backfill_channels() == {
        "300000000000000001",
        "300000000000000021",
    }
    assert adapter._missed_message_backfill_window_seconds() == 21600
    assert adapter._missed_message_backfill_limit() == 100
    assert adapter._missed_message_backfill_max_dispatches() == 10


def test_bot_policy_is_profile_local():
    """Bot admission and inline-mention policy must not leak between adapters."""
    ss.set_multiplex_active(True)
    permissive = _adapter_in_scope({
        "DISCORD_ALLOW_BOTS": "all",
        "DISCORD_BOTS_REQUIRE_INLINE_MENTION": "false",
    })
    restrictive = _adapter_in_scope({
        "DISCORD_ALLOW_BOTS": "none",
        "DISCORD_BOTS_REQUIRE_INLINE_MENTION": "true",
    })
    legacy_on = _adapter_in_scope({
        "DISCORD_BOTS_REQUIRE_INLINE_MENTION": "on",
    })

    assert (
        permissive._authorization_policy_allows("100000000000000001", is_bot=True)
        is True
    )
    assert (
        restrictive._authorization_policy_allows("100000000000000001", is_bot=True)
        is False
    )
    assert permissive._discord_bots_require_inline_mention() is False
    assert restrictive._discord_bots_require_inline_mention() is True
    assert legacy_on._discord_bots_require_inline_mention() is True


def test_gateway_user_grants_pass_discord_ingress():
    """Gateway union grants must not be dropped by Discord's earlier gate."""
    ss.set_multiplex_active(True)
    allowed = _adapter_in_scope({
        "GATEWAY_ALLOWED_USERS": "100000000000000001",
    })
    open_gateway = _adapter_in_scope({
        "GATEWAY_ALLOW_ALL_USERS": "true",
    })

    assert allowed._is_allowed_user("100000000000000001", is_dm=True) is True
    assert allowed._is_allowed_user("100000000000000002", is_dm=True) is False
    assert open_gateway._is_allowed_user("100000000000000002", is_dm=True) is True


def test_pairing_check_is_profile_local_for_messages_and_components():
    """A pairing grant belongs only to the adapter/view's owning profile."""
    ss.set_multiplex_active(True)
    paired = _adapter_in_scope({})
    unpaired = _adapter_in_scope({})
    paired.set_pairing_check(lambda uid: uid == "100000000000000001")
    unpaired.set_pairing_check(lambda _uid: False)

    assert paired._is_allowed_user("100000000000000001") is True
    assert unpaired._is_allowed_user("100000000000000001") is False

    paired_view = ExecApprovalView(
        "session-a", set(), access_policy=paired._access_policy
    )
    unpaired_view = ExecApprovalView(
        "session-b", set(), access_policy=unpaired._access_policy
    )
    interaction = SimpleNamespace(user=SimpleNamespace(id=100000000000000001))

    assert paired_view._check_auth(interaction) is True
    assert unpaired_view._check_auth(interaction) is False


def test_component_channel_hard_denial_precedes_user_grant():
    """An allowed user cannot operate a view in an explicitly ignored channel."""
    ss.set_multiplex_active(True)
    adapter = _adapter_in_scope({
        "DISCORD_ALLOWED_USERS": "100000000000000001",
        "DISCORD_IGNORED_CHANNELS": "300000000000000001",
    })
    view = ExecApprovalView(
        "session-a",
        {"100000000000000001"},
        access_policy=adapter._access_policy,
    )
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=100000000000000001, roles=[]),
        guild_id=300000000000000099,
        channel_id=300000000000000001,
        channel=SimpleNamespace(
            id=300000000000000001,
            name="ignored-room",
            parent=None,
        ),
    )

    assert view._check_auth(interaction) is False


def test_component_denies_when_restricted_channel_cannot_be_resolved():
    """A raw channel ID cannot prove that an ignored parent boundary is absent."""
    ss.set_multiplex_active(True)
    adapter = _adapter_in_scope({
        "DISCORD_ALLOWED_USERS": "100000000000000001",
        "DISCORD_IGNORED_CHANNELS": "300000000000000001",
    })
    view = ExecApprovalView(
        "session-a",
        {"100000000000000001"},
        access_policy=adapter._access_policy,
    )
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=100000000000000001, roles=[]),
        guild_id=300000000000000099,
        channel_id=300000000000000002,
        channel=None,
    )

    assert view._check_auth(interaction) is False


def test_component_allows_channel_only_policy_without_identity_grants():
    """A channel-only policy must keep views usable in an admitted channel."""
    ss.set_multiplex_active(True)
    adapter = _adapter_in_scope({
        "DISCORD_ALLOWED_CHANNELS": "300000000000000001",
    })
    view = ExecApprovalView(
        "session-a",
        set(),
        access_policy=adapter._access_policy,
    )
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=100000000000000001, roles=[]),
        guild_id=300000000000000099,
        channel_id=300000000000000001,
        channel=SimpleNamespace(
            id=300000000000000001,
            name="allowed-room",
            parent=None,
        ),
    )

    assert view._check_auth(interaction) is True


def test_profile_adapter_configuration_binds_only_the_owning_pairing_store():
    """The shared adapter-configuration helper binds each owning pairing store."""
    from gateway.run import GatewayRunner

    ss.set_multiplex_active(True)
    adapter_a = _adapter_in_scope({})
    adapter_b = _adapter_in_scope({})
    store_a = MagicMock()
    store_b = MagicMock()
    store_a.is_approved.side_effect = lambda platform, user: (
        platform == "discord" and user == "100000000000000001"
    )
    store_b.is_approved.side_effect = lambda platform, user: (
        platform == "discord" and user == "100000000000000002"
    )
    runner = object.__new__(GatewayRunner)
    runner.pairing_stores = {"profile-a": store_a, "profile-b": store_b}
    runner.session_store = MagicMock()
    runner._busy_text_mode = "queue"
    runner._make_profile_message_handler = MagicMock(return_value=AsyncMock())
    runner._make_profile_fatal_error_handler = MagicMock(return_value=AsyncMock())
    runner._handle_active_session_busy_message = AsyncMock()
    runner._handle_reaction_event = AsyncMock()
    runner._recover_telegram_topic_thread_id = AsyncMock()
    runner._make_adapter_auth_check = MagicMock(return_value=MagicMock())

    runner._configure_profile_adapter(adapter_a, "profile-a", Platform.DISCORD)
    runner._configure_profile_adapter(adapter_b, "profile-b", Platform.DISCORD)

    assert adapter_a._is_allowed_user("100000000000000001") is True
    assert adapter_a._is_allowed_user("100000000000000002") is False
    assert adapter_b._is_allowed_user("100000000000000002") is True
    assert adapter_b._is_allowed_user("100000000000000001") is False


@pytest.mark.asyncio
async def test_secondary_voice_callback_is_bound_to_owning_adapter():
    """Secondary voice input must not look up or execute through the primary bot."""
    from gateway.run import GatewayRunner

    adapter_a = _adapter_in_scope({"DISCORD_ALLOW_ALL_USERS": "true"})
    adapter_b = _adapter_in_scope({"DISCORD_ALLOW_ALL_USERS": "true"})
    runner = object.__new__(GatewayRunner)
    runner.pairing_stores = {"profile-a": MagicMock(), "profile-b": MagicMock()}
    runner.session_store = MagicMock()
    runner._busy_text_mode = "queue"
    runner._make_profile_message_handler = MagicMock(return_value=AsyncMock())
    runner._make_profile_fatal_error_handler = MagicMock(return_value=AsyncMock())
    runner._handle_active_session_busy_message = AsyncMock()
    runner._handle_reaction_event = AsyncMock()
    runner._recover_telegram_topic_thread_id = AsyncMock()
    runner._make_adapter_auth_check = MagicMock(return_value=MagicMock())
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._is_duplicate_voice_transcript = MagicMock(return_value=False)

    for adapter, profile in ((adapter_a, "profile-a"), (adapter_b, "profile-b")):
        adapter._voice_text_channels = {400000000000000001: 300000000000000001}
        adapter._voice_sources = {}
        adapter._client = SimpleNamespace(get_channel=lambda _channel_id: None)
        adapter._resolve_channel_prompt = MagicMock(return_value=None)
        adapter.handle_message = AsyncMock()
        runner._configure_profile_adapter(adapter, profile, Platform.DISCORD)

    assert callable(adapter_a._voice_input_callback)
    assert callable(adapter_b._voice_input_callback)
    assert adapter_a._voice_input_callback is not adapter_b._voice_input_callback

    await adapter_a._voice_input_callback(
        400000000000000001, 100000000000000001, "synthetic transcript"
    )

    adapter_a.handle_message.assert_awaited_once()
    adapter_b.handle_message.assert_not_awaited()
    event = adapter_a.handle_message.await_args.args[0]
    assert event.source.profile == "profile-a"
    assert event.source.transport_profile == "profile-a"


def test_single_profile_refresh_replaces_initialized_user_and_role_aliases(
    monkeypatch,
):
    """Live legacy allowlist changes must replace, not restore, stale aliases."""
    old_user = "100000000000000001"
    new_user = "100000000000000002"
    old_role = "200000000000000001"
    new_role = "200000000000000002"
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", old_user)
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", old_role)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", new_user)
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", new_role)
    policy = adapter._discord_access_policy()

    assert policy.allowed_user_ids == {new_user}
    assert policy.allowed_role_ids == {int(new_role)}


def test_single_profile_backfill_yaml_precedes_legacy_environment(monkeypatch):
    """Explicit YAML backfill settings remain authoritative over legacy env."""
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS", "env-channel")
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_LIMIT", "400")
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "missed_message_backfill": {
                    "enabled": False,
                    "channels": ["yaml-channel"],
                    "limit": 7,
                }
            },
        )
    )

    assert adapter._missed_message_backfill_enabled() is False
    assert adapter._missed_message_backfill_channels() == {"yaml-channel"}
    assert adapter._missed_message_backfill_limit() == 7


def test_explicit_empty_backfill_channels_do_not_expand_to_policy_channels():
    """An explicit empty recovery channel list must disable channel scanning."""
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "allowed_channels": ["allowed-channel"],
                "free_response_channels": ["free-channel"],
                "missed_message_backfill": {"channels": []},
            },
        )
    )

    assert adapter._missed_message_backfill_channels() == set()


def test_members_intent_refreshes_late_legacy_roles_without_multiplex_leak(
    monkeypatch,
):
    """Legacy connect-time config stays dynamic; multiplex snapshots stay fixed."""
    role_id = "200000000000000001"

    ss.set_multiplex_active(False)
    legacy_adapter = _adapter_in_scope({})
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", role_id)
    assert legacy_adapter._discord_members_intent_required() is True

    monkeypatch.delenv("DISCORD_ALLOWED_ROLES")
    ss.set_multiplex_active(True)
    multiplex_adapter = _adapter_in_scope({})
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", role_id)
    assert multiplex_adapter._discord_members_intent_required() is False


def test_message_admission_marks_only_an_actual_role_match_authorized():
    """A user-ID grant must not be mislabeled as a role grant downstream."""
    ss.set_multiplex_active(True)
    adapter = _adapter_in_scope({
        "DISCORD_ALLOWED_USERS": "100000000000000001",
        "DISCORD_ALLOWED_ROLES": "200000000000000001",
    })
    guild = SimpleNamespace(id=400000000000000001, get_member=lambda _uid: None)
    author = SimpleNamespace(
        id=100000000000000001,
        bot=False,
        guild=guild,
        roles=[SimpleNamespace(id=200000000000000099)],
    )
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=900000000000000001))
    message = SimpleNamespace(
        id=500000000000000001,
        type=discord.MessageType.default,
        author=author,
        guild=guild,
        channel=SimpleNamespace(id=300000000000000001),
        mentions=[],
        content="hello",
    )

    admitted, role_authorized = adapter._discord_message_admission(message, claim=False)

    assert admitted is True
    assert role_authorized is False


def test_slash_channel_policy_is_profile_local():
    """Slash authorization must use the adapter's own allow/ignore channels."""
    ss.set_multiplex_active(True)
    user_id = "100000000000000001"
    allowed_a = "300000000000000001"
    allowed_b = "300000000000000002"
    ignored_a = "300000000000000011"
    ignored_b = "300000000000000012"
    adapter_a = _adapter_in_scope({
        "DISCORD_ALLOWED_USERS": user_id,
        "DISCORD_ALLOWED_CHANNELS": f"{allowed_a},{ignored_a}",
        "DISCORD_IGNORED_CHANNELS": ignored_a,
    })
    adapter_b = _adapter_in_scope({
        "DISCORD_ALLOWED_USERS": user_id,
        "DISCORD_ALLOWED_CHANNELS": f"{allowed_b},{ignored_b}",
        "DISCORD_IGNORED_CHANNELS": ignored_b,
    })

    assert adapter_a._evaluate_slash_authorization(
        _slash_interaction(user_id, allowed_a)
    ) == (True, None)
    assert (
        adapter_a._evaluate_slash_authorization(_slash_interaction(user_id, allowed_b))[
            0
        ]
        is False
    )
    assert (
        adapter_a._evaluate_slash_authorization(_slash_interaction(user_id, ignored_a))[
            0
        ]
        is False
    )

    assert adapter_b._evaluate_slash_authorization(
        _slash_interaction(user_id, allowed_b)
    ) == (True, None)
    assert (
        adapter_b._evaluate_slash_authorization(_slash_interaction(user_id, allowed_a))[
            0
        ]
        is False
    )
    assert (
        adapter_b._evaluate_slash_authorization(_slash_interaction(user_id, ignored_b))[
            0
        ]
        is False
    )


@pytest.mark.asyncio
async def test_message_channel_policy_is_profile_local(monkeypatch):
    """Normal messages must use the adapter's captured channel policy."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    ss.set_multiplex_active(True)
    user_id = "100000000000000001"
    allowed_a = "300000000000000001"
    allowed_b = "300000000000000002"
    ignored_a = "300000000000000011"
    ignored_b = "300000000000000012"
    adapter_a = _adapter_in_scope({
        "DISCORD_ALLOWED_USERS": user_id,
        "DISCORD_ALLOWED_CHANNELS": f"{allowed_a},{ignored_a}",
        "DISCORD_IGNORED_CHANNELS": ignored_a,
        "DISCORD_REQUIRE_MENTION": "false",
    })
    adapter_b = _adapter_in_scope({
        "DISCORD_ALLOWED_USERS": user_id,
        "DISCORD_ALLOWED_CHANNELS": f"{allowed_b},{ignored_b}",
        "DISCORD_IGNORED_CHANNELS": ignored_b,
        "DISCORD_REQUIRE_MENTION": "false",
    })

    for adapter, own, foreign, ignored in (
        (adapter_a, allowed_a, allowed_b, ignored_a),
        (adapter_b, allowed_b, allowed_a, ignored_b),
    ):
        handler = _prepare_message_adapter(adapter)
        await adapter._handle_message(_message(user_id, own))
        handler.assert_awaited_once()

        handler.reset_mock()
        await adapter._handle_message(_message(user_id, foreign))
        handler.assert_not_awaited()

        await adapter._handle_message(_message(user_id, ignored))
        handler.assert_not_awaited()


def test_component_policy_is_profile_local():
    """Button authorization must not read another profile's global policy."""
    ss.set_multiplex_active(True)
    user_a = "100000000000000001"
    user_b = "100000000000000002"
    adapter_a = _adapter_in_scope({"GATEWAY_ALLOWED_USERS": user_a})
    adapter_b = _adapter_in_scope({"GATEWAY_ALLOWED_USERS": user_b})
    adapter_open = _adapter_in_scope({"DISCORD_ALLOW_ALL_USERS": "true"})
    adapter_closed = _adapter_in_scope({})

    def _view(adapter: DiscordAdapter) -> ExecApprovalView:
        return ExecApprovalView(
            session_key="synthetic-session",
            allowed_user_ids=adapter._allowed_user_ids,
            allowed_role_ids=adapter._allowed_role_ids,
            access_policy=adapter._access_policy,
        )

    interaction_a = SimpleNamespace(user=SimpleNamespace(id=int(user_a), roles=[]))
    interaction_b = SimpleNamespace(user=SimpleNamespace(id=int(user_b), roles=[]))
    unknown = SimpleNamespace(user=SimpleNamespace(id=100000000000000099, roles=[]))

    assert _view(adapter_a)._check_auth(interaction_a) is True
    assert _view(adapter_a)._check_auth(interaction_b) is False
    assert _view(adapter_b)._check_auth(interaction_b) is True
    assert _view(adapter_b)._check_auth(interaction_a) is False
    assert _view(adapter_open)._check_auth(unknown) is True
    assert _view(adapter_closed)._check_auth(unknown) is False


def test_stamped_secondary_profile_does_not_use_default_pairing_store():
    """A missing owner store must deny rather than borrow the default grant."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner.pairing_stores = {}
    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="100000000000000001",
        chat_id="300000000000000001",
        chat_type="dm",
        profile="profile-one",
    )

    assert runner._pairing_store_for(source) is None
    assert runner._is_user_authorized(source) is False
    runner.pairing_store.is_approved.assert_not_called()


@pytest.mark.asyncio
async def test_username_resolution_does_not_rewrite_process_globals(monkeypatch):
    """Resolving one profile's usernames must not mutate another profile's source."""
    ss.set_multiplex_active(True)
    adapter_a = _adapter_in_scope({"DISCORD_ALLOWED_USERS": "synthetic-a"})
    adapter_b = _adapter_in_scope({"DISCORD_ALLOWED_USERS": "synthetic-b"})
    member = SimpleNamespace(
        id=100000000000000001,
        name="synthetic-a",
        display_name="Synthetic A",
        global_name=None,
        discriminator="0",
    )
    adapter_a._client = SimpleNamespace(
        guilds=[
            SimpleNamespace(
                name="Synthetic Guild",
                members=[member],
                member_count=1,
            )
        ]
    )
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "process-global-sentinel")

    await adapter_a._resolve_allowed_usernames()

    assert adapter_a._allowed_user_ids == {str(member.id)}
    assert adapter_a._access_policy.allowed_user_ids == {str(member.id)}
    assert adapter_b._allowed_user_ids == {"synthetic-b"}
    assert os.environ["DISCORD_ALLOWED_USERS"] == "process-global-sentinel"


def test_restored_routed_source_keeps_transport_profile_authorization():
    """Persistence must not switch a routed session to its runtime bot policy."""
    from gateway.run import GatewayRunner

    ss.set_multiplex_active(True)
    transport_adapter = _adapter_in_scope({})
    runtime_adapter = _adapter_in_scope({"DISCORD_ALLOWED_USERS": "100000000000000001"})
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {Platform.DISCORD: runtime_adapter}
    runner._profile_adapters = {
        "transport-profile": {Platform.DISCORD: transport_adapter}
    }
    runner._active_profile_name = lambda: "runtime-profile"
    runner.pairing_stores = {}

    source = SessionSource(
        platform=Platform.DISCORD,
        user_id="100000000000000001",
        chat_id="300000000000000001",
        profile="runtime-profile",
    )
    source.transport_profile = "transport-profile"
    source._transport_adapter_ref = weakref.ref(transport_adapter)

    assert runner._is_user_authorized(source) is False
    restored = SessionSource.from_dict(source.to_dict())
    assert restored.profile == "runtime-profile"
    assert restored.transport_profile == "transport-profile"
    assert runner._adapter_for_source(restored) is transport_adapter
    assert runner._is_user_authorized(restored) is False


def test_secondary_default_profile_resolves_its_own_adapter():
    """An explicit secondary 'default' profile must not alias the active slot."""
    from gateway.run import GatewayRunner

    active_adapter = _adapter_in_scope({"DISCORD_ALLOWED_USERS": "100000000000000001"})
    default_adapter = _adapter_in_scope({"DISCORD_ALLOWED_USERS": "100000000000000002"})
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {Platform.DISCORD: active_adapter}
    runner._profile_adapters = {"default": {Platform.DISCORD: default_adapter}}
    runner._active_profile_name = lambda: "active-profile"
    runner.pairing_stores = {}

    assert runner._authorization_adapter(Platform.DISCORD, "default") is default_adapter
    assert (
        runner._is_user_authorized(
            SessionSource(
                platform=Platform.DISCORD,
                user_id="100000000000000001",
                chat_id="300000000000000001",
                profile="default",
            )
        )
        is False
    )


def _runner_with_discord_adapter(adapter: DiscordAdapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    runner.pairing_stores = {}
    return runner


def test_outer_authorization_preserves_channel_context_and_hard_denials():
    """Parent/name grants pass, while allowed/ignored misses remain authoritative."""
    ss.set_multiplex_active(True)
    user_id = "100000000000000001"
    thread_id = "300000000000000002"
    parent_id = "300000000000000001"

    parent_adapter = _adapter_in_scope({"DISCORD_ALLOWED_CHANNELS": parent_id})
    parent_source = SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        chat_id=thread_id,
        parent_chat_id=parent_id,
        authorization_channel_keys=[thread_id, parent_id],
        chat_type="thread",
        profile="default",
    )
    assert (
        _runner_with_discord_adapter(parent_adapter)._is_user_authorized(parent_source)
        is True
    )

    name_adapter = _adapter_in_scope({"DISCORD_ALLOWED_CHANNELS": "synthetic-room"})
    name_source = SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        chat_id=thread_id,
        authorization_channel_keys=[
            thread_id,
            "synthetic-room",
            "#synthetic-room",
        ],
        chat_type="channel",
        profile="default",
    )
    assert (
        _runner_with_discord_adapter(name_adapter)._is_user_authorized(name_source)
        is True
    )

    restricted_adapter = _adapter_in_scope({
        "DISCORD_ALLOWED_USERS": user_id,
        "DISCORD_ALLOWED_CHANNELS": parent_id,
    })
    foreign_source = SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        chat_id="300000000000000099",
        authorization_channel_keys=["300000000000000099"],
        chat_type="channel",
        profile="default",
    )
    assert (
        _runner_with_discord_adapter(restricted_adapter)._is_user_authorized(
            foreign_source
        )
        is False
    )

    ignored_adapter = _adapter_in_scope({"DISCORD_IGNORED_CHANNELS": parent_id})
    ignored_adapter.set_pairing_check(lambda candidate: candidate == user_id)
    ignored_source = SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        chat_id=thread_id,
        parent_chat_id=parent_id,
        authorization_channel_keys=[thread_id, parent_id],
        chat_type="thread",
        profile="default",
    )
    ignored_runner = _runner_with_discord_adapter(ignored_adapter)
    paired_store = MagicMock()
    paired_store.is_approved.return_value = True
    ignored_runner.pairing_stores = {"default": paired_store}
    assert ignored_runner._is_user_authorized(ignored_source) is False


def test_role_only_slash_event_preserves_verified_role_grant():
    """The outer gate must receive the role decision made at slash ingress."""
    ss.set_multiplex_active(True)
    role_id = 200000000000000001
    adapter = _adapter_in_scope({"DISCORD_ALLOWED_ROLES": str(role_id)})
    guild = SimpleNamespace(
        id=400000000000000001,
        name="Synthetic Guild",
        get_member=lambda _uid: None,
    )
    channel = SimpleNamespace(
        id=300000000000000001,
        name="synthetic-room",
        guild=guild,
        topic=None,
    )
    user = SimpleNamespace(
        id=100000000000000001,
        display_name="Synthetic User",
        roles=[SimpleNamespace(id=role_id)],
        guild=guild,
    )
    interaction = SimpleNamespace(
        user=user,
        guild=guild,
        guild_id=guild.id,
        channel=channel,
        channel_id=channel.id,
    )

    assert adapter._evaluate_slash_authorization(interaction) == (True, None)
    event = adapter._build_slash_event(interaction, "/status")
    assert event.source.role_authorized is True
    assert (
        _runner_with_discord_adapter(adapter)._is_user_authorized(event.source) is True
    )
