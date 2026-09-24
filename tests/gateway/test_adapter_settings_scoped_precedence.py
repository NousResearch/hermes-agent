"""Per-profile adapter settings resolve explicit scoped env → the profile's own YAML → default.

Invariants from the #108440 post-merge review: a multiplexed secondary constructed under the real
``_profile_runtime_scope`` reads its own YAML lists/flags (Matrix authz lists, Telegram proxy,
Discord mentions), never inherits the launch process's env on a scoped miss, and the central
``allow_bots`` gate agrees with the adapter; an explicit env value still beats YAML for the owning
profile (single-profile contract). Real loader, real constructors; only transport is substituted.
"""

from __future__ import annotations

import asyncio
import contextlib
import types
from unittest.mock import AsyncMock, patch

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import Platform, load_gateway_config
from hermes_cli.plugins import discover_plugins


@pytest.fixture(autouse=True)
def _plugins():
    discover_plugins()


class _Mentions:
    """``discord.AllowedMentions`` stand-in exposing the four flags (the suite stubs ``discord``)."""

    def __init__(self, *, everyone, roles, users, replied_user):
        self.everyone, self.roles, self.users, self.replied_user = everyone, roles, users, replied_user


def _allowed_mentions(extra):
    from plugins.platforms.discord import adapter as discord_adapter
    with patch.object(discord_adapter, "DISCORD_AVAILABLE", True), \
            patch.object(discord_adapter, "discord", types.SimpleNamespace(AllowedMentions=_Mentions), create=True):
        return discord_adapter._build_allowed_mentions(extra)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """(launch_home, secondary_home); HERMES_HOME points at the launch home, multiplex off on exit."""
    launch = tmp_path / "launch"
    secondary = tmp_path / "launch" / "profiles" / "b2"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    yield launch, secondary
    set_multiplex_active(False)


@contextlib.contextmanager
def _secondary_scope(home):
    from gateway.run import _profile_runtime_scope
    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(home, prepared_secret_scope={}):
            yield
    finally:
        set_multiplex_active(False)


def test_secondary_reads_own_yaml_and_never_the_launch_env(homes, monkeypatch):
    launch, secondary = homes
    (launch / "config.yaml").write_text(
        "matrix:\n  process_notices: true\n  session_scope: room\n"
        "discord:\n  reactions: false\n  allow_mentions:\n    everyone: true\n"
        "slack:\n  reactions: false\n  ignored_channels: [C_LAUNCH]\n"
        "telegram:\n  reactions: true\n", encoding="utf-8")
    load_gateway_config()  # launch profile bridges its YAML into os.environ (single-profile contract)
    (secondary / "config.yaml").write_text(
        "matrix:\n  enabled: true\n  user_id: '@bot:example.org'\n"
        "  allowed_users: ['@owner:example.org']\n  ignore_user_patterns: ['^@ignored:']\n"
        "discord:\n  enabled: true\nslack:\n  enabled: true\n"
        "telegram:\n  enabled: true\n  proxy_url: http://127.0.0.1:18080\n", encoding="utf-8")
    from plugins.platforms.discord.adapter import DiscordAdapter
    from plugins.platforms.matrix.adapter import MatrixAdapter
    from plugins.platforms.slack.adapter import SlackAdapter
    from plugins.platforms.telegram import adapter as tg
    with _secondary_scope(secondary):
        cfg = load_gateway_config()
        m = MatrixAdapter(cfg.platforms[Platform.MATRIX])
        d = DiscordAdapter(cfg.platforms[Platform.DISCORD])
        s = SlackAdapter(cfg.platforms[Platform.SLACK])
        t = tg.TelegramAdapter(cfg.platforms[Platform.TELEGRAM])
        # Omitted keys resolve to the DEFAULT, not the launch profile's bridged env.
        assert (m._process_notices, m._matrix_session_scope) == (False, "auto")
        assert d._reactions_enabled() and s._reactions_enabled() and s._slack_ignored_channels() == set()
        assert not t._reactions_enabled()
        assert _allowed_mentions(d.config.extra).everyone is False
        # The seeded YAML lists reach the Matrix authz consumers.
        assert m._allowed_user_ids == {"@owner:example.org"} and len(m._ignored_user_patterns) == 1
        # The secondary's YAML proxy reaches request construction without an env bridge.
        monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")
        built: list = []
        with patch.object(tg, "HTTPXRequest", lambda **kw: built.append(kw) or types.SimpleNamespace()), \
                patch.object(t, "_instrument_polling_request", side_effect=lambda r: r):
            asyncio.run(t._build_ptb_requests())
        assert [str(kw["httpx_kwargs"]["transport"].proxy.url) for kw in built] == ["http://127.0.0.1:18080"] * 2


def test_explicit_env_beats_yaml_for_the_owning_profile(homes, monkeypatch):
    """Single-profile / owning-profile contract: DISCORD_ALLOW_MENTION_EVERYONE=false beats
    ``everyone: true`` and TELEGRAM_REACTIONS=true beats the stock ``reactions: false`` (#109032)."""
    launch, _ = homes
    (launch / "config.yaml").write_text(
        "discord:\n  allow_mentions:\n    everyone: true\ntelegram:\n  reactions: false\n", encoding="utf-8")
    monkeypatch.setenv("DISCORD_ALLOW_MENTION_EVERYONE", "false")
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    from plugins.platforms.telegram.adapter import TelegramAdapter
    cfg = load_gateway_config()
    assert _allowed_mentions(cfg.platforms[Platform.DISCORD].extra).everyone is False
    assert TelegramAdapter(cfg.platforms[Platform.TELEGRAM])._reactions_enabled() is True


def test_central_allow_bots_gate_honours_a_secondary_yaml_policy(homes):
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from plugins.platforms.discord.adapter import DiscordAdapter
    from plugins.platforms.slack.adapter import SlackAdapter
    _, secondary = homes
    (secondary / "config.yaml").write_text(
        "discord:\n  enabled: true\n  allow_bots: all\nslack:\n  enabled: true\n  allow_bots: all\n", encoding="utf-8")
    with _secondary_scope(secondary):
        cfg = load_gateway_config()
        d, s = DiscordAdapter(cfg.platforms[Platform.DISCORD]), SlackAdapter(cfg.platforms[Platform.SLACK])
        runner = object.__new__(GatewayRunner)
        runner.config, runner._primary_profile_name, runner.adapters = cfg, "default", {}
        runner._profile_adapters = {"b2": {Platform.DISCORD: d, Platform.SLACK: s}}
        for platform, user_id in ((Platform.DISCORD, "123"), (Platform.SLACK, None)):
            src = SessionSource(platform=platform, chat_id="C_TEST", chat_type="group", user_id=user_id,
                                is_bot=True, profile="b2")
            assert runner._is_user_authorized(src), platform


def test_matrix_yaml_lists_gate_intake_and_approval(homes):
    from plugins.platforms.matrix.adapter import MatrixAdapter
    _, secondary = homes
    (secondary / "config.yaml").write_text(
        "matrix:\n  enabled: true\n  user_id: '@bot:example.org'\n"
        "  allowed_users: ['@owner:example.org']\n  ignore_user_patterns: ['^@ignored:']\n", encoding="utf-8")
    with _secondary_scope(secondary):
        a = MatrixAdapter(load_gateway_config().platforms[Platform.MATRIX])
        a._user_id = "@bot:example.org"
        a._is_allowed_matrix_room_event = AsyncMock(return_value=True)
        a._handle_text_message = AsyncMock()
        a.send = AsyncMock()
        event = types.SimpleNamespace(room_id="!r:example.org", sender="@ignored:example.org", event_id="$1",
                                      content={"msgtype": "m.text", "body": "hello"})
        asyncio.run(a._on_room_message(event))
        prompt = types.SimpleNamespace(requester_user_id="@owner:example.org")
        owner_ok = asyncio.run(a._validate_matrix_prompt_reactor("!r:example.org", "$a", "@owner:example.org", prompt, "approval"))
        other_ok = asyncio.run(a._validate_matrix_prompt_reactor("!r:example.org", "$a", "@other:example.org", prompt, "approval"))
    assert a._handle_text_message.await_count == 0 and owner_ok and not other_ok


def test_yuanbao_secondary_home_channel_is_live_and_reloadable(homes):
    """Auto-sethome from a secondary lands in ``platforms.yuanbao.home_channel`` of ITS config (read back
    by ``load_gateway_config``) and on the live PlatformConfig; the process env stays untouched."""
    import os
    from gateway.platforms.yuanbao import AutoSetHomeMiddleware
    _, secondary = homes
    (secondary / "config.yaml").write_text(
        "platforms:\n  yuanbao:\n    enabled: true\n    extra:\n      app_id: a\n      app_secret: b\n", encoding="utf-8")
    adapter = types.SimpleNamespace(name="yuanbao-b2")
    ctx = types.SimpleNamespace(chat_id="dm:tenant-b2", chat_name="b2")
    with _secondary_scope(secondary):
        adapter.config = load_gateway_config().platforms[Platform.YUANBAO]
        AutoSetHomeMiddleware._persist_home(adapter, ctx)
        reloaded = load_gateway_config().get_home_channel(Platform.YUANBAO)
    assert adapter.config.home_channel.chat_id == "dm:tenant-b2"
    assert reloaded is not None and reloaded.chat_id == "dm:tenant-b2"
    assert "YUANBAO_HOME_CHANNEL" not in os.environ


def test_whatsapp_bridge_env_carries_the_secondary_effective_policy(homes, monkeypatch):
    """bridge.js gates DMs and group intake before Python: it must receive the adapter's resolved
    dm_policy/allow_from/group_policy, not the launch process's WHATSAPP_* values."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    _, secondary = homes
    monkeypatch.setenv("WHATSAPP_DM_POLICY", "allowlist")
    monkeypatch.setenv("WHATSAPP_GROUP_POLICY", "disabled")
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15550001111")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "120363000000000000@g.us")
    (secondary / "config.yaml").write_text(
        "whatsapp:\n  enabled: true\n  dm_policy: pairing\n  group_policy: allowlist\n"
        "  group_allow_from: [120363001234567890@g.us]\n", encoding="utf-8")
    with _secondary_scope(secondary):
        a = WhatsAppAdapter(load_gateway_config().platforms[Platform.WHATSAPP])
        env = a._bridge_env()
    assert a._dm_policy == "pairing" == env["WHATSAPP_DM_POLICY"]
    assert a._group_policy == "allowlist" == env["WHATSAPP_GROUP_POLICY"]
    assert env["WHATSAPP_GROUP_ALLOWED_USERS"] == "120363001234567890@g.us"
    assert "WHATSAPP_ALLOWED_USERS" not in env


@pytest.mark.parametrize("secondary_prefix", [None, "Secondary Bot: ", ""])
def test_whatsapp_reply_prefix_isolated_across_profile_scopes(
    homes, monkeypatch, secondary_prefix
):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    launch, secondary = homes
    monkeypatch.setenv("WHATSAPP_REPLY_PREFIX", "Launch Bot: ")
    (launch / "config.yaml").write_text(
        'whatsapp:\n  enabled: true\n  reply_prefix: "Launch YAML: "\n'
    )
    launch_before = WhatsAppAdapter(
        load_gateway_config().platforms[Platform.WHATSAPP]
    )
    assert launch_before._bridge_env()["WHATSAPP_REPLY_PREFIX"] == "Launch Bot: "

    secondary_yaml = "whatsapp:\n  enabled: true\n"
    if secondary_prefix is not None:
        secondary_yaml += f'  reply_prefix: "{secondary_prefix}"\n'
    (secondary / "config.yaml").write_text(secondary_yaml)
    with _secondary_scope(secondary):
        secondary_adapter = WhatsAppAdapter(
            load_gateway_config().platforms[Platform.WHATSAPP]
        )
        assert (
            secondary_adapter._bridge_env().get("WHATSAPP_REPLY_PREFIX")
            == secondary_prefix
        )

    launch_after = WhatsAppAdapter(
        load_gateway_config().platforms[Platform.WHATSAPP]
    )
    assert launch_after._bridge_env()["WHATSAPP_REPLY_PREFIX"] == "Launch Bot: "
