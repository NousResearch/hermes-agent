"""Standalone-vs-served parity for adapter SETTINGS (not credentials) under ``gateway.multiplex_profiles``.

A served secondary profile's adapter is constructed inside ``_profile_runtime_scope`` while ``os.environ``
still holds the DEFAULT profile's ``.env``. Every non-credential setting an adapter reads with a bare
``os.getenv`` (ports, hosts, mention gating, reactions, thread policy, notification targets) therefore
resolves to the default profile's value — the adapter behaves differently served than it does standalone.

The invariant: with the SAME profile ``.env`` the adapter resolves the SAME setting whether the profile is
the ambient process home (standalone) or a scoped secondary (served). Each row names the setting and the
callable that resolves it from a constructed adapter, so a regression is reported by setting name.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from agent import secret_scope as ss
from gateway.config import PlatformConfig


@pytest.fixture(autouse=True)
def _multiplex_off_after():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _served(monkeypatch, default_env: dict, secondary_env: dict, build):
    """``(standalone_value, served_value)`` for one setting: standalone = the secondary's env IS the
    process env; served = default's env in the process, secondary's only in the scope."""
    for k, v in secondary_env.items():
        monkeypatch.setenv(k, v)
    standalone = build()
    for k in secondary_env:
        monkeypatch.delenv(k, raising=False)
    for k, v in default_env.items():
        monkeypatch.setenv(k, v)
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(dict(secondary_env))
    try:
        served = build()
    finally:
        ss.reset_secret_scope(token)
    return standalone, served


# (module, env var, default-profile value, secondary value, resolver(module) -> value)
_SETTINGS = [
    ("plugins.platforms.slack.adapter", "SLACK_REQUIRE_MENTION", "true", "false",
     lambda m: _slack(m, "_slack_require_mention")),
    ("plugins.platforms.slack.adapter", "SLACK_REACTIONS", "true", "false",
     lambda m: _slack(m, "_reactions_enabled")),
    ("plugins.platforms.slack.adapter", "SLACK_REACTION_TRIGGER_TARGET", "C_DEFAULT", "C_SECONDARY",
     lambda m: _slack(m, "_slack_reaction_trigger_target")),
    ("plugins.platforms.matrix.adapter", "MATRIX_REQUIRE_MENTION", "true", "false",
     lambda m: m.MatrixAdapter._parse_require_mention(PlatformConfig(enabled=True))),
    ("plugins.platforms.matrix.adapter", "MATRIX_THREAD_REQUIRE_MENTION", "false", "true",
     lambda m: m.MatrixAdapter._parse_thread_require_mention(PlatformConfig(enabled=True))),
    ("plugins.platforms.matrix.adapter", "MATRIX_MAX_MESSAGE_LENGTH", "1111", "2222",
     lambda m: m._resolve_max_message_length(PlatformConfig(enabled=True))),
    ("plugins.platforms.matrix.adapter", "MATRIX_E2EE_MODE", "required", "off",
     lambda m: m._resolve_e2ee_mode({})),
    ("plugins.platforms.matrix.adapter", "MATRIX_ALLOW_PUBLIC_ROOMS", "true", "false",
     lambda m: m._env_truthy("MATRIX_ALLOW_PUBLIC_ROOMS")),
    ("plugins.platforms.teams.adapter", "TEAMS_PORT", "18001", "18101",
     lambda m: (m._env_enablement() or {}).get("port")),
    ("plugins.platforms.feishu.adapter", "FEISHU_WEBHOOK_PORT", "18073", "18173",
     lambda m: m.FeishuAdapter._load_settings({}).webhook_port),
    ("plugins.platforms.feishu.adapter", "FEISHU_CONNECTION_MODE", "websocket", "webhook",
     lambda m: m.FeishuAdapter._load_settings({}).connection_mode),
    ("gateway.platforms.bluebubbles", "BLUEBUBBLES_WEBHOOK_PORT", "18010", "18110",
     lambda m: m._setting({}, "webhook_port", "BLUEBUBBLES_WEBHOOK_PORT", "0")),
    ("gateway.platforms.signal", "SIGNAL_REACTIONS", "true", "false",
     lambda m: _signal_reactions(m)),
    ("plugins.platforms.line.adapter", "LINE_PORT", "18015", "18115",
     lambda m: (m._env_enablement() or {}).get("port")),
    ("plugins.platforms.buzz.adapter", "BUZZ_RELAY_URL", "wss://default.relay", "wss://secondary.relay",
     lambda m: (m._env_enablement() or {}).get("relay_url")),
    ("plugins.platforms.a2a.adapter", "A2A_AGENT_NAME", "default-agent", "secondary-agent",
     lambda m: m._default_agent_name()),
    ("plugins.platforms.discord.adapter", "DISCORD_COMMAND_SYNC_POLICY", "full", "off",
     lambda m: _discord(m, "_get_discord_command_sync_policy")),
    ("plugins.platforms.discord.adapter", "DISCORD_ALLOW_MENTION_EVERYONE", "true", "false",
     lambda m: m._env_bool("DISCORD_ALLOW_MENTION_EVERYONE", False)),
]


def _slack(mod, method):
    adapter = object.__new__(mod.SlackAdapter)
    adapter.config = PlatformConfig(enabled=True)
    return getattr(adapter, method)()


def _signal_reactions(mod):
    adapter = object.__new__(mod.SignalAdapter)
    adapter.dm_allow_from = {"*"}
    return adapter._reactions_enabled()


def _discord(mod, method):
    adapter = object.__new__(mod.DiscordAdapter)
    adapter.config = PlatformConfig(enabled=True)
    adapter.platform = adapter.config and __import__("gateway.config", fromlist=["Platform"]).Platform.DISCORD
    return getattr(adapter, method)()


@pytest.mark.parametrize(("module", "var", "default_value", "secondary_value", "resolve"), _SETTINGS,
                         ids=[f"{m.rsplit('.', 2)[-2]}:{v}" for m, v, *_ in _SETTINGS])
def test_served_secondary_resolves_its_own_setting(monkeypatch, module, var, default_value, secondary_value, resolve):
    mod = importlib.import_module(module)
    for name in (var, "LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET", "TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET",
                 "TEAMS_TENANT_ID", "SIGNAL_ACCOUNT", "BUZZ_PRIVATE_KEY"):
        monkeypatch.delenv(name, raising=False)
    creds = {"LINE_CHANNEL_ACCESS_TOKEN": "t", "LINE_CHANNEL_SECRET": "s", "TEAMS_CLIENT_ID": "a",
             "TEAMS_CLIENT_SECRET": "b", "TEAMS_TENANT_ID": "c", "SIGNAL_ACCOUNT": "+1", "BUZZ_PRIVATE_KEY": "k"}
    standalone, served = _served(monkeypatch, {var: default_value, **creds}, {var: secondary_value, **creds},
                                 lambda: resolve(mod))
    assert served == standalone, f"{var}: served={served!r} standalone={standalone!r}"


def test_signal_startup_gate_reads_the_profile_env(monkeypatch):
    """A secondary whose SIGNAL_HTTP_URL/SIGNAL_ACCOUNT live only in its own .env must pass the startup
    gate served exactly as it does standalone; a secondary WITHOUT them must not borrow the default's."""
    from gateway.platforms import signal as sig
    for k in ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"):
        monkeypatch.delenv(k, raising=False)
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"SIGNAL_HTTP_URL": "http://secondary.invalid:8080", "SIGNAL_ACCOUNT": "+15550000001"})
    try:
        assert sig.validate_signal_config(PlatformConfig(enabled=True)) is True
    finally:
        ss.reset_secret_scope(token)
    monkeypatch.setenv("SIGNAL_HTTP_URL", "http://default.invalid:8080")
    monkeypatch.setenv("SIGNAL_ACCOUNT", "+15550000000")
    token = ss.set_secret_scope({})
    try:
        assert sig.validate_signal_config(PlatformConfig(enabled=True)) is False
    finally:
        ss.reset_secret_scope(token)


def test_sms_webhook_listener_is_profile_scoped(monkeypatch):
    from plugins.platforms.sms import adapter as sms
    for k, v in {"TWILIO_ACCOUNT_SID": "AC0", "TWILIO_AUTH_TOKEN": "tok", "TWILIO_PHONE_NUMBER": "+1",
                 "SMS_WEBHOOK_PORT": "18039", "SMS_WEBHOOK_HOST": "default-host", "SMS_WEBHOOK_URL": "http://d/"}.items():
        monkeypatch.setenv(k, v)
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"TWILIO_ACCOUNT_SID": "AC1", "TWILIO_AUTH_TOKEN": "tok2", "TWILIO_PHONE_NUMBER": "+2",
                                 "SMS_WEBHOOK_PORT": "18139", "SMS_WEBHOOK_HOST": "secondary-host", "SMS_WEBHOOK_URL": "http://s/"})
    try:
        adapter = sms.SmsAdapter(PlatformConfig(enabled=True))
    finally:
        ss.reset_secret_scope(token)
    assert (adapter._webhook_port, adapter._webhook_host, adapter._webhook_url) == (18139, "secondary-host", "http://s/")
