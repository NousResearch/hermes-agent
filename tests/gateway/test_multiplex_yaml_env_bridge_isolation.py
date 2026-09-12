"""A multiplexed secondary profile's config.yaml must never be bridged into the process env.

``_load_secondary_profile_config`` runs ``load_gateway_config()`` inside ``_profile_runtime_scope``;
every ``apply_yaml_config_fn`` hook and ``bridge_core_env_settings`` used to write ``os.environ`` there
(first-writer-wins), so the first secondary's mention/allowlist policy became the DEFAULT profile's
(#80099, #72348). The values must instead reach that profile's ``PlatformConfig.extra``.
"""
from __future__ import annotations

import os

import pytest

from agent.secret_scope import (
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_SECONDARY_YAML = """\
require_mention: false
telegram:
  mention_patterns: ['^bot2']
  reactions: false
matrix:
  require_mention: false
  allowed_users: ['@b2:example.org']
  session_scope: room
whatsapp:
  dm_policy: open
  allow_from: ['+15550002']
feishu:
  allow_bots: all
slack:
  allow_bots: all
  ignored_channels: [C_B2]
dingtalk:
  allowed_users: [b2user]
discord:
  auto_thread: false
  reactions: false
signal:
  require_mention: false
"""

_BRIDGED_ENV = (
    "TELEGRAM_REQUIRE_MENTION", "TELEGRAM_MENTION_PATTERNS", "TELEGRAM_REACTIONS",
    "MATRIX_REQUIRE_MENTION", "MATRIX_ALLOWED_USERS", "MATRIX_SESSION_SCOPE",
    "WHATSAPP_DM_POLICY", "WHATSAPP_ALLOWED_USERS", "FEISHU_ALLOW_BOTS",
    "SLACK_ALLOW_BOTS", "SLACK_IGNORED_CHANNELS", "DINGTALK_ALLOWED_USERS",
    "DISCORD_AUTO_THREAD", "DISCORD_REACTIONS", "SIGNAL_REQUIRE_MENTION",
)

_EXPECTED_EXTRA = (
    ("telegram", "mention_patterns"), ("telegram", "reactions"), ("matrix", "session_scope"),
    ("matrix", "allowed_users"), ("whatsapp", "dm_policy"), ("feishu", "allow_bots"),
    ("slack", "allow_bots"), ("slack", "ignored_channels"), ("dingtalk", "allowed_users"),
    ("discord", "auto_thread"), ("signal", "require_mention"),
)


@pytest.fixture
def secondary_scope(tmp_path, monkeypatch):
    default_home = tmp_path / "hermes"
    secondary = default_home / "profiles" / "bot2"
    secondary.mkdir(parents=True)
    (default_home / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n")
    (secondary / "config.yaml").write_text(_SECONDARY_YAML)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    for name in _BRIDGED_ENV:
        monkeypatch.delenv(name, raising=False)
    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(secondary))
    secret_token = set_secret_scope({"TELEGRAM_BOT_TOKEN": "222:b2"})
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)


def test_secondary_profile_yaml_reaches_its_extra_not_the_process_env(secondary_scope):
    from hermes_cli.plugins import discover_plugins
    from gateway.config import Platform, load_gateway_config

    discover_plugins()
    cfg = load_gateway_config()

    poisoned = {name: os.environ[name] for name in _BRIDGED_ENV if name in os.environ}
    assert poisoned == {}, f"secondary profile wrote into process env: {poisoned}"
    missing = [
        f"{plat}.{key}" for plat, key in _EXPECTED_EXTRA
        if key not in ((cfg.platforms.get(Platform(plat)) or _Empty()).extra or {})
    ]
    assert missing == [], f"secondary profile's own YAML not seeded into extra: {missing}"


class _Empty:
    extra: dict = {}
