"""The DingTalk ``LEFT_CORE`` row: homes core ran DingTalk for get the ``dingtalk`` Labs plugin, others never.

Real config/.env files under a temp home; the catalog lookup and the install are stand-ins (the live install is
exercised end to end outside the unit suite).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm

_CREDS = "DINGTALK_CLIENT_ID=cid\nDINGTALK_CLIENT_SECRET=sec\n"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    for name in ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)


def _home(tmp_path: Path, *, env: str = "", config: str = "") -> Path:
    home = tmp_path / "home"
    home.mkdir()
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


def _dingtalk():
    return next(f for f in lcm.LEFT_CORE if f.plugin == "dingtalk")


# Core's gate: the env credential pair (EnvCred) or ``_is_connected`` (client_id + client_secret from the
# platform block's extra first, then env), unless the merged block says ``enabled: false``.
@pytest.mark.parametrize(("env", "config", "in_use"), [
    (_CREDS, "", True),
    ("", "platforms:\n  dingtalk:\n    enabled: true\n", True),
    ("", "platforms:\n  dingtalk:\n    extra: {client_id: c, client_secret: s}\n", True),
    ("DINGTALK_CLIENT_SECRET=s\n", "platforms:\n  dingtalk:\n    extra: {client_id: c}\n", True),
    ("", "gateway:\n  platforms:\n    dingtalk:\n      enabled: true\n", True),
    ("", "platform_toolsets:\n  telegram: [hermes-dingtalk]\n", True),  # resolves only once the platform registers
    ("", "", False),
    ("DINGTALK_CLIENT_ID=cid\n", "", False),  # half a credential pair never enabled the adapter
    ("DINGTALK_WEBHOOK_URL=https://oapi.dingtalk.com/robot/send?access_token=x\n", "", False),  # cron-only sender
    ("", "platforms:\n  dingtalk:\n    extra: {client_id: c}\n", False),
    (_CREDS, "platforms:\n  dingtalk:\n    enabled: false\n", False),
    (_CREDS, "dingtalk:\n  enabled: false\n", False),
    ("", "platform_toolsets:\n  dingtalk: [hermes-dingtalk]\n", False),  # its own row scoped its own sessions
    # Older ``hermes config set`` builds saved a bare list argument as a string; the runtime still parses it.
    ("", "platform_toolsets:\n  telegram: '[\"hermes-dingtalk\"]'\n", True),
    ("", "toolsets: '[\"hermes-dingtalk\"]'\n", False),  # the runtime never parsed a string top-level list
])
def test_dingtalk_in_use_matches_what_core_ran(tmp_path, env, config, in_use):
    assert lcm.dingtalk_in_use(_home(tmp_path, env=env, config=config)) is in_use


def test_a_disable_in_legacy_gateway_json_vetoes_env_credentials(tmp_path):
    home = _home(tmp_path, env=_CREDS)
    (home / "gateway.json").write_text(json.dumps({"platforms": {"dingtalk": {"enabled": False}}}))
    assert lcm.dingtalk_in_use(home) is False


@pytest.mark.parametrize("name", ["platforms/dingtalk", "dingtalk-platform"])
def test_plugins_disabled_naming_the_bundled_adapter_is_not_use(tmp_path, name):
    home = _home(tmp_path, env=_CREDS, config=f"plugins:\n  disabled: [{name}]\n")
    assert lcm.dingtalk_in_use(home) is False
    assert lcm.legacy_plugin_alias(name) == "dingtalk"


def test_a_dingtalk_home_gets_the_plugin_once_with_the_labs_notice(tmp_path):
    home = _home(tmp_path, env=_CREDS)
    calls, said = [], []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == ["dingtalk"]
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == ["dingtalk"]
    assert "DingTalk is now a Hermes Labs plugin: built by Nous Research, not supported" in said[0]


def test_a_saved_home_channel_does_not_stop_the_migration(tmp_path):
    home = _home(tmp_path, env=_CREDS, config="platforms:\n  dingtalk:\n    enabled: true\n"
                 "    home_channel: {platform: dingtalk, chat_id: cid1, name: Home}\n")
    installed = lcm.migrate_home(home, install=lambda name: (home / "plugins" / name).mkdir(parents=True)
                                 or {"ok": True}, say=lambda message: None)
    assert installed == ["dingtalk"]


def test_a_secondary_whose_credentials_come_from_its_secret_scope_uses_dingtalk(tmp_path):
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    home = _home(tmp_path)
    creds = {"DINGTALK_CLIENT_ID": "cid", "DINGTALK_CLIENT_SECRET": "sec"}
    for bound, in_use in ((home, True), (tmp_path / "other", False)):
        token = set_secret_scope(creds, profile_home=str(bound))
        try:
            assert lcm.dingtalk_in_use(home) is in_use
        finally:
            reset_secret_scope(token)


def test_child_processes_keep_the_dingtalk_strip_set_core_had():
    """While core shipped DingTalk its manifest kept these out of every child; the row keeps them out with the
    plugin absent (same set as main: CLIENT_SECRET a secret, the rest provider env)."""
    from tools.environments import local_env_policy as policy
    assert {n for n in policy._HERMES_PROVIDER_ENV_BLOCKLIST if n.startswith("DINGTALK_")} == {
        "DINGTALK_ALLOWED_USERS", "DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET", "DINGTALK_HOME_CHANNEL",
        "DINGTALK_HOME_CHANNEL_NAME", "DINGTALK_WEBHOOK_URL"}
    assert "DINGTALK_CLIENT_SECRET" in _dingtalk().secret_env
    # An adapter secret (a restart's launch-profile strip) and the allowlists as profile gates, as on main.
    assert "DINGTALK_CLIENT_SECRET" in policy._ADAPTER_SECRET_ENV
    assert policy.is_profile_gate_env("DINGTALK_ALLOWED_USERS")
    assert policy.is_profile_gate_env("DINGTALK_ALLOW_ALL_USERS")


def test_yuanbao_tools_stay_off_in_dingtalk_sessions():
    """A home upgrading across both removals: core never carried the yb_* tools in hermes-dingtalk."""
    yuanbao = next(f for f in lcm.LEFT_CORE if f.plugin == "yuanbao")
    assert "dingtalk" in yuanbao.off_platforms


def test_a_restart_keeps_non_platform_left_core_secrets():
    """Only platform rows' secrets are adapter secrets (a supervised restart strips those); a service unit's
    CAMOFOX_API_KEY / HERMES_LANGFUSE_SECRET_KEY still reaches the restarted gateway."""
    from tools.environments import local_env_policy as policy
    assert "DINGTALK_CLIENT_SECRET" in policy._ADAPTER_SECRET_ENV
    assert not {"CAMOFOX_API_KEY", "HERMES_LANGFUSE_SECRET_KEY"} & policy._ADAPTER_SECRET_ENV


def test_the_plugin_platform_keeps_its_cron_home_target_slot(monkeypatch):
    """``deliver=origin`` with no origin picks the first platform with a home chat: DingTalk ranked ahead of
    Feishu in core, and the plugin platform keeps that slot."""
    from types import SimpleNamespace

    import cron.scheduler_delivery as delivery
    from gateway.platform_registry import platform_registry
    entries = [SimpleNamespace(name=n, cron_deliver_env_var=f"{n.upper()}_HOME_CHANNEL") for n in ("feishu", "dingtalk")]
    monkeypatch.setattr(platform_registry, "plugin_entries", lambda: entries)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)
    assert list(delivery._iter_home_target_platforms()) == ["dingtalk", "feishu"]
