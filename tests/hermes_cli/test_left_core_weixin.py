"""The Weixin ``LEFT_CORE`` row: homes core ran Weixin for get the ``weixin`` catalog plugin, others never.

Each in_use case is a home whose verdict on main came from a real probe: ``load_gateway_config()`` plus the
adapter core created and its connect preflight (token and account id, the QR login's saved account file
included). Real config/.env files under a temp home; the catalog lookup and the install are stand-ins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm

_CREDS = "WEIXIN_ACCOUNT_ID=acc1\nWEIXIN_TOKEN=tok1\n"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    for name in ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)


def _home(tmp_path: Path, *, env: str = "", config: str = "", saved_token: str = "") -> Path:
    home = tmp_path / "home"
    home.mkdir()
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    if saved_token:  # what the QR login (save_weixin_account) wrote
        accounts = home / "weixin" / "accounts"
        accounts.mkdir(parents=True)
        (accounts / "acc1.json").write_text(json.dumps({"token": saved_token, "base_url": ""}), encoding="utf-8")
    return home


def _install(home: Path):
    calls: list[str] = []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
        return {"ok": True}
    return install, calls


# The platform_toolsets block cli-config.yaml.example shipped while Weixin was in core.
_SHIPPED_PLATFORM_TOOLSETS = (
    "platform_toolsets:\n  cli: [hermes-cli]\n  telegram: [hermes-telegram]\n  weixin: [hermes-weixin]\n"
)


@pytest.mark.parametrize(("env", "config", "saved_token", "in_use"), [
    (_CREDS, "", "", True),
    ("", "", "", False),
    ("WEIXIN_TOKEN=tok1\n", "", "", False),  # no account: the adapter's preflight refused to connect
    ("WEIXIN_ACCOUNT_ID=acc1\n", "", "", False),  # no token anywhere
    ("WEIXIN_ACCOUNT_ID=acc1\n", "", "tok1", True),  # the QR login's saved token filled it in
    ("", "platforms:\n  weixin:\n    enabled: true\n    extra: {account_id: acc1}\n", "tok1", True),
    ("", "platforms:\n  weixin:\n    extra: {account_id: acc1, token: tok1}\n", "", True),
    # A block token with no enabled key: only extra/env reached the enable pass's probe, so it stayed off.
    ("", "platforms:\n  weixin:\n    token: tok1\n    extra: {account_id: acc1}\n", "", False),
    ("", "platforms:\n  weixin:\n    enabled: true\n    token: tok1\n    extra: {account_id: acc1}\n", "", True),
    ("WEIXIN_TOKEN=tok1\n", "platforms:\n  weixin:\n    extra: {account_id: acc1}\n", "", True),
    ("WEIXIN_ACCOUNT_ID=acc1\n", "platforms:\n  weixin:\n    token: tok1\n", "", True),
    ("", "platforms:\n  weixin:\n    enabled: true\n", "", False),
    # An explicit config.yaml disable vetoed env credentials, legacy and top-level blocks included.
    (_CREDS, "platforms:\n  weixin:\n    enabled: false\n", "", False),
    (_CREDS, "gateway:\n  weixin:\n    enabled: false\n", "", False),
    (_CREDS, "weixin:\n  enabled: false\n", "", False),
    (_CREDS, "platforms:\n  weixin:\n    enabled: false\nweixin:\n  enabled: true\n", "", True),
    # A saved home channel names Platform("weixin"), which exists only once the plugin loads.
    (_CREDS, "platforms:\n  weixin:\n    enabled: true\n    home_channel: {platform: weixin, chat_id: x}\n", "", True),
    ("", _SHIPPED_PLATFORM_TOOLSETS, "", False),
    ("", "platform_toolsets:\n  telegram: [hermes-weixin]\n", "", False),  # core tools only, never the adapter
    (_CREDS, "plugins:\n  disabled: [platforms/weixin]\n", "", False),
    (_CREDS, "plugins:\n  disabled: [weixin-platform]\n", "", False),
    (_CREDS, "plugins:\n  disabled: [weixin]\n", "", True),  # core's adapter had another name
])
def test_weixin_in_use_matches_what_core_ran(tmp_path, env, config, saved_token, in_use):
    assert lcm.weixin_in_use(_home(tmp_path, env=env, config=config, saved_token=saved_token)) is in_use


def test_a_disable_in_legacy_gateway_json_did_not_stop_env_credentials(tmp_path):
    """gateway.json's ``enabled: false`` is overwritten by the env enable (its marker never reached the pass)."""
    home = _home(tmp_path, env=_CREDS)
    (home / "gateway.json").write_text(json.dumps({"platforms": {"weixin": {"enabled": False}}}))
    assert lcm.weixin_in_use(home) is True


def test_a_weixin_home_gets_the_plugin_once(tmp_path):
    home = _home(tmp_path, env=_CREDS)
    install, calls = _install(home)
    said: list[str] = []
    assert lcm.migrate_home(home, install=install, say=said.append) == ["weixin"]
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == ["weixin"]
    assert "plugins.disabled" not in said[0]
    # No plugin toolset: nothing is recorded as off anywhere.
    from hermes_cli.config import read_user_config_raw
    assert "weixin" not in str(read_user_config_raw(home / "config.yaml").get("known_plugin_toolsets") or {})


def test_a_fresh_or_shipped_default_home_installs_nothing(tmp_path):
    for config in ("", _SHIPPED_PLATFORM_TOOLSETS):
        home = tmp_path / ("shipped" if config else "fresh")
        home.mkdir()
        if config:
            (home / "config.yaml").write_text(config, encoding="utf-8")
        said: list[str] = []
        assert lcm.migrate_home(home, install=lambda name: pytest.fail("installed"), say=said.append) == []
        assert said == []


def test_a_secondary_whose_credentials_come_from_its_secret_scope_uses_weixin(tmp_path):
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    home = _home(tmp_path)
    other = tmp_path / "other"
    for bound, in_use in ((home, True), (other, False)):
        token = set_secret_scope({"WEIXIN_ACCOUNT_ID": "acc1", "WEIXIN_TOKEN": "tok1"}, profile_home=str(bound))
        try:
            assert lcm.weixin_in_use(home) is in_use
        finally:
            reset_secret_scope(token)


@pytest.mark.parametrize("name", ["platforms/weixin", "weixin-platform"])
def test_core_names_in_plugins_disabled_still_disable_the_plugin(tmp_path, name):
    from hermes_cli.plugins_discovery import gate_manifest
    from hermes_cli.plugins_manifest import PluginManifest
    home = _home(tmp_path)
    manifest = PluginManifest(name="weixin", source="user", path=str(home / "plugins" / "weixin"), key="weixin")
    assert gate_manifest(manifest, {name}, None).error == "disabled via config"
    assert lcm.legacy_plugin_alias(name) == "weixin"


def test_a_bare_weixin_disable_that_survives_the_install_is_named(tmp_path):
    home = _home(tmp_path, env=_CREDS, config="plugins:\n  disabled: [weixin]\n")
    sentence = "plugins.disabled lists 'weixin', which now turns Weixin off"
    for lifted in (True, False):
        (home / "config.yaml").write_text("plugins:\n  disabled: [weixin]\n", encoding="utf-8")
        said: list[str] = []

        def install(name):
            (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
            if lifted:  # what the real enable does (_apply_activation)
                (home / "config.yaml").write_text("plugins:\n  enabled: [weixin]\n  disabled: []\n", encoding="utf-8")
            return {"ok": True}
        lcm._attempted.clear()
        import shutil
        shutil.rmtree(home / "plugins", ignore_errors=True)
        assert lcm.migrate_home(home, install=install, say=said.append) == ["weixin"]
        assert (sentence in said[0]) is (not lifted)


def test_child_processes_keep_every_weixin_setting_core_stripped():
    from tools.environments.local_env_policy import _HERMES_PROVIDER_ENV_BLOCKLIST
    assert {n for n in _HERMES_PROVIDER_ENV_BLOCKLIST if n.startswith("WEIXIN_")} == {
        "WEIXIN_TOKEN", "WEIXIN_ACCOUNT_ID", "WEIXIN_ALLOWED_USERS", "WEIXIN_ALLOW_ALL_USERS", "WEIXIN_BASE_URL",
        "WEIXIN_CDN_BASE_URL", "WEIXIN_DM_POLICY", "WEIXIN_GROUP_ALLOWED_USERS", "WEIXIN_GROUP_POLICY",
        "WEIXIN_HOME_CHANNEL", "WEIXIN_HOME_CHANNEL_NAME", "WEIXIN_SPLIT_MULTILINE_MESSAGES"}
