"""The Feishu ``LEFT_CORE`` row: homes core ran Feishu for get the ``feishu`` catalog plugin, others never.

Real config/.env files under a temp home; the catalog lookup and the install are stand-ins (the live install is
exercised end to end outside the unit suite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    for name in ("FEISHU_APP_ID", "FEISHU_APP_SECRET"):
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


_CREDS = "FEISHU_APP_ID=cli_a\nFEISHU_APP_SECRET=s\n"


@pytest.mark.parametrize(("env", "config", "in_use"), [
    # Core's EnvCred row enabled the adapter on the env pair.
    (_CREDS, "", True),
    ("FEISHU_APP_ID=cli_a\n", "", False),
    ("FEISHU_APP_SECRET=s\n", "", False),
    # Core's is_connected: a platforms.feishu block whose extra carries app_id.
    ("", "platforms:\n  feishu:\n    extra: {app_id: cli_a}\n", True),
    ("", "platforms:\n  feishu: {app_id: cli_a}\n", True),
    ("", "platforms:\n  feishu:\n    enabled: true\n", True),
    ("", "feishu:\n  app_id: cli_a\n", True),
    # An explicit disable kept core's adapter off whatever credentials were set.
    (_CREDS, "platforms:\n  feishu:\n    enabled: false\n", False),
    (_CREDS, "feishu:\n  enabled: false\n", False),
    # The feishu_doc / feishu_drive tools selected outside Feishu's own row.
    ("", "platform_toolsets:\n  telegram: [hermes-telegram, feishu_doc]\n", True),
    ("", "toolsets: [hermes-feishu]\n", True),
    ("", "platform_toolsets:\n  feishu: [hermes-feishu]\n", False),
    ("", "", False),
])
def test_feishu_in_use_matches_what_core_ran(tmp_path, env, config, in_use):
    assert lcm.feishu_in_use(_home(tmp_path, env=env, config=config)) is in_use


@pytest.mark.parametrize("name", ["platforms/feishu", "feishu-platform"])
def test_plugins_disabled_naming_the_bundled_adapter_is_not_use(tmp_path, name):
    home = _home(tmp_path, env=_CREDS, config=f"plugins:\n  disabled: [{name}]\n")
    assert lcm.feishu_in_use(home) is False
    assert lcm.legacy_plugin_alias(name) == "feishu"


def test_a_feishu_home_gets_the_plugin_once_with_no_placement_notice(tmp_path):
    home = _home(tmp_path, env=_CREDS, config="agent:\n  disabled_toolsets: [hermes-feishu]\n")
    calls, said = [], []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == ["feishu"]
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == ["feishu"]
    # An official plugin: the outcome line carries no "not supported" placement sentence.
    assert len(said) == 1 and "not supported" not in said[0] and "Labs" not in said[0]
    assert lcm._feishu_notice(home) == ""
    from hermes_cli.config import read_user_config_raw
    config = read_user_config_raw(home / "config.yaml")
    # The feishu_* tools lived only in hermes-feishu: every other platform keeps them off.
    assert config["known_plugin_toolsets"]["telegram"] == ["feishu_doc", "feishu_drive"]
    assert "feishu" not in config["known_plugin_toolsets"]
    # A disabled hermes-feishu bundle removed the feishu_* tools; the plugin's toolsets stay disabled.
    assert config["agent"]["disabled_toolsets"] == ["hermes-feishu", "feishu_doc", "feishu_drive"]


def test_a_platform_keeps_exactly_the_feishu_toolsets_it_selected(tmp_path):
    """Resolved per toolset: a saved ``feishu_doc`` alone kept the comment tools off; ``hermes-feishu``
    (or a composite) carried both; a platform with no saved list had neither."""
    from hermes_cli.tools_config import _enabled_plugin_toolsets
    selections = {"telegram": ["hermes-telegram", "feishu_doc"], "slack": ["web", "hermes-feishu"],
                  "discord": ["web"], "feishu": ["web"]}  # core re-added feishu_* to Feishu's own saved list
    rows = "".join(f"  {p}: [{', '.join(sel)}]\n" for p, sel in selections.items())
    home = _home(tmp_path, env=_CREDS, config="platform_toolsets:\n" + rows)
    lcm.migrate_home(home, install=lambda n: {"ok": True}, say=lambda m: None)
    config = lcm._read_config(home)
    plugin = {"feishu_doc", "feishu_drive"}
    enabled = {p: _enabled_plugin_toolsets(config, p, sel, plugin) for p, sel in {**selections, "cli": ["hermes-cli"]}.items()}
    assert enabled == {"telegram": {"feishu_doc"}, "slack": plugin, "discord": set(), "feishu": plugin, "cli": set()}


def test_a_string_saved_selection_counts_as_use(tmp_path):
    """Older ``hermes config set`` saved a list as a string; the runtime parses it, so the migration does."""
    home = _home(tmp_path, config="platform_toolsets:\n  telegram: '[\"hermes-telegram\", \"hermes-feishu\"]'\n")
    assert lcm.feishu_in_use(home) is True


def test_a_home_with_both_catalog_platforms_keeps_each_ones_tools_off_on_the_other(tmp_path):
    from hermes_cli.tools_config import _enabled_plugin_toolsets
    home = _home(tmp_path, env=_CREDS + "YUANBAO_APP_ID=a\nYUANBAO_APP_SECRET=s\n")
    lcm.migrate_home(home, install=lambda n: {"ok": True}, say=lambda m: None)
    config = lcm._read_config(home)
    assert not _enabled_plugin_toolsets(config, "feishu", ["hermes-feishu"], {"yuanbao"})
    assert not _enabled_plugin_toolsets(config, "yuanbao", ["hermes-yuanbao"], {"feishu_doc", "feishu_drive"})


def test_the_notice_names_host_isolation(tmp_path):
    home = _home(tmp_path, env=_CREDS, config="plugins:\n  isolation: host\n")
    assert "plugins.isolation: host" in lcm._feishu_notice(home)
    (tmp_path / "b").mkdir()
    assert "isolation" not in lcm._feishu_notice(_home(tmp_path / "b", env=_CREDS))


def test_child_processes_keep_the_feishu_strip_set():
    """Core's blocklist carried every FEISHU_* name it declared; it still does with the adapter gone."""
    from tools.environments.local_env_policy import _ADAPTER_SECRET_ENV, _HERMES_PROVIDER_ENV_BLOCKLIST
    assert {n for n in _HERMES_PROVIDER_ENV_BLOCKLIST if n.startswith("FEISHU_")} == {
        "FEISHU_ALLOWED_USERS", "FEISHU_ALLOW_ALL_USERS", "FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_DOMAIN",
        "FEISHU_ENCRYPT_KEY", "FEISHU_HOME_CHANNEL", "FEISHU_HOME_CHANNEL_NAME", "FEISHU_VERIFICATION_TOKEN"}
    assert {"FEISHU_ENCRYPT_KEY", "FEISHU_VERIFICATION_TOKEN"} <= _ADAPTER_SECRET_ENV
