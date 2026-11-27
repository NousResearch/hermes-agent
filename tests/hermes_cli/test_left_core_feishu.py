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


def test_a_feishu_disable_survives_the_automatic_install(tmp_path, monkeypatch):
    """Greptile #159651: a credentialed home core ran Feishu for despite a bare ``feishu`` in plugins.disabled
    gets the plugin through the real install and enable path, yet the entry stays and Feishu stays off; the
    outcome line names the enable that turns it on."""
    from types import SimpleNamespace
    import hermes_cli.plugins_cmd_install as pci
    monkeypatch.setattr(pci, "_resolve_source", lambda identifier, catalog_name: (
        SimpleNamespace(name=catalog_name, known_issues=()), catalog_name, [], None))

    def place(entry, identifier, **_kw):  # the clone, network-free
        target = home / "plugins" / entry.name
        target.mkdir(parents=True)
        (target / "plugin.yaml").write_text(f"name: {entry.name}\nversion: 1.0.0\nkind: platform\n", encoding="utf-8")
        return target, {"name": entry.name}, entry.name
    monkeypatch.setattr(pci, "_place_tree", place)
    monkeypatch.setattr("hermes_cli.plugins_admission.admit_plugin_set_change",
                        lambda *a, **k: pytest.fail("the automatic install enabled a plugin the home disabled"))
    home = _home(tmp_path, env=_CREDS, config="plugins:\n  disabled: [feishu]\n")
    said = []

    assert lcm.migrate_home(home, install=lcm._install_into(home, consent=True), say=said.append) == ["feishu"]
    from hermes_cli.config import read_user_config_raw
    plugins = read_user_config_raw(home / "config.yaml")["plugins"]
    assert plugins["disabled"] == ["feishu"] and "feishu" not in (plugins.get("enabled") or [])
    assert "so it is installed but off: run `hermes plugins enable feishu` to turn it on" in said[0]


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


def test_explicit_legacy_feishu_bundle_selects_the_installed_plugin_tools(monkeypatch, tmp_path):
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
    from model_tools import _apply_toolset_selection
    from tools import registry as registry_module
    from toolsets import validate_toolset

    registry = registry_module.ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", registry)
    plugin_home = tmp_path / "feishu-profile"
    for tool, toolset in (("feishu_doc_read", "feishu_doc"), ("feishu_drive_add_comment", "feishu_drive")):
        registry.register(tool, toolset, {"name": tool}, lambda: None, scope=hermes_home_key(plugin_home))

    token = set_hermes_home_override(plugin_home)
    try:
        selected = set()
        _apply_toolset_selection(selected, ["hermes-feishu"], quiet_mode=True, disable=False)
        assert {"feishu_doc_read", "feishu_drive_add_comment", "terminal"} <= selected
        _apply_toolset_selection(selected, ["hermes-feishu"], quiet_mode=True, disable=True)
        assert "terminal" in selected
        assert not {"feishu_doc_read", "feishu_drive_add_comment"} & selected
    finally:
        reset_hermes_home_override(token)

    assert not validate_toolset("hermes-feishu")


@pytest.mark.parametrize("present", [True, False], ids=["kept", "removed"])
def test_a_feishu_home_the_default_installer_reached_first_keeps_its_scope_and_is_not_reinstalled(tmp_path, present):
    """feishu is a ``default: true`` plugin, so ``hermes update``'s default installer can place it before the
    first start's migration runs. The migration must not install it again (a removal sticks), but must still
    keep the feishu_* tools off where core had them off. A home that removed that idle copy while core still
    ran Feishu is told Feishu is off and how to get it back."""
    home = _home(tmp_path, env=_CREDS, config="_default_plugins_installed: [feishu]\n")
    if present:
        (home / "plugins" / "feishu").mkdir(parents=True)
    calls, said = [], []
    assert lcm.migrate_home(home, install=lambda name: calls.append(name) or {"ok": True}, say=said.append) == []
    assert calls == []
    off = [line for line in said if "so it is off" in line]
    assert said == ([] if present else off) and len(off) == (0 if present else 1), said
    assert present or "`hermes plugins install feishu`" in off[0]
    config = lcm._read_config(home)
    assert config["known_plugin_toolsets"]["telegram"] == ["feishu_doc", "feishu_drive"]


def test_the_default_installer_skips_feishu_the_left_core_migration_placed(tmp_path):
    from hermes_cli import default_plugins
    home = _home(tmp_path, config="_left_core_installed: [feishu]\n")
    assert "feishu" in default_plugins._recorded(home)
    assert default_plugins._recorded(home, own_only=True) == set()


@pytest.mark.parametrize("config", ["", *(f"plugins:\n  disabled: [{name}]\n"
                                        for name in ("feishu", "platforms/feishu", "feishu-platform"))],
                         ids=["feishu-pending", "disabled-feishu", "disabled-platforms/feishu", "disabled-feishu-platform"])
def test_feishu_credentials_never_name_the_hermes_tag_override(tmp_path, monkeypatch, config):
    """The real catalog's ``hermes-tag`` requires ``FEISHU_APP_ID``/``FEISHU_APP_SECRET``. Those credentials are
    Feishu's: a home whose feishu plugin is not there yet is not told 'hermes-tag' is missing (the left-core
    install serves them and reports its own outcome, as core's bundled Feishu served them), and neither is one
    that turned Feishu off under any of its names (hermes-tag extends the Feishu adapter, so it is no substitute)."""
    import hermes_cli.plugin_catalog as catalog
    from hermes_cli import catalog_platforms
    home = _home(tmp_path, env=_CREDS, config=config)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(catalog, "fetch_live_catalog", lambda **kwargs: None)
    assert "hermes-tag" in {entry.name for entry in catalog.load_catalog()}
    assert "hermes-tag" not in {entry.name for entry in catalog_platforms.pending_entries(home)}


@pytest.mark.parametrize("placed", ["left-core", "default"])
def test_a_hermes_tag_home_is_told_once_the_official_feishu_plugin_replaces_its_override(tmp_path, placed):
    """``hermes-tag`` overrides Feishu by importing core's bundled adapter. Its home hears where Feishu went
    exactly once, also when the default installer placed the feishu plugin first (no install outcome line)."""
    config = "_default_plugins_installed: [feishu]\n" if placed == "default" else ""
    home = _home(tmp_path, env=_CREDS, config=config)
    (home / "plugins" / "hermes-tag").mkdir(parents=True)
    if placed == "default":
        (home / "plugins" / "feishu").mkdir()
    said = []

    def install(name):
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    lcm.migrate_home(home, install=install, say=said.append)
    lcm.migrate_home(home, install=install, say=said.append)
    told = [line for line in said if "'hermes-tag'" in line]
    assert len(told) == 1 and "The replacement is the official 'feishu' plugin" in told[0]
    assert "runs on" not in told[0]  # said before the install's outcome is known: no claim that Feishu runs


def test_a_feishu_home_without_hermes_tag_hears_nothing_about_it(tmp_path):
    home = _home(tmp_path, env=_CREDS)
    said = []
    lcm.migrate_home(home, install=lambda name: (home / "plugins" / name).mkdir(parents=True) or {"ok": True},
                     say=said.append)
    assert not [line for line in said if "hermes-tag" in line]


@pytest.mark.parametrize("recorded", ["_default_plugins_installed: [feishu]\n", ""], ids=["default-placed", "user-placed"])
def test_a_present_feishu_the_home_disables_is_said_off_once_and_stays_disabled(tmp_path, recorded):
    # Core ran Feishu despite a bare ``feishu`` in plugins.disabled; the plugin already in the home honours it, so
    # the upgrade turns Feishu off. The disable stays (nothing installs or enables) and the home is told, once.
    home = _home(tmp_path, env=_CREDS, config=recorded + "plugins:\n  disabled: [feishu]\n")
    (home / "plugins" / "feishu").mkdir(parents=True)
    calls, said = [], []

    def install(name):
        calls.append(name)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == []
    from hermes_cli.config import read_user_config_raw
    assert read_user_config_raw(home / "config.yaml")["plugins"]["disabled"] == ["feishu"]
    off = [line for line in said if "plugins.disabled lists 'feishu'" in line]
    assert len(off) == 1 and "so it is installed but off: run `hermes plugins enable feishu`" in off[0], said


@pytest.mark.parametrize("recorded", ["_default_plugins_installed: [feishu]\n", ""], ids=["default-placed", "user-placed"])
def test_a_present_feishu_under_host_isolation_is_told_once(tmp_path, recorded):
    """Whoever placed the feishu plugin, a profile with ``plugins.isolation: host`` cannot load its adapter
    (core's bundled one was exempt): the upgrade says so once, and installs nothing."""
    home = _home(tmp_path, env=_CREDS, config=recorded + "plugins:\n  isolation: host\n")
    (home / "plugins" / "feishu").mkdir(parents=True)
    calls, said = [], []

    def install(name):
        calls.append(name)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == []
    told = [line for line in said if "plugins.isolation: host" in line]
    assert len(told) == 1 and "Feishu" in told[0], said


def test_host_isolation_pinned_in_the_managed_layer_is_named(tmp_path, monkeypatch):
    """Greptile #159651: an administrator's `plugins.isolation: host` in the managed config.yaml keeps the
    plugin loader from starting the Feishu adapter; the notice reads the same layers, not the profile file alone."""
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("plugins:\n  isolation: host\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    home = _home(tmp_path, env=_CREDS)
    (home / "plugins" / "feishu").mkdir(parents=True)
    assert "plugins.isolation: host" in next(f for f in lcm.LEFT_CORE if f.plugin == "feishu").notice(home)
