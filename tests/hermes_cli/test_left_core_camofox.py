"""Camofox left core for the ``camofox`` catalog plugin: a home that ran core's Camofox backend gets the
plugin once, and its core-era ``browser`` toolset (Camofox's browser_* tools then; browser_exec now) becomes
the plugin's ``camofox`` toolset, a one-time conversion a later run never undoes.

Real config/.env files under temp homes, a fixture catalog dir; the network install is a stand-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm
import hermes_yaml as yaml
from hermes_cli.config_defaults import DEFAULT_CONFIG

_CATALOG_ENTRY = """\
name: camofox
repo: https://github.com/NousResearch/hermes-official-plugins
sha: 0123456789abcdef0123456789abcdef01234567
subdir: camofox
description: Camofox anti-detection browser backend (fixture).
maintainer: NousResearch
tier: official
category: tools
capabilities:
  provides_tools: [browser_navigate]
  provides_hooks: []
  provides_middleware: []
  requires_env: []
"""


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(lcm, "_attempted", set())
    monkeypatch.setattr(lcm, "_undelivered", {})
    for name in ("CAMOFOX_URL", "BROWSER_CDP_URL"):
        monkeypatch.delenv(name, raising=False)
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import plugin_catalog
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "camofox.yaml").write_text(_CATALOG_ENTRY, encoding="utf-8")
    monkeypatch.setattr(plugin_catalog, "get_catalog_dir", lambda: catalog)
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **_: None)
    return catalog


def _home(tmp_path: Path, name: str = "home", *, env: str = "", config: dict | None = None) -> Path:
    home = tmp_path / name
    home.mkdir(parents=True)
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    if config is not None:
        (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return home


def _config(home: Path) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


def _installer(home: Path, calls: list):
    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}
    return install


@pytest.mark.parametrize(("env", "config", "in_use"), [
    ("CAMOFOX_URL=http://localhost:9377\n", None, True),                       # A: the address alone selected it
    ("", None, False),                                                          # B
    ("", {"browser": {"cloud_provider": "camofox"}}, True),
    # browser.camofox settings never selected Camofox: without an address core ran browser_exec.
    ("", {"browser": {"camofox": {"managed_persistence": True, "user_id": "me"}}}, False),
    ("CAMOFOX_URL=http://localhost:9377\n", {"browser": {"camofox": {"managed_persistence": True}}}, True),
    # A stored selection decides alone; a CDP endpoint won over Camofox.
    ("CAMOFOX_URL=http://localhost:9377\n", {"browser": {"cloud_provider": "browserbase"}}, False),
    ("CAMOFOX_URL=http://localhost:9377\n", {"browser": {"use_gateway": True}}, False),
    ("CAMOFOX_URL=http://localhost:9377\n", {"browser": {"cdp_url": "ws://127.0.0.1:9222"}}, False),
    ("CAMOFOX_URL=http://localhost:9377\nBROWSER_CDP_URL=ws://127.0.0.1:9222\n", None, False),
])
def test_camofox_in_use_reads_core_selection(tmp_path, env, config, in_use):
    assert lcm.camofox_in_use(_home(tmp_path, env=env, config=config)) is in_use


def test_leftover_camofox_settings_keep_browser_exec(tmp_path):
    """A browser.camofox block with no address is not a Camofox home: nothing is installed and core's
    browser toolset stays on (converting it would leave the home with no browser at all)."""
    home = _home(tmp_path, config={
        "_config_version": 51,
        "browser": {"camofox": {"managed_persistence": True}},
        "platform_toolsets": {"cli": ["terminal", "browser", "file"]},
    })
    before = (home / "config.yaml").read_bytes()
    assert lcm.migrate_home(home, install=lambda name: pytest.fail("not a Camofox home"), say=print) == []
    assert (home / "config.yaml").read_bytes() == before


def test_managed_env_url_counts(tmp_path):
    (tmp_path / "managed" / ".env").write_text("CAMOFOX_URL=http://camofox:9377\n", encoding="utf-8")
    assert lcm.camofox_in_use(_home(tmp_path)) is True


def test_process_env_url_counts_only_for_the_launch_home(tmp_path, monkeypatch):
    launch = _home(tmp_path, "launch")
    secondary = _home(tmp_path, "profiles/work")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
    assert lcm.camofox_in_use(launch) is False
    assert lcm.camofox_in_use(launch, process_env=True) is True
    assert lcm.camofox_in_use(secondary, process_env=True) is False


def _resolved(home: Path, platform: str, monkeypatch) -> set[str]:
    from hermes_cli import tools_config
    monkeypatch.setattr(tools_config, "_get_plugin_toolset_keys", lambda: {"camofox"})
    return tools_config._get_platform_tools(_config(home), platform, include_default_mcp_servers=False)


def test_browser_becomes_camofox_where_core_carried_it(tmp_path, monkeypatch):
    home = _home(tmp_path, env="CAMOFOX_URL=http://localhost:9377\n", config={
        "_config_version": 51,
        "platform_toolsets": {
            "cli": ["terminal", "browser", "file"],   # browser named directly
            "telegram": ["hermes-telegram"],          # through the composite
            "discord": ["terminal", "file"],          # core had browser off here
        },
        "agent": {"disabled_toolsets": ["memory"]},
    })
    calls, said = [], []
    install = _installer(home, calls)
    assert lcm.migrate_home(home, install=install, say=said.append) == ["camofox"]
    assert calls == ["camofox"]
    assert "Camofox browser backend moved out of core" in said[0]

    config = _config(home)
    assert config["platform_toolsets"]["cli"] == ["terminal", "file", "camofox"]
    assert config["agent"]["disabled_toolsets"] == ["memory", "browser"]
    known = config["known_plugin_toolsets"]
    assert "camofox" in known["discord"] and "camofox" in known["webhook"]
    assert "cli" not in known and "telegram" not in known
    # The end state per platform: the camofox surface where core had browser on, never browser_exec.
    for platform in ("cli", "telegram", "slack"):  # slack: no saved list, default composite
        enabled = _resolved(home, platform, monkeypatch)
        assert "camofox" in enabled and "browser" not in enabled, platform
    for platform in ("discord", "webhook"):
        enabled = _resolved(home, platform, monkeypatch)
        assert "camofox" not in enabled and "browser" not in enabled, platform


def _boot(home: Path, monkeypatch, said: list | None = None) -> list[str]:
    """Agent start for *home* (``recover_at_startup``), the plugin install a stand-in."""
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(lcm, "_install_into", lambda target, consent=None: _installer(target, []))
    return lcm.recover_at_startup(say=said.append if said is not None else lambda _m: None)


@pytest.mark.parametrize("saved", [{"cli": ["terminal", "browser", "file"]}, None], ids=["saved-list", "no-saved-list"])
def test_globally_disabled_browsing_stays_off_after_the_conversion(tmp_path, monkeypatch, saved):
    """A Camofox home that had turned browsing off everywhere (``agent.disabled_toolsets: [browser]``): a
    saved list still becomes ``camofox``, but the plugin's toolset is disabled too, so browsing stays off
    on every platform, with or without a saved list."""
    config = {"_config_version": 51, "agent": {"disabled_toolsets": ["browser"]}}
    if saved is not None:
        config["platform_toolsets"] = saved
    home = _home(tmp_path, env="CAMOFOX_URL=http://localhost:9377\n", config=config)
    assert _boot(home, monkeypatch) == ["camofox"]
    config = _config(home)
    if saved is not None:
        assert config["platform_toolsets"]["cli"] == ["terminal", "file", "camofox"]
    assert set(config["agent"]["disabled_toolsets"]) == {"browser", "camofox"}
    for platform in ("cli", "telegram"):
        enabled = _resolved(home, platform, monkeypatch)
        assert "camofox" not in enabled and "browser" not in enabled, platform


def test_conversion_is_never_undone(tmp_path):
    home = _home(tmp_path, env="CAMOFOX_URL=http://localhost:9377\n",
                 config={"_config_version": 51, "platform_toolsets": {"cli": ["terminal", "browser"]}})
    lcm._record_migration(home, next(f for f in lcm.LEFT_CORE if f.plugin == "camofox"))
    config = _config(home)
    assert config["_left_core_scoped"] == ["camofox"]
    # The user turns core's browser (browser_exec) back on for the CLI afterwards.
    config["platform_toolsets"]["cli"] = ["terminal", "camofox", "browser"]
    config["agent"]["disabled_toolsets"] = []
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    before = (home / "config.yaml").read_bytes()
    calls = []
    install = _installer(home, calls)
    lcm.migrate_home(home, install=install, say=lambda _m: None)
    after = _config(home)
    assert after["platform_toolsets"]["cli"] == ["terminal", "camofox", "browser"]
    assert after["agent"]["disabled_toolsets"] == []
    assert calls == ["camofox"]
    lcm.migrate_home(home, install=install, say=lambda _m: None)
    assert calls == ["camofox"]
    assert {k: v for k, v in _config(home).items() if k != "_left_core_installed"} == yaml.safe_load(before)


def test_homes_are_islands_a_b_c_a(tmp_path):
    """A (CAMOFOX_URL in .env) and C (cloud_provider: camofox) are migrated; B is never touched."""
    a = _home(tmp_path, "a", env="CAMOFOX_URL=http://localhost:9377\n",
              config={"_config_version": 51, "platform_toolsets": {"cli": ["browser"]}})
    b = _home(tmp_path, "b", config={"_config_version": 51, "platform_toolsets": {"cli": ["browser", "terminal"]}})
    c = _home(tmp_path, "c", config={"_config_version": 51, "browser": {"cloud_provider": "camofox"}})
    b_before = (b / "config.yaml").read_bytes()
    calls: list[tuple[str, str]] = []

    def run(home):
        def install(name):
            calls.append((home.name, name))
            (home / "plugins" / name).mkdir(parents=True)
            return {"ok": True}
        return lcm.migrate_home(home, install=install, say=lambda _m: None)

    assert run(a) == ["camofox"]
    assert run(b) == []
    assert run(c) == ["camofox"]
    assert run(a) == []
    assert calls == [("a", "camofox"), ("c", "camofox")]
    assert (b / "config.yaml").read_bytes() == b_before
    assert not (b / "plugins").exists()
    assert _config(a)["platform_toolsets"]["cli"] == ["camofox"]
    assert _config(c)["agent"]["disabled_toolsets"] == ["browser"]


def _migrate_config(home: Path) -> None:
    from hermes_cli.config import migrate_config
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        migrate_config(interactive=False, quiet=True)
    finally:
        reset_hermes_home_override(token)


_SPLIT_ENV = "CAMOFOX_URL=http://localhost:9377\n"
_SPLIT_TOOLSETS = {"cli": ["terminal", "browser", "file"]}


def test_upgrade_converts_after_the_config_step(tmp_path, monkeypatch):
    """`hermes update` / Desktop migrate config.yaml to v52 before the left-core pass: the v52 step
    records the core-era Camofox use, so the pass still converts the home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path, env=_SPLIT_ENV, config={"_config_version": 51, "platform_toolsets": _SPLIT_TOOLSETS})
    _migrate_config(home)
    assert _config(home)["_config_version"] == DEFAULT_CONFIG["_config_version"]
    calls = []
    assert lcm.migrate_home(home, install=_installer(home, calls), say=lambda _m: None) == ["camofox"]
    config = _config(home)
    assert config["platform_toolsets"]["cli"] == ["terminal", "file", "camofox"]
    assert config["agent"]["disabled_toolsets"] == ["browser"]


@pytest.mark.parametrize("stamp", [51, 52])
def test_home_from_after_the_split_keeps_browser_exec(tmp_path, monkeypatch, stamp):
    """A home that crossed v52 without Camofox (or started at v52) and sets up the camofox plugin later
    runs both side by side: the row never turns browser_exec off for it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path, config={"_config_version": stamp, "platform_toolsets": _SPLIT_TOOLSETS})
    _migrate_config(home)
    (home / "plugins" / "camofox").mkdir(parents=True)
    (home / ".env").write_text(_SPLIT_ENV, encoding="utf-8")
    before = (home / "config.yaml").read_bytes()
    assert lcm.migrate_home(home, install=lambda name: pytest.fail("installed by the user"), say=print) == []
    assert (home / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("config", [None, {"platform_toolsets": _SPLIT_TOOLSETS}], ids=["no-config", "unstamped"])
def test_an_unversioned_camofox_home_is_converted(tmp_path, monkeypatch, config):
    """Main ran Camofox for a home with no config.yaml (CAMOFOX_URL in .env alone) or an unstamped one, and
    Hermes always stamps a config.yaml it creates: such a home ran the core feature and is converted."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path, env=_SPLIT_ENV, config=config)
    calls = []
    assert lcm.migrate_home(home, install=_installer(home, calls), say=lambda _m: None) == ["camofox"]
    assert _config(home)["agent"]["disabled_toolsets"] == ["browser"]


def test_the_config_yaml_an_env_only_conversion_creates_is_stamped(tmp_path, monkeypatch, caplog):
    """An env-only Camofox home has no config.yaml; the conversion creates one. It carries the current
    stamp (a missing file read as current) and the row in use, so the next boot's config step neither
    warns that the file predates the support floor nor makes the home look newer than the split."""
    import logging

    from hermes_cli import post_update

    home = _home(tmp_path, env=_SPLIT_ENV)
    assert _boot(home, monkeypatch) == ["camofox"]
    with caplog.at_level(logging.WARNING):
        assert post_update.step_migrate_config() == {"ok": True, "skipped": "up-to-date"}
    assert "predates version" not in caplog.text
    config = _config(home)
    assert config["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert config["_left_core_used"] == ["camofox"]

    monkeypatch.setattr(lcm, "_attempted", set())  # the next process
    said = []
    assert _boot(home, monkeypatch, said) == []
    assert said == []


def test_a_plain_save_stamping_an_unversioned_camofox_home_keeps_its_conversion(tmp_path, monkeypatch):
    """Any config write before the first left-core pass (`hermes config set`, a model switch) stamps an
    unstamped config.yaml at the current version without running the v52 step; the home still converts."""
    from hermes_cli.config import load_config, save_config
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path, env=_SPLIT_ENV, config={"platform_toolsets": _SPLIT_TOOLSETS})
    token = set_hermes_home_override(home)
    try:
        save_config(load_config())
    finally:
        reset_hermes_home_override(token)
    assert _config(home)["_config_version"] == DEFAULT_CONFIG["_config_version"]
    calls = []
    assert lcm.migrate_home(home, install=_installer(home, calls), say=lambda _m: None) == ["camofox"]
    assert _config(home)["platform_toolsets"]["cli"] == ["terminal", "file", "camofox"]


def test_a_stamped_camofox_home_without_the_plugin_is_told_where_it_went(tmp_path, monkeypatch):
    """A Nix-managed config.yaml carries the package's stamp (52) from activation, so neither the stamp nor
    the v52 step shows this home ran core's Camofox. It is told where Camofox went (once per backoff window
    at start) and nothing about it changes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path, env=_SPLIT_ENV, config={"_config_version": 52, "platform_toolsets": _SPLIT_TOOLSETS})
    before = (home / "config.yaml").read_bytes()
    said = []
    assert lcm.migrate_home(home, install=lambda name: pytest.fail("nothing is installed"), say=said.append) == []
    assert len(said) == 1
    assert "Camofox browser backend moved out of core" in said[0] and "hermes plugins install camofox" in said[0]
    assert (home / "config.yaml").read_bytes() == before

    assert lcm._pending(home, say=said.append, backoff=True) == []
    assert len(said) == 1, "a start inside the backoff window repeats nothing"


def test_catalog_miss_names_the_install_command(tmp_path, _fresh):
    (_fresh / "camofox.yaml").unlink()
    home = _home(tmp_path, env="CAMOFOX_URL=http://localhost:9377\n", config={"_config_version": 51})
    said = []
    assert lcm.migrate_home(home, install=lambda name: pytest.fail("no catalog entry"), say=said.append) == []
    assert "cannot find in the plugin catalog yet" in said[0]
    assert "hermes plugins install camofox" in said[0]


def test_camofox_env_stays_out_of_children():
    """With the plugin absent no manifest declares CAMOFOX_*: core keeps stripping them."""
    from tools.environments import local_env_policy as policy
    assert "CAMOFOX_URL" in policy._build_provider_env_blocklist()
    assert "CAMOFOX_API_KEY" in policy._ALWAYS_STRIP_KEYS
