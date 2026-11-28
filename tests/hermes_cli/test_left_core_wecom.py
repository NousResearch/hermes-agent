"""The WeCom ``LEFT_CORE`` rows: homes core ran WeCom (bot or callback mode) for get the ``wecom`` Hermes Labs
plugin once, others never; plus the two generic seams the move added (``PlatformEntry.listen_port``,
``StandaloneSendPolicy.media_via_adapter``).

Real config/.env files under a temp home; the catalog lookup and the install are stand-ins (the live install is
exercised end to end outside the unit suite).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.left_core_migration as lcm

_BOT_ENV = "WECOM_BOT_ID=b\nWECOM_SECRET=s\n"
_CALLBACK_ENV = "WECOM_CALLBACK_CORP_ID=c\nWECOM_CALLBACK_CORP_SECRET=s\n"
_CORE_ERA = "_config_version: 54\n"  # a config.yaml from before v55: the proof a home ran core's WeCom


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    for name in ("WECOM_BOT_ID", "WECOM_SECRET", "WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET"):
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


@pytest.mark.parametrize(("env", "config", "in_use"), [
    (_BOT_ENV, "", True),
    ("", "platforms:\n  wecom:\n    extra: {bot_id: b}\n", True),  # core's registry pass: bot_id alone
    ("", "", False),
    ("WECOM_BOT_ID=b\n", "", False),  # the env row needed the secret too
    (_BOT_ENV, "platforms:\n  wecom:\n    enabled: false\n", False),
    (_BOT_ENV, "plugins:\n  disabled: [platforms/wecom]\n", False),
    (_BOT_ENV, "plugins:\n  disabled: [wecom-platform]\n", False),
    # Core's adapter was ``wecom-platform``: a bare ``wecom`` never stopped it.
    (_BOT_ENV, "plugins:\n  disabled: [wecom]\n", True),
])
def test_wecom_bot_in_use_matches_what_core_ran(tmp_path, env, config, in_use):
    assert lcm.wecom_in_use(_home(tmp_path, env=env, config=config)) is in_use


@pytest.mark.parametrize(("env", "config", "in_use"), [
    (_CALLBACK_ENV, "", True),
    ("", "platforms:\n  wecom_callback:\n    extra: {corp_id: c}\n", True),
    ("", "platforms:\n  wecom_callback:\n    extra:\n      apps: [{corp_id: c, corp_secret: s}]\n", True),
    ("", "", False),
    ("WECOM_CALLBACK_CORP_ID=c\n", "", False),
    (_CALLBACK_ENV, "plugins:\n  disabled: [platforms/wecom]\n", False),
    (_CALLBACK_ENV, "plugins:\n  disabled: [wecom-platform]\n", False),
    (_CALLBACK_ENV, "plugins:\n  disabled: [wecom]\n", True),
])
def test_wecom_callback_in_use_matches_what_core_ran(tmp_path, env, config, in_use):
    assert lcm.wecom_callback_in_use(_home(tmp_path, env=env, config=config)) is in_use


@pytest.mark.parametrize("env", [_BOT_ENV + _CALLBACK_ENV, _CALLBACK_ENV])
def test_both_rows_install_the_one_wecom_plugin_once(tmp_path, env):
    home = _home(tmp_path, env=env, config=_CORE_ERA)
    calls = []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=lambda message: None) == ["wecom"]
    assert lcm.migrate_home(home, install=install, say=lambda message: None) == []
    assert calls == ["wecom"]


def test_a_fresh_home_installs_nothing(tmp_path):
    home = _home(tmp_path)
    calls = []
    assert lcm.migrate_home(home, install=lambda name: calls.append(name) or {"ok": True},
                            say=lambda message: None) == []
    assert calls == []


_ENABLED_ONLY = "platforms:\n  wecom:\n    enabled: true\n"


def _migrate_config(home: Path) -> None:
    from hermes_cli.config import migrate_config
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        migrate_config(interactive=False, quiet=True)
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize(("env", "config"), [(_BOT_ENV, ""), (_CALLBACK_ENV, ""), ("", _ENABLED_ONLY)],
                         ids=["bot", "callback", "enabled-only"])
def test_an_upgrade_crossing_v55_still_installs_the_plugin(tmp_path, monkeypatch, env, config):
    """`hermes update` migrates config.yaml past v55 before the left-core pass: the shared v55 step records that
    the home ran WeCom (either mode) in core, so the pass still installs the plugin once."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    home = _home(tmp_path, env=env, config=_CORE_ERA + config)
    _migrate_config(home)
    import hermes_yaml as yaml
    stamped = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert stamped["_config_version"] >= 55 and stamped["_left_core_used"] == ["wecom"]
    installed = lcm.migrate_home(home, install=lambda name: (home / "plugins" / name).mkdir(parents=True)
                                 or {"ok": True}, say=lambda _m: None)
    assert installed == ["wecom"]


@pytest.mark.parametrize(("env", "config"), [("", _ENABLED_ONLY), (_BOT_ENV, ""), (_CALLBACK_ENV, ""),
                                             (_BOT_ENV + _CALLBACK_ENV, "")],
                         ids=["enabled-only", "bot", "callback", "both-modes"])
def test_a_home_from_v55_on_that_sets_wecom_up_is_told_the_install_command(tmp_path, env, config):
    """Labs code is installed only when asked: a home that never ran core's WeCom gets one line naming
    `hermes plugins install wecom` (one, even with both modes set up), and nothing is installed or written."""
    home = _home(tmp_path, env=env, config="_config_version: 55\n" + config)
    before = (home / "config.yaml").read_bytes()
    said = []
    assert lcm.migrate_home(home, install=lambda name: pytest.fail("Labs code is installed only when asked"),
                            say=said.append) == []
    assert len(said) == 1 and "hermes plugins install wecom" in said[0]
    assert (home / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("config", ["", _ENABLED_ONLY], ids=["no-config", "unstamped"])
def test_a_home_that_cannot_prove_it_ran_core_wecom_is_told_the_install_command(tmp_path, config):
    """A missing or unstamped config.yaml is not proof of core-era use (a new home whose setup wrote only
    .env, both modes here, looks the same): it gets the one install command, also after its first stamp."""
    from hermes_cli.config import save_config
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    home = _home(tmp_path, env=_BOT_ENV + _CALLBACK_ENV, config=config)
    for _ in range(2):
        said = []
        lcm._attempted.clear()
        (home / "cache" / "left-core.failed-wecom").unlink(missing_ok=True)
        assert lcm.migrate_home(home, install=lambda name: pytest.fail("Labs code is installed only when asked"),
                                say=said.append) == []
        assert len(said) == 1 and "hermes plugins install wecom" in said[0], said
        token = set_hermes_home_override(home)
        try:
            save_config({"display": {"compact": True}}, merge_existing=True)  # the home's first stamp
        finally:
            reset_hermes_home_override(token)
    assert not (home / "plugins" / "wecom").exists()


def test_a_bare_wecom_in_plugins_disabled_is_kept_with_one_enable_notice(tmp_path):
    home = _home(tmp_path, env=_BOT_ENV, config=_CORE_ERA + "plugins:\n  disabled: [wecom]\n")
    said = []

    def install(name):
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True, "kept_disabled": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == ["wecom"]
    assert len(said) == 1 and said[0].count("plugins.disabled lists 'wecom'") == 1, said
    assert "hermes plugins enable wecom" in said[0]


def test_no_channels_install_takes_the_labs_plugin_for_a_set_up_wecom_home(monkeypatch):
    """Neither Channels page lists a Labs platform, so the Channels install route refuses one even for a
    home that has WeCom (here callback mode) set up: the explicit choice is `hermes plugins install wecom`."""
    import hermes_cli.plugins_cmd_install as pci
    from hermes_cli import plugin_catalog
    from hermes_cli.catalog_platform_listing import install_catalog_platform
    from hermes_cli.plugin_catalog import PluginCatalogEntry
    from hermes_constants import get_hermes_home

    entry = PluginCatalogEntry(name="wecom", repo="https://github.com/NousResearch/hermes-labs-plugins",
                               sha="1" * 40, description="WeCom.", maintainer="Nous Research", tier="community",
                               category="platform")
    monkeypatch.setattr(plugin_catalog, "load_catalog_live", lambda: [entry])
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda *a, **k: [entry])
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(pci, "dashboard_install_plugin", lambda *a, **k: pytest.fail("Labs code installed from a list"))
    home = get_hermes_home()
    (home / "config.yaml").write_text("_config_version: 55\n", encoding="utf-8")
    (home / ".env").write_text(_CALLBACK_ENV, encoding="utf-8")
    assert install_catalog_platform("wecom")["ok"] is False


@pytest.fixture
def _plugin_platform():
    """A throwaway plugin PlatformEntry, removed after the test."""
    from gateway.platform_registry import PlatformEntry, platform_registry
    names = []

    def register(name, **kwargs):
        platform_registry.register(PlatformEntry(name=name, label=name, adapter_factory=lambda cfg: None,
                                                 check_fn=lambda: True, source="plugin", **kwargs))
        names.append(name)

    yield register
    for name in names:
        platform_registry.unregister(name)


def test_a_plugin_platform_binds_a_port_only_when_it_declares_listen_port(_plugin_platform):
    from gateway.config import platform_binds_port
    _plugin_platform("seam-listener", listen_port=("port", 8645))
    _plugin_platform("seam-quiet")
    assert platform_binds_port("seam-listener") is True
    assert platform_binds_port("seam-quiet") is False


@pytest.mark.asyncio
async def test_media_of_a_media_via_adapter_platform_goes_through_the_live_adapter(_plugin_platform, monkeypatch):
    from gateway.outbound import send
    from gateway.config import Platform
    from gateway.platforms.send_targets import StandaloneSendPolicy
    _plugin_platform("seam-media", standalone_send=StandaloneSendPolicy(text_direct=True, media_via_adapter=True))
    routed = []

    async def via_adapter(platform, pconfig, chat_id, chunk, *, thread_id=None, media_files=None,
                          force_document=False):
        routed.append((platform.value, chat_id, chunk, media_files))
        return {"success": True}

    monkeypatch.setattr(send, "_send_via_adapter", via_adapter)
    media = [("/nonexistent/a.png", False)]
    await send._send_to_platform(Platform("seam-media"), SimpleNamespace(extra={}), "c1", "hi", media_files=media)
    assert routed == [("seam-media", "c1", "hi", media)]


def test_the_callback_toolset_keeps_its_core_era_name(_plugin_platform):
    import toolsets
    _plugin_platform("seam_cb")
    assert toolsets._plugin_platform_bundle("hermes-seam-cb")
    assert toolsets._plugin_platform_bundle("hermes-seam_cb")
    assert toolsets._plugin_platform_bundle("hermes-seam-unknown") == []


def test_a_positional_reply_file_over_keeps_its_meaning():
    from gateway.platforms.send_targets import StandaloneSendPolicy
    names = [f.name for f in __import__("dataclasses").fields(StandaloneSendPolicy)]
    assert names.index("media_via_adapter") > names.index("reply_file_over")


def test_listen_port_is_read_in_the_owning_profiles_plugin_scope(_plugin_platform, tmp_path, monkeypatch):
    # A multiplexed secondary's plugin registers in that profile's scope only.
    from gateway import config as gw_config
    from gateway.platform_registry import platform_registry
    from hermes_constants import get_hermes_home
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    seen = []
    real_get = platform_registry.get

    def scoped_get(name):
        seen.append(str(get_hermes_home()))
        return real_get(name)

    monkeypatch.setattr(platform_registry, "get", scoped_get)
    _plugin_platform("seam-scoped", listen_port=("port", 8645))
    assert gw_config.platform_binds_port("seam-scoped", home=secondary) is True
    assert seen[-1] == str(secondary)
