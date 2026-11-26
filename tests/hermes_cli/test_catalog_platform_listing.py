"""Catalog platforms a home does not have yet: listed by every setup surface, installed on pick.

Also: a platform plugin installed after this process imported config is labelled and masked from
its own manifest on the Channels card, not shown as raw env names in plain text.
"""

import pytest

from gateway.platform_registry import PlatformEntry, platform_registry
from hermes_constants import get_hermes_home


LABS = "https://github.com/NousResearch/hermes-labs-plugins"


@pytest.fixture
def fixture_catalog(monkeypatch):
    """A small catalog, one entry per rule the list applies, instead of the shipped one."""
    from hermes_cli import plugin_catalog
    from hermes_cli.plugin_catalog import PluginCatalogEntry

    def entry(name, tier, repo="https://github.com/NousResearch/hermes-official-plugins", category="platform"):
        return PluginCatalogEntry(name=name, repo=repo, sha="1" * 40, description=f"{name} platform. More words.",
                                  maintainer="Nous Research", tier=tier, category=category)

    entries = [entry("chatone", "official"), entry("chattwo", "verified"), entry("labschat", "community", repo=LABS),
               entry("commchat", "community", repo="https://github.com/someone/commchat"),
               entry("toolx", "official", category="desktop")]
    monkeypatch.setattr(plugin_catalog, "load_catalog_live", lambda: list(entries))
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda *a, **k: list(entries))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)  # the suite's opt-out; config decides here


def test_only_supported_catalog_platforms_are_offered_not_labs_or_community(fixture_catalog):
    from hermes_cli.catalog_platform_listing import available_catalog_platforms, install_catalog_platform

    rows = {row.name: row for row in available_catalog_platforms(provided=())}
    # Hermes Labs, community and non-platform entries are not setup picks: the plugin browser has them.
    assert set(rows) == {"chatone", "chattwo"}
    assert rows["chatone"].badge == "official" and rows["chattwo"].badge == "verified"
    assert rows["chatone"].install_command == "hermes plugins install chatone"
    assert rows["chatone"].summary == "chatone platform." and rows["chatone"].blocked == ""
    assert "chatone" not in {row.name for row in available_catalog_platforms(provided={"chatone"})}
    for name in ("labschat", "commchat"):  # a hand-crafted request does not get past the list's rule
        refused = install_catalog_platform(name)
        assert refused["ok"] is False and "not a catalog platform this home can add" in refused["error"]
    assert not (get_hermes_home() / "plugins" / "labschat").exists()


def test_setup_menu_lists_catalog_platforms_and_lazy_off_says_why(fixture_catalog):
    from hermes_cli import gateway

    rows = gateway._all_platforms(catalog=True)
    pick = next(p for p in rows if p["key"] == "chatone" and p.get("_catalog_platform"))
    assert gateway._platform_status(pick) == "not installed · official"
    assert not any(p["key"] == "labschat" and p.get("_catalog_platform") for p in rows)
    assert not any(p.get("_catalog_platform") for p in gateway._all_platforms())  # summaries: no progress

    (get_hermes_home() / "config.yaml").write_text("security:\n  allow_lazy_installs: false\n", encoding="utf-8")
    from hermes_cli.catalog_platform_listing import install_catalog_platform
    refused = install_catalog_platform("chatone")
    assert refused["ok"] is False and "allow_lazy_installs is off" in refused["error"]
    assert refused["install_command"] == "hermes plugins install chatone"


def test_platform_plugin_installed_after_import_is_labelled_and_masked():
    from hermes_cli.web_routers.messaging import _messaging_env_info
    from hermes_cli.web_server_messaging import _messaging_platform_catalog

    plugin_dir = get_hermes_home() / "plugins" / "lateplat"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: lateplat\nkind: platform\nlabel: Late Platform\n"
        "requires_env:\n  - name: LATEPLAT_APP_SECRET\n    prompt: App Secret\n"
        "optional_env:\n  - name: LATEPLAT_ALLOWED_USERS\n    prompt: Allowed users\n    password: false\n",
        encoding="utf-8")
    entry = PlatformEntry(name="lateplat", label="Late Platform", adapter_factory=lambda cfg: None,
                          check_fn=lambda: True, required_env=["LATEPLAT_APP_SECRET"], plugin_name="lateplat")
    scope = platform_registry.current_scope_key()
    platform_registry.register(entry, scope=scope)
    try:
        card = next(c for c in _messaging_platform_catalog() if c["id"] == "lateplat")
        assert card["env_vars"] == ("LATEPLAT_APP_SECRET", "LATEPLAT_ALLOWED_USERS")
        secret = _messaging_env_info("LATEPLAT_APP_SECRET", card["env_info"].get("LATEPLAT_APP_SECRET"))
        assert secret["is_password"] is True and secret["prompt"] == "App Secret"
        assert _messaging_env_info("LATEPLAT_ALLOWED_USERS", card["env_info"]["LATEPLAT_ALLOWED_USERS"])["is_list"]
    finally:
        platform_registry.unregister("lateplat", scope=scope)


def _install_late_platform(dir_name: str, platform: str, *, catalog_name: str = "",
                           repo: str = "https://github.com/NousResearch/hermes-official-plugins") -> None:
    """What an installer running in another process leaves behind: the tree, its install record when
    it came from the catalog, and the enable in config.yaml."""
    import json

    home = get_hermes_home()
    plugin_dir = home / "plugins" / dir_name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(f"name: {dir_name}\nkind: platform\nversion: 1.0.0\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    ctx.register_platform(name={platform!r}, label='Late', adapter_factory=lambda cfg: None,\n"
        "                          check_fn=lambda: True, required_env=['LATE_TOKEN'])\n", encoding="utf-8")
    if catalog_name:
        record = {"catalog": {"name": catalog_name, "repo": repo, "tier": "official" if repo != LABS else "community",
                              "sha": "0" * 40, "pin": "0" * 40}}
        (home / "plugins" / ".install-metadata.json").write_text(json.dumps({dir_name: record}), encoding="utf-8")
    (home / "config.yaml").write_text(f"plugins:\n  enabled:\n    - {dir_name}\n", encoding="utf-8")


def test_channels_page_open_before_an_install_elsewhere_lists_it_on_the_next_read():
    from hermes_cli.web_server_messaging import _messaging_platform_catalog

    assert "lateplug" not in {card["id"] for card in _messaging_platform_catalog()}  # discovery ran, cached
    _install_late_platform("lateplug", "lateplug")
    assert "lateplug" in {card["id"] for card in _messaging_platform_catalog()}


def test_catalog_platform_installed_by_another_process_is_not_offered_and_install_says_already(fixture_catalog):
    from hermes_cli.catalog_platform_listing import available_catalog_platforms, install_catalog_platform

    assert "chatone" in {row.name for row in available_catalog_platforms()}  # the page drew it as installable
    _install_late_platform("chatone", "chatone", catalog_name="chatone")  # the background default install lands
    assert "chatone" not in {row.name for row in available_catalog_platforms()}
    result = install_catalog_platform("chatone")  # the stale page's Install click
    assert result["ok"] is True and result["already_installed"] is True
    assert result["plugin_name"] == "chatone" and result["platforms"] == ["chatone"]


def test_an_installed_labs_platform_keeps_its_channel_card(fixture_catalog):
    # A home that already has a Hermes Labs platform (installed by hand or by the left-core migration)
    # configures it like any other: only the not-installed offer leaves out Labs.
    from hermes_cli.catalog_platform_listing import available_catalog_platforms
    from hermes_cli.web_server_messaging import _messaging_platform_catalog

    _install_late_platform("labschat", "labschat", catalog_name="labschat", repo=LABS)
    assert "labschat" in {card["id"] for card in _messaging_platform_catalog()}
    assert "labschat" not in {row.name for row in available_catalog_platforms()}


def test_channels_page_poll_signature_moves_when_a_platform_plugin_lands_elsewhere():
    import asyncio

    from hermes_cli.web_routers.messaging import get_messaging_platforms_signature

    def poll():
        return asyncio.run(get_messaging_platforms_signature())["signature"]

    loaded = poll()
    assert poll() == loaded  # nothing moved: the page does not reload
    _install_late_platform("lateplug", "lateplug")  # the background default install lands
    assert poll() != loaded
