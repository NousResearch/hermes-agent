"""The dashboard turns a plugin on by the same identity the plugin loader uses.

A plugin is installed in a directory (its key, what ``hermes plugins enable`` writes) and names itself
in ``plugin.yaml``; its ``dashboard/manifest.json`` carries the dashboard name (route prefix and tab
id). The dashboard must agree with the loader on which installed plugins are on:

- a plugin enabled by its directory key mounts, even when its dashboard name differs;
- a plugin that is not enabled stays inert even when its dashboard name is another, enabled plugin's
  name, and that other plugin keeps its own dashboard entry.

Driven through the real app and real discovery against a temp HERMES_HOME; each plugin backend
touches an ``IMPORTED`` marker when Python runs it.
"""
from __future__ import annotations

import json
import sys

import pytest

from hermes_cli import web_server

_PLUGIN_API = '''
from pathlib import Path

from fastapi import APIRouter

Path(__file__).with_name("IMPORTED").touch()
router = APIRouter()


@router.get("/ping")
async def ping():
    return {"pong": Path(__file__).parent.parent.name}
'''


def _install(home, directory, dashboard_name, *, yaml_name=None):
    """A user plugin in ``plugins/<directory>`` whose dashboard calls itself ``dashboard_name``."""
    plugin = home / "plugins" / directory
    dashboard = plugin / "dashboard"
    dashboard.mkdir(parents=True)
    if yaml_name is not None:
        (plugin / "plugin.yaml").write_text(f"name: {yaml_name}\nversion: 1.0.0\n")
    (dashboard / "manifest.json").write_text(json.dumps({"name": dashboard_name, "api": "plugin_api.py"}))
    (dashboard / "plugin_api.py").write_text(_PLUGIN_API)
    (dashboard / "index.js").write_text("export {}\n")
    return dashboard / "IMPORTED"


def _plugins(home, *, enabled=(), disabled=()):
    (home / "config.yaml").write_text(json.dumps({"plugins": {
        "enabled": list(enabled), "disabled": list(disabled), "isolation": "in_process"}}))


@pytest.fixture
def dash(_isolate_hermes_home):
    """The dashboard app over a temp home; ``start()`` runs the startup discovery and mount."""
    from types import SimpleNamespace

    from starlette.testclient import TestClient

    from hermes_cli.web_server_dashboard import _invalidate_plugins_hub_cache, _mount_plugin_api_routes
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    routes, modules = list(web_server.app.router.routes), set(sys.modules)
    owner = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    client = TestClient(web_server.app)

    def start():
        # A fresh startup: plugin routers mounted when the app module loaded are not this home's.
        web_server.app.router.routes[:] = [r for r in routes if not getattr(r, "path", "").startswith("/api/plugins/")]
        web_server._dashboard_plugins_cache = None
        _invalidate_plugins_hub_cache()
        _mount_plugin_api_routes()
        # Mounted after the SPA catch-all here (at real startup it comes later): match them first.
        fresh = [r for r in web_server.app.router.routes if getattr(r, "path", "").startswith("/api/plugins/")]
        web_server.app.router.routes[:] = fresh + [r for r in web_server.app.router.routes if r not in fresh]

    def listed():
        return {p["name"]: p for p in client.get("/api/dashboard/plugins", headers=owner).json()}

    def hub_rows():
        _invalidate_plugins_hub_cache()
        return {r["name"]: r for r in client.get("/api/dashboard/plugins/hub", headers=owner).json()["plugins"]}

    try:
        yield SimpleNamespace(home=home, client=client, owner=owner, start=start, listed=listed, hub_rows=hub_rows,
                              ping=lambda name: client.get(f"/api/plugins/{name}/ping", headers=owner),
                              asset=lambda name: client.get(f"/dashboard-plugins/{name}/index.js"),
                              rescan=lambda: client.get("/api/dashboard/plugins/rescan", headers=owner))
    finally:
        web_server.app.router.routes[:] = routes
        web_server._dashboard_plugins_cache = None
        _invalidate_plugins_hub_cache()
        for name in set(sys.modules) - modules:
            if name.startswith("hermes_dashboard_plugin_"):
                del sys.modules[name]


def _assert_stays_inert(dash, imported, name):
    """Not imported at startup, on a rescan, or by a request; its route, files and tab are refused."""
    assert not imported.exists(), "imported at startup"
    assert dash.rescan().status_code == 200
    response = dash.ping(name)
    assert response.status_code == 404 or response.json() != {"pong": "harmless"}, response.text
    assert not imported.exists(), "imported on rescan / first request"
    assert dash.asset(name).status_code == 404
    assert dash.listed().get(name, {"source": "bundled"})["source"] == "bundled"


@pytest.mark.parametrize("bundled_owner", ["disk-cleanup", "kanban"])
def test_unenabled_plugin_borrowing_a_bundled_plugins_name_stays_inert(dash, bundled_owner):
    """``harmless`` is installed but not on; its dashboard manifest says ``bundled_owner``, a bundled
    plugin the user enabled. Its backend never runs and the bundled plugin keeps its own entry."""
    imported = _install(dash.home, "harmless", bundled_owner, yaml_name="harmless")
    _plugins(dash.home, enabled=[bundled_owner])
    dash.start()

    _assert_stays_inert(dash, imported, bundled_owner)
    entries = [p for p in web_server._get_dashboard_plugins() if p["name"] == bundled_owner]
    assert all(p["source"] == "bundled" for p in entries), entries
    assert (bundled_owner == "kanban") == bool(entries)
    rows = dash.hub_rows()
    assert rows["harmless"]["runtime_status"] == "inactive"
    owner_manifest = rows.get(bundled_owner, {}).get("dashboard_manifest")
    assert owner_manifest is None or owner_manifest["source"] == "bundled"


def test_unenabled_plugin_borrowing_a_user_plugins_name_stays_inert(dash):
    """The same with an enabled user plugin as the name's owner: its own tab, route and files stay
    exactly as if ``harmless`` were not installed."""
    imported = _install(dash.home, "harmless", "owner-tool", yaml_name="harmless")
    owner_imported = _install(dash.home, "owner-tool", "owner-tool", yaml_name="owner-tool")
    _plugins(dash.home, enabled=["owner-tool"])
    dash.start()

    assert not imported.exists()
    assert owner_imported.exists()
    assert dash.ping("owner-tool").json() == {"pong": "owner-tool"}
    assert dash.asset("owner-tool").status_code == 200
    assert dash.rescan().status_code == 200
    assert dash.ping("owner-tool").json() == {"pong": "owner-tool"}
    assert not imported.exists()
    entry = next(p for p in web_server._get_dashboard_plugins() if p["name"] == "owner-tool")
    assert entry["_dir"].endswith("owner-tool/dashboard")
    assert "owner-tool" in dash.listed()
    rows = dash.hub_rows()
    assert rows["harmless"]["runtime_status"] == "inactive"
    assert rows["owner-tool"]["runtime_status"] == "enabled"


@pytest.mark.parametrize("tab", ["own-tab", "borrowed"])
@pytest.mark.parametrize("owner", ["disk-cleanup", "owner-tool"])
def test_a_plugin_yaml_name_borrowed_from_another_plugin_grants_no_consent(dash, owner, tab):
    """``plugins.enabled`` naming a plugin never turns on another directory whose plugin.yaml claims that
    name (the loader skips such a directory too), whatever its dashboard name."""
    tab = owner if tab == "borrowed" else tab
    if owner == "owner-tool":
        _install(dash.home, "owner-tool", "owner-tool", yaml_name="owner-tool")
    imported = _install(dash.home, "other-dir", tab, yaml_name=owner)
    _plugins(dash.home, enabled=[owner])
    dash.start()

    assert not imported.exists()
    response = dash.ping(tab)
    assert response.status_code == 404 or response.json() == {"pong": "owner-tool"}  # never the other plugin
    assert tab == "owner-tool" or tab not in dash.listed()
    assert not imported.exists()
    if owner == "owner-tool":  # the plugin that owns the name keeps it
        assert dash.ping("owner-tool").json() == {"pong": "owner-tool"}


def test_a_plugin_yaml_name_another_plugin_also_answers_to_still_disables_it(dash):
    """Owning a name decides consent only: ``plugins.disabled`` still matches every name a plugin declares."""
    _install(dash.home, "shared-name", "shared-name", yaml_name="shared-name")
    imported = _install(dash.home, "dash-dir", "dir-tab", yaml_name="shared-name")
    _plugins(dash.home, enabled=["dash-dir"], disabled=["shared-name"])
    dash.start()

    assert not imported.exists()
    assert dash.ping("dir-tab").status_code == 404
    assert dash.asset("dir-tab").status_code == 404
    assert not imported.exists()


def test_a_plugin_enabled_by_its_own_plugin_yaml_name_mounts(dash):
    """The loader's other spelling: ``plugins.enabled`` naming the plugin.yaml name turns it on too."""
    imported = _install(dash.home, "dash-dir", "dir-tab", yaml_name="yaml-only-name")
    _plugins(dash.home, enabled=["yaml-only-name"])
    dash.start()

    assert imported.exists()
    assert dash.ping("dir-tab").json() == {"pong": "dash-dir"}


def _categorized(root, category, directory, yaml_name):
    """A plugin one category down (``plugins/<category>/<directory>``) with no dashboard of its own."""
    plugin = root / "plugins" / category / directory
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {yaml_name}\nversion: 1.0.0\n")


def test_a_same_named_directory_one_category_down_is_another_plugin(dash):
    """``plugins/foo-tool`` and ``plugins/memory/foo-tool`` are two plugins: the name both declare in
    plugin.yaml turns neither dashboard on, as for any two plugins that share a name."""
    imported = _install(dash.home, "foo-tool", "foo-tab", yaml_name="shared-name")
    _categorized(dash.home, "memory", "foo-tool", "shared-name")
    _plugins(dash.home, enabled=["shared-name"])
    dash.start()

    assert not imported.exists()
    assert dash.ping("foo-tab").status_code == 404
    assert not imported.exists()


def test_a_dashboard_name_from_a_same_named_directory_one_category_down_is_not_registered(dash):
    """The other plugin's plugin.yaml name is not this plugin's dashboard name either."""
    imported = _install(dash.home, "foo-tool", "shared-name", yaml_name="foo-tool")
    _categorized(dash.home, "memory", "foo-tool", "shared-name")
    _plugins(dash.home, enabled=["shared-name", "foo-tool"])
    dash.start()

    assert not imported.exists()
    assert "shared-name" not in dash.listed()
    assert dash.ping("shared-name").status_code == 404


@pytest.fixture
def profile_process(dash, tmp_path, monkeypatch):
    """A process launched as profile ``work`` (``HERMES_HOME=<root>/profiles/work``): the dashboard
    scans the profile's own ``plugins/`` and then the root's."""
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return root, profile


def test_a_profile_plugin_and_a_root_plugin_in_same_named_directories_are_two_plugins(dash, profile_process):
    """The profile's ``plugins/foo-tool`` and the root's are different plugins: a plugin.yaml name
    both declare turns neither on."""
    root, profile = profile_process
    profile_imported = _install(profile, "foo-tool", "profile-tab", yaml_name="shared-name")
    root_imported = _install(root, "foo-tool", "root-tab", yaml_name="shared-name")
    _plugins(profile, enabled=["shared-name"])
    dash.start()

    assert dash.ping("profile-tab").status_code == 404
    assert dash.ping("root-tab").status_code == 404
    assert not profile_imported.exists() and not root_imported.exists()


def test_a_root_plugin_cannot_take_the_profile_plugins_name_from_a_same_named_directory(dash, profile_process):
    """The root's ``foo-tool`` whose dashboard name is the profile's ``foo-tool`` plugin.yaml name is
    not registered; the profile's plugin, enabled by that name, keeps it."""
    root, profile = profile_process
    profile_imported = _install(profile, "foo-tool", "profile-tab", yaml_name="profile-name")
    root_imported = _install(root, "foo-tool", "profile-name", yaml_name="foo-tool")
    _plugins(profile, enabled=["profile-name"])
    dash.start()

    assert profile_imported.exists()
    assert dash.ping("profile-tab").json() == {"pong": "foo-tool"}
    assert "profile-name" not in dash.listed()
    assert not root_imported.exists()


def test_a_profile_and_root_copy_enabled_by_directory_name_behave_as_before(dash, profile_process):
    """Control: the profile's copy of a plugin installed in both places wins its dashboard name."""
    root, profile = profile_process
    profile_imported = _install(profile, "foo-tool", "foo-tool", yaml_name="foo-tool")
    root_imported = _install(root, "foo-tool", "foo-tool", yaml_name="foo-tool")
    _plugins(profile, enabled=["foo-tool"])
    dash.start()

    assert profile_imported.exists()
    assert not root_imported.exists()
    assert dash.listed()["foo-tool"]["source"] == "user"
    assert dash.ping("foo-tool").status_code == 200


def test_plugin_enabled_by_its_directory_key_mounts_and_lists(dash):
    """``hermes plugins enable hermes-mobile`` turns on a plugin whose dashboard calls itself ``mobile``."""
    imported = _install(dash.home, "hermes-mobile", "mobile", yaml_name="hermes-mobile")
    _plugins(dash.home, enabled=["hermes-mobile"])
    dash.start()

    assert imported.exists()
    assert dash.ping("mobile").json() == {"pong": "hermes-mobile"}
    assert dash.asset("mobile").status_code == 200
    assert "mobile" in dash.listed()
    assert dash.hub_rows()["hermes-mobile"]["runtime_status"] == "enabled"


def test_plugin_enabled_by_its_directory_key_hot_mounts(dash):
    """The same plugin installed and enabled after startup mounts on its first request."""
    dash.start()
    imported = _install(dash.home, "hermes-mobile", "mobile", yaml_name="hermes-mobile")
    _plugins(dash.home, enabled=["hermes-mobile"])
    assert dash.rescan().status_code == 200

    assert dash.ping("mobile").json() == {"pong": "hermes-mobile"}
    assert imported.exists()


# Controls: the same answers on the base commit and with this change.

def test_plugin_enabled_by_its_dashboard_name_still_mounts(dash):
    imported = _install(dash.home, "dash-alias", "alias-tab")
    _plugins(dash.home, enabled=["alias-tab"])
    dash.start()

    assert imported.exists()
    assert dash.ping("alias-tab").json() == {"pong": "dash-alias"}
    assert dash.asset("alias-tab").status_code == 200
    assert "alias-tab" in dash.listed()


@pytest.mark.parametrize("enabled", [True, False])
def test_plugin_whose_dashboard_name_is_its_directory_behaves_as_before(dash, enabled):
    imported = _install(dash.home, "same-name", "same-name", yaml_name="same-name")
    _plugins(dash.home, enabled=["same-name"] if enabled else [])
    dash.start()

    assert imported.exists() is enabled
    assert (dash.ping("same-name").status_code == 200) is enabled
    assert (dash.asset("same-name").status_code == 200) is enabled
    assert ("same-name" in dash.listed()) is enabled


def test_same_named_directory_still_overrides_the_bundled_plugin(dash):
    imported = _install(dash.home, "kanban", "kanban", yaml_name="kanban")
    _plugins(dash.home, enabled=["kanban"])
    dash.start()

    assert imported.exists()
    assert dash.ping("kanban").json() == {"pong": "kanban"}
    entry = next(p for p in web_server._get_dashboard_plugins() if p["name"] == "kanban")
    assert entry["source"] == "user"
    assert dash.listed()["kanban"]["source"] == "user"


@pytest.mark.parametrize(("directory", "dashboard_name"), [("dash-alias", "alias-tab"), ("same-name", "same-name")])
def test_disabled_by_dashboard_name_still_wins(dash, directory, dashboard_name):
    imported = _install(dash.home, directory, dashboard_name)
    _plugins(dash.home, enabled=[dashboard_name], disabled=[dashboard_name])
    dash.start()

    assert not imported.exists()
    assert dash.ping(dashboard_name).status_code == 404
    assert dash.asset(dashboard_name).status_code == 404
    assert dashboard_name not in dash.listed()
    assert not imported.exists()


def test_disabled_by_directory_key_wins_over_dashboard_name_enable(dash):
    """New with this change: ``hermes plugins disable dash-alias`` now also turns off its dashboard."""
    imported = _install(dash.home, "dash-alias", "alias-tab")
    _plugins(dash.home, enabled=["alias-tab"], disabled=["dash-alias"])
    dash.start()

    assert not imported.exists()
    assert dash.ping("alias-tab").status_code == 404
    assert "alias-tab" not in dash.listed()


@pytest.fixture
def project_plugin(dash, tmp_path, monkeypatch):
    """A project plugin (``./.hermes/plugins/proj-dir``, dashboard name ``proj-tab``) with the
    project source turned on."""
    repo = tmp_path / "repo"
    imported = _install(repo / ".hermes", "proj-dir", "proj-tab")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
    return imported


@pytest.mark.parametrize("disabled_as", ["proj-tab", "proj-dir"])
def test_disabled_project_plugin_is_not_listed_or_served(dash, project_plugin, disabled_as):
    """A project plugin in ``plugins.disabled`` (by its dashboard name or its directory) stays off."""
    _plugins(dash.home, disabled=[disabled_as])
    dash.start()

    assert "proj-tab" not in dash.listed()
    assert dash.asset("proj-tab").status_code == 404
    assert not project_plugin.exists()


@pytest.mark.parametrize("enabled_as", ["hermes-mobile", "mobile"])
def test_project_plugin_cannot_take_a_user_plugins_dashboard_name(dash, tmp_path, monkeypatch, enabled_as):
    """A project plugin (opt-in, ships with the CWD) whose directory is a user plugin's dashboard name
    does not displace that plugin: user and bundled entries still outrank the project source."""
    imported = _install(dash.home, "hermes-mobile", "mobile", yaml_name="hermes-mobile")
    repo = tmp_path / "repo"
    project_imported = _install(repo / ".hermes", "mobile", "mobile")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
    _plugins(dash.home, enabled=[enabled_as])
    dash.start()

    assert imported.exists()
    assert dash.ping("mobile").json() == {"pong": "hermes-mobile"}
    assert dash.listed()["mobile"]["source"] == "user"
    assert not project_imported.exists()


def test_project_plugin_without_a_disabled_entry_is_listed_as_before(dash, project_plugin):
    """Control: no enable entry is needed for a project plugin; its files are served, its Python never runs."""
    _plugins(dash.home)
    dash.start()

    assert dash.listed()["proj-tab"]["source"] == "project"
    assert dash.asset("proj-tab").status_code == 200
    assert dash.ping("proj-tab").status_code == 404
    assert not project_plugin.exists()
