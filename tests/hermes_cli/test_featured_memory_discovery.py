"""CLI and HTTP discovery keep install, configure and selection separate."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from hermes_cli import memory_setup, plugin_catalog


def _entry(name="featured-provider", **kwargs):
    return plugin_catalog.PluginCatalogEntry(name=name, repo=f"https://github.com/example/{name}",
        sha="a" * 40, description="Featured memory", maintainer="example", category="memory", **kwargs)


@pytest.fixture
def catalog(monkeypatch):
    # A real offline catalog resolver; no external requests in picker/API tests.
    rows = [_entry(featured=True)]
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda *a: rows)
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **kw: None)
    monkeypatch.setattr(plugin_catalog, "load_removed_list", lambda *a: [])
    return rows


@pytest.mark.parametrize("choice", ["cancel", "marketplace", "schema-less"])
def test_picker_offers_missing_provider_without_mutating_until_install(catalog, choice, monkeypatch):
    from hermes_cli import config, plugins_cmd
    config.save_config({"memory": {"provider": "previous"}})
    before = (memory_setup.get_hermes_home() / "config.yaml").read_bytes()
    loaded = []
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: loaded)
    installs, deps = [], []

    class Provider:
        def get_config_schema(self):
            return []

    def install(name, **kwargs):
        installs.append((name, kwargs))
        assert config.load_config()["memory"]["provider"] == "previous"
        assert kwargs == {"enable": True, "select_memory": False}
        target = memory_setup.get_hermes_home() / "plugins" / name
        target.mkdir(parents=True)
        (target / "plugin.yaml").write_text(f"name: {name}\n")
        # The installer's own admission lists it as enabled; selection stays put.
        config.save_config({**config.load_config(), "plugins": {"enabled": [name]}})
        loaded.append((name, "no setup needed", Provider()))

    monkeypatch.setattr(plugins_cmd, "cmd_install", install)
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda name: deps.append(name))
    screens = []

    def select(title, items, **kwargs):
        screens.append(items)
        if len(screens) == 1:
            assert any("Featured" in label and "install" in (label + desc).lower() for label, desc in items)
            assert any("Built-in" in label for label, _ in items)
            assert any("Marketplace" in label for label, _ in items)
            if choice == "cancel":
                return -1
            word = "Marketplace" if choice == "marketplace" else "featured-provider"
            return next(i for i, (label, _) in enumerate(items) if word in label)
        assert "install" in title.lower()
        return 1

    monkeypatch.setattr(memory_setup, "_curses_select", select)
    memory_setup.cmd_setup(SimpleNamespace())
    assert screens
    assert deps == []
    if choice == "schema-less":
        assert installs == [("featured-provider", {"enable": True, "select_memory": False})]
        assert config.load_config()["memory"]["provider"] == "featured-provider"
    else:
        assert installs == []
        assert (memory_setup.get_hermes_home() / "config.yaml").read_bytes() == before


def test_installed_featured_choices_stay_prominent(catalog, monkeypatch):
    loaded = [("ordinary", "local", object()), ("featured-provider", "local", object())]
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: loaded)
    def select(title, items, **kwargs):
        assert items[0][0].startswith("Featured")
        assert "featured-provider" in items[0][0]
        assert "install required" not in str(items[0]).lower()
        assert sum("featured-provider" in label for label, _ in items) == 1
        return -1
    monkeypatch.setattr(memory_setup, "_curses_select", select)
    memory_setup.cmd_setup(SimpleNamespace())


def test_http_featured_discovery_alternates_homes_and_retains_missing(catalog, tmp_path, monkeypatch, request):
    import plugins.memory as memory
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_cli.config import atomic_config_write
    from agent import secret_scope

    home = tmp_path / ".hermes"
    second = home / "profiles" / "second"
    second.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")
    monkeypatch.setattr(memory, "_iter_entry_points", lambda: [])
    for owner in (home, second):
        atomic_config_write(owner / "config.yaml", {"memory": {"provider": "featured-provider"}})
    provider = home / "plugins" / "featured-provider"
    provider.mkdir(parents=True)
    (provider / "__init__.py").write_text('"""MemoryProvider awaiting dependencies."""\n')
    (provider / "plugin.yaml").write_text("name: featured-provider\ndescription: Installed fixture\n")
    before = {p: (p / "config.yaml").read_bytes() for p in (home, second)}
    client = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    # Fixture finalizer restores process mode even if an assertion fails.
    request.addfinalizer(lambda: secret_scope.set_multiplex_active(previous_multiplex))
    for profile in ("default", "second", "default"):
        response = client.get("/api/memory", params={"profile": profile})
        assert response.status_code == 200, response.text
        data = response.json()
        row = next(r for r in data["providers"] if r["name"] == "featured-provider")
        assert data["active"] == "featured-provider"
        if profile == "second":
            assert row["status"] == "missing" and row["featured"] is False
            assert data["catalog_providers"] == [{k: getattr(catalog[0], k) for k in
                ("name", "title", "description", "repo", "sha", "subdir", "featured")}]
        else:
            assert row["status"] != "missing" and row["featured"] is True
            assert data["catalog_providers"] == []
    monkeypatch.setattr(plugin_catalog, "load_catalog_live", lambda **kw: (_ for _ in ()).throw(OSError("offline")))
    data = client.get("/api/memory", params={"profile": "default"}).json()
    assert data["providers"][0]["featured"] is False
    assert data["catalog_providers"] == []
    assert all((p / "config.yaml").read_bytes() == raw for p, raw in before.items())


def test_http_discovery_reads_the_cached_catalog_without_a_live_fetch(monkeypatch):
    """GET /api/memory is polled by Settings and Command Center: an expired cache or a dead
    catalog host must not cost the request a network timeout."""
    import httpx
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    cached = _entry(name="cached-provider", featured=True)
    monkeypatch.setattr(plugin_catalog, "load_catalog", lambda *a: [])
    monkeypatch.setattr(plugin_catalog, "load_removed_list", lambda *a: [])
    monkeypatch.setattr(plugin_catalog, "_stale_live_cache", lambda path: {"entries": [cached.to_dict()]})

    def no_network(*args, **kwargs):
        raise AssertionError("GET /api/memory reached the network")

    monkeypatch.setattr(httpx, "get", no_network)
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", no_network)
    response = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN}).get("/api/memory")
    assert response.status_code == 200, response.text
    assert [row["name"] for row in response.json()["catalog_providers"]] == ["cached-provider"]
