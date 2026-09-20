"""Install, configure, authorize, select and remove an external memory provider in one named profile
through the real installer and HTTP routes; the catalog entry and Git repository are synthetic and local."""

import json
import os
import subprocess
import textwrap
import time

import pytest
import yaml


PLUGIN = "lifecycle_probe"
REPO_URL = "https://fixture.invalid/lifecycle-probe.git"
PACKAGE = {
    "plugin.yaml": "name: lifecycle_probe\nkind: memory\nversion: '1.0.0'\ndescription: Offline lifecycle fixture\n",
    "__init__.py": '''
        import json
        from agent.memory_provider import MemoryProvider
        from hermes_constants import get_hermes_home
        class LifecycleProbe(MemoryProvider):
            name = "lifecycle_probe"
            def is_available(self):
                path = get_hermes_home() / "lifecycle_probe" / "config.json"
                data = json.loads(path.read_text()) if path.exists() else {}
                return bool(data.get("workspace") and data.get("grant"))
            def initialize(self, session_id, **kwargs): raise AssertionError("Settings must not initialize a session")
            def get_tool_schemas(self): return []
            def get_config_schema(self): return [{"key": "workspace", "required": True}, {"key": "grant", "secret": True}]
    ''',
    "config_schema.py": '''
        from plugins.memory.config_schema import ProviderConfigSchema, ProviderField
        CONFIG_SCHEMA = ProviderConfigSchema(name="lifecycle_probe", label="Offline fixture", fields=(
            ProviderField(key="workspace", label="Workspace"),
            ProviderField(key="grant", label="Fixture grant", kind="secret"),
        ))
    ''',
    "oauth_flow.py": '''
        import json, os
        from pathlib import Path
        from agent.secret_scope import get_secret
        def _config(hermes_home):
            path = Path(hermes_home) / "lifecycle_probe" / "config.json"
            return path, json.loads(path.read_text()) if path.exists() else {}
        def get_flow_status(*, hermes_home):
            connected = bool(_config(hermes_home)[1].get("grant"))
            return {"state": "connected" if connected else "idle", "connected": connected, "detail": "private"}
        def start_loopback_flow_background(*, hermes_home):
            path, data = _config(hermes_home)
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({**data, "grant": "offline-" + get_secret("LIFECYCLE_OWNER")}))
            Path(hermes_home, "oauth-receipt").write_text(str(os.getpid()))
    ''',
}


def _git(directory, *args):
    return subprocess.check_output(["git", *args], cwd=directory, text=True).strip()


def _state(home):
    files = [home / "config.yaml", home / ".env", *(home / "plugins").rglob("*")]
    return {str(p.relative_to(home)): p.read_bytes() for p in files if p.is_file() and "__pycache__" not in p.parts}


@pytest.fixture
def offline_lifecycle(memory_homes, tmp_path, monkeypatch):
    from hermes_cli import plugin_catalog, plugins, plugins_cmd

    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-git-config"))
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    for home in memory_homes.values():
        (home / "config.yaml").write_text(yaml.safe_dump({"memory": {"provider": ""}, "terminal": {"backend": "local"}}))
        (home / ".env").write_text(f"LIFECYCLE_OWNER={home.name}\n")
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: tmp_path / "no-bundled")
    repo = tmp_path / "fixture-repo"
    repo.mkdir()
    for filename, source in PACKAGE.items():
        (repo / filename).write_text(textwrap.dedent(source))
    commit = ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm")
    for args in (("init", "-q"), ("add", "."), (*commit, "offline fixture")):
        _git(repo, *args)
    catalog = {"entries": [{"name": PLUGIN, "repo": REPO_URL, "sha": _git(repo, "rev-parse", "HEAD"), "description": "Synthetic",
                            "maintainer": "Offline test fixture", "category": "memory"}], "removed": []}
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **kwargs: catalog)
    resolve = plugins_cmd._resolve_git_url
    monkeypatch.setattr(plugins_cmd, "_resolve_git_url", lambda identifier: (repo.as_uri(), resolve(identifier)[1]))
    return memory_homes


def _request(client, method, url, *, expected=200, **kwargs):
    response = client.request(method, url, params={"profile": "b", "surface": "declared"}, **kwargs)
    assert response.status_code == expected, response.text
    return response.json()


def _status(client, active=""):
    data = _request(client, "GET", "/api/memory")
    assert data["active"] == active
    return next(row for row in data["providers"] if row["name"] == PLUGIN)["status"]


def test_lifecycle_stays_in_the_named_profile(offline_lifecycle, dashboard_client):
    from hermes_cli.plugins_cmd import dashboard_install_plugin, dashboard_remove_user_plugin
    from hermes_cli.web_server_profiles import _config_profile_scope

    client, homes = dashboard_client, offline_lifecycle
    home, other_before = homes["b"], _state(homes["default"])
    with _config_profile_scope("b"):  # the install route has no profile parameter; the installer runs under B's scope
        result = dashboard_install_plugin("", force=False, enable=False, catalog_name=PLUGIN)
    assert result["ok"] and result["enabled"] is False, result
    assert (home / "plugins" / PLUGIN / "oauth_flow.py").is_file()

    url = f"/api/memory/providers/{PLUGIN}"
    assert _status(client) == "needs_config"
    form = _request(client, "GET", url + "/config")
    assert form["label"] == "Offline fixture" and form["capabilities"]["supports_partial_updates"] is True
    _request(client, "PUT", url + "/config", json={"values": {"workspace": "b"}, "activate": False})
    assert json.loads((home / PLUGIN / "config.json").read_text()) == {"workspace": "b"}
    assert _request(client, "GET", url + "/oauth/status")["state"] == "idle"
    _request(client, "POST", url + "/oauth/start")
    deadline = time.monotonic() + 10  # the hermes_home hook runs in a thread of this process
    while (oauth := _request(client, "GET", url + "/oauth/status"))["state"] != "connected":
        assert time.monotonic() < deadline, oauth
        time.sleep(0.02)
    assert oauth == {"supported": True, "state": "connected", "connected": True, "detail": "Connected"}
    assert json.loads((home / PLUGIN / "config.json").read_text())["grant"] == "offline-b"  # B's secrets, in this process
    assert (home / "oauth-receipt").read_text() == str(os.getpid())
    assert next(f for f in _request(client, "GET", url + "/config")["fields"] if f["key"] == "grant")["is_set"] is True
    assert _status(client) == "ready"

    _request(client, "PUT", "/api/memory/provider", json={"provider": PLUGIN})
    assert yaml.safe_load((home / "config.yaml").read_text())["memory"]["provider"] == PLUGIN
    assert _status(client, active=PLUGIN) == "ready" and _state(homes["default"]) == other_before
    with _config_profile_scope("b"):
        assert dashboard_remove_user_plugin(PLUGIN)["ok"] is True
    assert not (home / "plugins" / PLUGIN).exists() and PLUGIN not in json.loads((home / "plugins/.install-metadata.json").read_text())
    assert _status(client, active=PLUGIN) == "missing" and _state(homes["default"]) == other_before
