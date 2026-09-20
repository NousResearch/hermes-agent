"""/api/memory and /api/memory/provider act on the requested profile only."""


import json

import pytest
import yaml


PLUGIN = "scope_probe"
PROVIDER = '''from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home

class ScopeProbe(MemoryProvider):
    name = "scope_probe"

    def is_available(self):
        return get_secret("MEMORY_ROUTE_TEST_KEY") == get_hermes_home().name

    def initialize(self, session_id, **kwargs):
        raise AssertionError("Settings must not initialize a provider")

    def get_tool_schemas(self):
        return []

    def get_config_schema(self):
        return [{"key": "workspace", "required": True}]
'''


@pytest.fixture
def scope_profiles(memory_homes, dashboard_client):
    for name, home in memory_homes.items():
        plugin = home / "plugins" / PLUGIN
        plugin.mkdir(parents=True)
        (plugin / "__init__.py").write_text(PROVIDER, encoding="utf-8")
        (plugin / "plugin.yaml").write_text(f"name: {PLUGIN}\ndescription: {name}\n", encoding="utf-8")
        (home / "plugins/.install-metadata.json").write_text(json.dumps({PLUGIN: {"identifier": name}}), encoding="utf-8")
        (home / ".env").write_text(f"MEMORY_ROUTE_TEST_KEY={home.name}\n", encoding="utf-8")
        (home / "config.yaml").write_text(yaml.safe_dump({
            "memory": {"provider": PLUGIN, PLUGIN: {"workspace": name if name == "default" else ""}},
            "plugins": {"enabled": [PLUGIN]},
        }), encoding="utf-8")
        (home / "memories").mkdir(exist_ok=True)
        (home / "memories/MEMORY.md").write_text(name, encoding="utf-8")
        (home / "memories/USER.md").write_text(f"user-{name}", encoding="utf-8")
    return dashboard_client, memory_homes


def _config(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))


def _status(client, profile, *, active=PLUGIN):
    response = client.get("/api/memory", params={"profile": profile})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["active"] == active
    return next(row for row in data["providers"] if row["name"] == PLUGIN)


def test_status_and_selection_use_the_requested_profile(scope_profiles):
    client, homes = scope_profiles
    before = {name: _config(home) for name, home in homes.items()}
    for name in ("default", "b", "default"):
        response = client.get("/api/memory", params={"profile": name})
        assert response.json()["builtin_files"] == {"memory": len(name), "user": len(f"user-{name}")}
        probe = _status(client, name)
        assert probe["description"] == name
        assert probe["available"] is True  # real get_secret() under the requested home
        assert probe["status"] == ("ready" if name == "default" else "needs_config")
    for name, provider, status in (("b", PLUGIN, 400), ("b", "built-in", 200), ("default", "built-in", 200), ("default", PLUGIN, 200)):
        response = client.put("/api/memory/provider", params={"profile": name}, json={"provider": provider})
        assert response.status_code == status, response.text
        if status == 200:
            before[name]["memory"]["provider"] = "" if provider == "built-in" else provider
            assert response.json() == {"ok": True, "active": before[name]["memory"]["provider"]}
        else:
            assert "needs config" in response.json()["detail"]
        assert {key: _config(home)["memory"] for key, home in homes.items()} == {key: cfg["memory"] for key, cfg in before.items()}
    for profile, status in (("../escape", 400), ("unknown", 404)):
        assert client.get("/api/memory", params={"profile": profile}).status_code == status
        assert client.put("/api/memory/provider", params={"profile": profile}, json={"provider": "built-in"}).status_code == status
    # Callers without a profile still address the launch home.
    assert client.put("/api/memory/provider", json={"provider": "built-in"}).status_code == 200
    assert _config(homes["default"])["memory"]["provider"] == "" and _config(homes["b"])["memory"] == before["b"]["memory"]
