"""Settings › Subagents routes: ``delegation.backend`` and ``delegation.backends.*`` are read, validated and
written in the TARGET profile (``?profile=``). Real plugin directory registering real delegation providers, the
real loader and real config files under a temp home.
"""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient

ROUTES_PLUGIN = '''
from agent.delegation_provider import DelegationProvider


class _Routes(DelegationProvider):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def route_task(self, task, config):
        return dict(config)


def register(ctx):
    ctx.register_delegation_provider(_Routes("model_map"))
    ctx.register_delegation_provider(_Routes("profile_routes"))
'''

SECRET = "sk-route-secret-0123456789abcdef"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod

    root = tmp_path / "hermes_home"
    work = root / "profiles" / "work"
    plugin_dir = work / "plugins" / "routes"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "routes", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(ROUTES_PLUGIN, encoding="utf-8")
    (work / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["routes"]},
        "delegation": {"backend": "model_map", "backends": {"model_map": {"options": [
            {"name": "fast", "description": "Quick", "provider": "openrouter", "model": "m-fast", "api_key": SECRET,
             "request_overrides": {"extra_body": {"seed": 7}}},
            {"name": "deep", "description": "Careful", "provider": "anthropic", "model": "m-deep"},
        ], "default": "deep"}}},
    }), encoding="utf-8")
    # The launch profile saved a plugin backend it does not have: configured, but nothing can run it.
    (root / "config.yaml").write_text(yaml.safe_dump({"delegation": {"backend": "model_map"}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    from agent import secret_scope
    from tui_gateway import launch_profile_policy as lpp
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(lpp, "_snapshot", None)
    plugins_mod._reset_plugin_managers_for_tests()
    yield root, work
    plugins_mod._reset_plugin_managers_for_tests()


@pytest.fixture
def client(homes):
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _delegation(home) -> dict:
    return (yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}).get("delegation") or {}


def test_routing_reads_the_target_profile_and_never_returns_a_route_key(client):
    work = client.get("/api/delegation/routing?profile=work").json()
    assert {b["name"] for b in work["backends"]} == {"default", "model_map", "profile_routes"}
    assert (work["configured"], work["blocked_reason"]) == ("model_map", None)
    assert [o["name"] for o in work["model_map"]["options"]] == ["fast", "deep"]
    assert work["model_map"]["default"] == "deep"
    assert SECRET not in str(work) and work["model_map"]["options"][0]["api_key"]
    assert {"default", "work"} <= set(work["profiles"])

    # Core fails closed on a saved backend no plugin provides: the page must say why, not show Default.
    launch = client.get("/api/delegation/routing").json()
    assert [b["name"] for b in launch["backends"]] == ["default"]
    assert launch["configured"] == "model_map" and "not installed" in launch["blocked_reason"]


def test_a_backend_can_only_be_chosen_where_its_plugin_is_registered(client, homes):
    root, work = homes
    resp = client.put("/api/delegation/backend?profile=work", json={"backend": "profile_routes"})
    assert resp.status_code == 200, resp.text
    assert _delegation(work)["backend"] == "profile_routes"

    before = (root / "config.yaml").read_text(encoding="utf-8")
    resp = client.put("/api/delegation/backend", json={"backend": "profile_routes"})
    assert resp.status_code == 400
    assert "not installed" in resp.json()["detail"]
    assert (root / "config.yaml").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(("path", "body"), [
    ("/api/delegation/backend", {"backend": "default"}),
    ("/api/delegation/backends/model_map", {"options": []}),
    ("/api/delegation/backends/profile_routes", {"profiles": []}),
])
def test_a_managed_install_refuses_every_routing_write_with_the_reason(client, homes, monkeypatch, path, body):
    """config.yaml of a package-manager-managed install is not ours: each write is a 409 naming the manager,
    never a silent ``ok`` over an untouched file."""
    _root, work = homes
    before = (work / "config.yaml").read_text(encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    resp = client.put(f"{path}?profile=work", json=body)
    assert resp.status_code == 409, resp.text
    assert "managed by nixos" in resp.json()["detail"] and "\n" not in resp.json()["detail"]
    assert (work / "config.yaml").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(("options", "problem"), [
    ([{"name": "a", "description": "x", "provider": "p", "model": "m"},
      {"name": "a", "description": "y", "provider": "p", "model": "m"}], "unique"),
    ([{"name": "a", "provider": "p", "model": "m"}], "description"),
    ([{"name": "a", "description": "x", "model": "m"}], "provider"),
    ([{"name": "a", "description": "x", "provider": "p", "model": "m", "service_tier": "warp"}], "service tier"),
    ([{"name": "a", "description": "x", "provider": "p", "model": "m", "reasoning_effort": "huge"}], "reasoning"),
])
def test_invalid_model_routes_are_refused_before_anything_is_written(client, homes, options, problem):
    _root, work = homes
    before = (work / "config.yaml").read_text(encoding="utf-8")
    resp = client.put("/api/delegation/backends/model_map?profile=work", json={"options": options})
    assert resp.status_code == 400
    assert problem in resp.json()["detail"]
    assert (work / "config.yaml").read_text(encoding="utf-8") == before


def test_saving_model_routes_replaces_the_block_and_keeps_the_stored_key(client, homes):
    root, work = homes
    shown = client.get("/api/delegation/routing?profile=work").json()["model_map"]["options"][0]
    # Rename and edit "fast" as the renderer would (preview key and untouched fields sent back, the old name in
    # original_name), delete "deep", add "local".
    body = {"options": [{**shown, "name": "speedy", "original_name": "fast", "model": "m-faster",
                         "reasoning_effort": "low"},
                        {"name": "local", "description": "Private", "base_url": "http://127.0.0.1:8080/v1",
                         "model": "qwen", "service_tier": "fast"}],
            "default": "local"}
    resp = client.put("/api/delegation/backends/model_map?profile=work", json=body)
    assert resp.status_code == 200, resp.text
    assert SECRET not in resp.text

    block = _delegation(work)["backends"]["model_map"]
    assert block["default"] == "local"
    assert [o["name"] for o in block["options"]] == ["speedy", "local"]
    assert block["options"][0] == {"name": "speedy", "description": "Quick", "provider": "openrouter",
                                   "model": "m-faster", "reasoning_effort": "low", "api_key": SECRET,
                                   "request_overrides": {"extra_body": {"seed": 7}}}
    assert "backends" not in _delegation(root)


def test_profile_routes_only_name_existing_profiles(client, homes):
    _root, work = homes
    before = (work / "config.yaml").read_text(encoding="utf-8")
    resp = client.put("/api/delegation/backends/profile_routes?profile=work",
                      json={"profiles": [{"name": "ghost"}]})
    assert resp.status_code == 400 and "does not exist" in resp.json()["detail"]
    assert (work / "config.yaml").read_text(encoding="utf-8") == before

    resp = client.put("/api/delegation/backends/profile_routes?profile=work",
                      json={"profiles": [{"name": "default", "description": "  The main one  "}], "default": "default"})
    assert resp.status_code == 200, resp.text
    assert _delegation(work)["backends"]["profile_routes"] == {
        "profiles": [{"name": "default", "description": "The main one"}], "default": "default"}
