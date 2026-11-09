"""The dashboard's channel edit routes resolve the platform against the TARGET profile's platforms, the same
catalog ``GET /api/messaging/platforms?profile=`` shows: a platform plugin installed only in a secondary
profile is editable there (with its own value rules) and still unknown for the dashboard's own profile.
Real plugin directories, real loader, real config/.env files under a temp home.
"""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient

PLATFORM_PLUGIN = '''
def register(ctx):
    ctx.register_platform(
        name="workchat", label="Work Chat", adapter_factory=lambda cfg: None, check_fn=lambda: True,
        required_env=["WORKCHAT_TOKEN"],
        env_value_rules={"WORKCHAT_TOKEN": (lambda v: v.startswith("wc-"), "WORKCHAT_TOKEN starts with wc-")})
'''


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod

    root = tmp_path / "hermes_home"
    work = root / "profiles" / "work"
    plugin_dir = work / "plugins" / "workchat"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "workchat", "kind": "platform", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(PLATFORM_PLUGIN, encoding="utf-8")
    (work / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["workchat"]}}), encoding="utf-8")
    (root / "config.yaml").write_text(yaml.safe_dump({"model": {"default": "gpt-5"}}), encoding="utf-8")
    for home in (root, work):
        (home / ".env").write_text("", encoding="utf-8")
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


def test_a_gateway_block_for_a_profile_plugin_stays_in_that_profile(client, homes):
    """A ``gateway.<plugin id>`` block next to a token-free built-in must not register the profile's
    plugin process-wide while the dashboard reads the built-in's status."""
    from gateway.config import Platform

    _, work = homes
    (work / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["workchat"]},
        "gateway": {"workchat": {"enabled": True}},
        "platforms": {"webhook": {"enabled": True}},
    }), encoding="utf-8")
    leaked_before = "workchat" in Platform._value2member_map_  # an earlier test may have registered it

    scoped = client.get("/api/messaging/platforms", params={"profile": "work"}).json()["platforms"]
    root_ids = {p["id"] for p in client.get("/api/messaging/platforms").json()["platforms"]}

    assert next(p for p in scoped if p["id"] == "webhook")["configured"] is True
    assert ("workchat" in Platform._value2member_map_) is leaked_before
    assert "workchat" not in root_ids


def test_a_managed_gateway_block_for_a_profile_plugin_stays_in_that_profile(client, homes, tmp_path, monkeypatch):
    """The administrator's config goes through the same filtered steps: a managed ``gateway.<plugin id>``
    block must not register the profile's plugin process-wide either."""
    from gateway.config import Platform
    from hermes_cli import managed_scope

    _, work = homes
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump({"gateway": {"workchat": {"enabled": True}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    (work / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["workchat"]}, "platforms": {"webhook": {"enabled": True}},
    }), encoding="utf-8")
    leaked_before = "workchat" in Platform._value2member_map_  # an earlier test may have registered it
    try:
        scoped = client.get("/api/messaging/platforms", params={"profile": "work"}).json()["platforms"]
    finally:
        managed_scope.invalidate_managed_cache()

    assert next(p for p in scoped if p["id"] == "webhook")["configured"] is True
    assert ("workchat" in Platform._value2member_map_) is leaked_before


def test_a_platform_installed_only_in_the_target_profile_is_editable_there(client, homes):
    root, work = homes
    listed = client.get("/api/messaging/platforms?profile=work").json()
    assert "workchat" in {p["id"] for p in listed["platforms"]}

    # The target profile's plugin rules still guard its values.
    resp = client.put("/api/messaging/platforms/workchat?profile=work", json={"env": {"WORKCHAT_TOKEN": "nope"}})
    assert resp.status_code == 400 and "starts with wc-" in resp.text
    resp = client.put("/api/messaging/platforms/workchat?profile=work", json={"env": {"WORKCHAT_TOKEN": "wc-1"}})
    assert resp.status_code == 200, resp.text
    assert "WORKCHAT_TOKEN=wc-1" in (work / ".env").read_text(encoding="utf-8")
    assert client.post("/api/messaging/platforms/workchat/test?profile=work").status_code == 200

    # The dashboard's own profile does not have the plugin: still unknown there, .env untouched.
    resp = client.put("/api/messaging/platforms/workchat", json={"env": {"WORKCHAT_TOKEN": "wc-1"}})
    assert resp.status_code == 404 and "Unknown messaging platform" in resp.text
    assert client.post("/api/messaging/platforms/workchat/test").status_code == 404
    assert "WORKCHAT_TOKEN" not in (root / ".env").read_text(encoding="utf-8")
