"""The dashboard's toolset routes validate the toolset name against the TARGET profile's toolsets, the same
list ``GET /api/tools/toolsets?profile=`` shows: a plugin toolset installed only in a secondary profile
(spotify, homeassistant, ...) is configurable there and still unknown for the dashboard's own profile.
Real plugin directories, real loader, real config files under a temp home.
"""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient

TOOLSET_PLUGIN = '''
def register(ctx):
    ctx.register_tool(
        name="work_ping", toolset="worktools",
        schema={"name": "work_ping", "description": "ping", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: "pong")
'''


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod

    root = tmp_path / "hermes_home"
    work = root / "profiles" / "work"
    plugin_dir = work / "plugins" / "worktools"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "worktools", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(TOOLSET_PLUGIN, encoding="utf-8")
    (work / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["worktools"]}}), encoding="utf-8")
    (root / "config.yaml").write_text(yaml.safe_dump({"model": {"default": "gpt-5"}}), encoding="utf-8")
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


def _config(home) -> str:
    return (home / "config.yaml").read_text(encoding="utf-8")


def test_a_toolset_installed_only_in_the_target_profile_can_be_enabled_there(client, homes):
    root, work = homes
    listed = client.get("/api/tools/toolsets?profile=work").json()
    assert "worktools" in {row["name"] for row in listed}

    resp = client.put("/api/tools/toolsets/worktools?profile=work", json={"enabled": True})
    assert resp.status_code == 200, resp.text
    cli_toolsets = (yaml.safe_load(_config(work)).get("platform_toolsets") or {}).get("cli") or []
    assert "worktools" in cli_toolsets

    # The dashboard's own profile does not have the plugin: still unknown there, config untouched.
    before = _config(root)
    resp = client.put("/api/tools/toolsets/worktools", json={"enabled": True})
    assert resp.status_code == 400
    assert "Unknown toolset" in resp.text
    assert _config(root) == before


@pytest.mark.parametrize(("method", "path", "body", "scoped_answer"), [
    ("get", "/api/tools/toolsets/worktools/config", None, None),
    ("put", "/api/tools/toolsets/worktools/env", {"env": {}}, None),
    ("put", "/api/tools/toolsets/worktools/provider", {"provider": "x"}, "no configurable category"),
    ("post", "/api/tools/toolsets/worktools/post-setup", {"key": "nope"}, "Unknown post-setup key"),
])
def test_every_toolset_route_resolves_the_name_in_the_target_profile(client, method, path, body, scoped_answer):
    kwargs = {"json": body} if body is not None else {}
    resp = client.request(method.upper(), f"{path}?profile=work", **kwargs)
    assert "Unknown toolset" not in resp.text
    if scoped_answer is None:
        assert resp.status_code == 200, resp.text
    else:
        assert resp.status_code == 400 and scoped_answer in resp.text

    resp = client.request(method.upper(), path, **kwargs)
    assert resp.status_code == 400 and "Unknown toolset" in resp.text
