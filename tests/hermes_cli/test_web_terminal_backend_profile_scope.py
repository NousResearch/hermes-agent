"""``PUT /api/tools/terminal/backend`` validates the backend against the TARGET profile's plugin
backends, the same list ``GET /api/tools/terminal/backends?profile=`` shows: a backend plugin installed
only in a secondary profile (the ``singularity`` plugin after the backend left core) is selectable
there and still unknown for the dashboard's own profile. Real plugin directories, real loader, real
config files under a temp home.
"""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient

BACKEND_PLUGIN = '''
from agent.terminal_env_provider import TerminalEnvironmentProvider

class Box(TerminalEnvironmentProvider):
    @property
    def name(self):
        return "workbox"
    def is_available(self):
        return True
    def create_environment(self, **kw):
        raise NotImplementedError

def register(ctx):
    ctx.register_terminal_environment_provider(Box())
'''


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod

    root = tmp_path / "hermes_home"
    work = root / "profiles" / "work"
    plugin_dir = work / "plugins" / "workbox"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "workbox", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(BACKEND_PLUGIN, encoding="utf-8")
    (work / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["workbox"]}}), encoding="utf-8")
    (root / "config.yaml").write_text(yaml.safe_dump({"terminal": {"backend": "local"}}), encoding="utf-8")
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


def _backend(home):
    return (yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")).get("terminal") or {}).get("backend")


def test_a_backend_installed_only_in_the_target_profile_is_selectable_there(client, homes):
    root, work = homes
    listed = client.get("/api/tools/terminal/backends?profile=work").json()
    assert "workbox" in {row["name"] for row in listed["backends"]}

    resp = client.put("/api/tools/terminal/backend?profile=work", json={"backend": "workbox"})
    assert resp.status_code == 200, resp.text
    assert _backend(work) == "workbox"

    # The dashboard's own profile does not have the plugin: still unknown there, config untouched.
    resp = client.put("/api/tools/terminal/backend", json={"backend": "workbox"})
    assert resp.status_code == 400
    assert "Unknown terminal backend" in resp.text
    assert _backend(root) == "local"
