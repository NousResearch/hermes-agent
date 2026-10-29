"""On a headless host the Webapp itself copies agent+desktop packages' desktop halves into
``<root>/desktop-plugins``, the copy native Desktop makes in Electron
(``reconcileUnifiedDesktopHalves``) and nothing else makes when Desktop never runs there.
"""
import json
import os

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def hermes_root(monkeypatch, tmp_path):
    root = tmp_path / "hermes"
    (root / "profiles" / "work").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "work"))
    return root


@pytest.fixture
def webapp(monkeypatch, hermes_root):
    for key, value in (("auth_required", False), ("bound_host", None), ("ui_surface", "webapp")):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    yield client
    client.close()


def _package(home, name, source):
    desktop = home / "plugins" / name / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "plugin.js").write_text(source, encoding="utf-8")
    return desktop


def _loaded(webapp):
    """What the browser's loader sees: each folder's plugin.js text and package marker."""
    root = webapp.get("/api/desktop-plugins/root").json()["root"]
    seen = {}
    for folder in webapp.get("/api/desktop-plugins/list", params={"path": root}).json()["entries"]:
        files = {e["name"]: e["path"] for e in webapp.get("/api/desktop-plugins/list", params={"path": folder["path"]}).json()["entries"]}
        read = lambda name: webapp.get("/api/desktop-plugins/read", params={"path": files[name]}).json()["text"]
        marker = json.loads(read(".hermes-package.json")) if ".hermes-package.json" in files else None
        seen[folder["name"]] = (read("plugin.js"), marker)
    return seen


def test_an_installed_package_brings_its_desktop_half_until_it_is_uninstalled(webapp, hermes_root):
    mine = hermes_root / "desktop-plugins" / "notes"
    mine.mkdir(parents=True)
    (mine / "plugin.js").write_text("mine", encoding="utf-8")
    source = _package(hermes_root / "profiles" / "work", "radar", "export default { v: 1 }")

    text, marker = _loaded(webapp)["radar"]
    assert text == "export default { v: 1 }"
    # The marker is what pairs the half with its agent row and keeps it off until enabled.
    assert (marker["package"], marker["source"]) == ("radar", str(source))

    # A corrupt timestamp in one marker is stale metadata, never an error that hides every plugin.
    marker_path = hermes_root / "desktop-plugins" / "radar" / ".hermes-package.json"
    marker_path.write_text(json.dumps({**marker, "sourceMtimeMs": None}), encoding="utf-8")
    loaded = _loaded(webapp)
    assert loaded["notes"] == ("mine", None)
    assert isinstance(loaded["radar"][1]["sourceMtimeMs"], (int, float))

    (source / "plugin.js").write_text("export default { v: 2 }", encoding="utf-8")
    later = os.stat(source / "plugin.js").st_mtime + 5
    os.utime(source / "plugin.js", (later, later))
    touched = webapp.post("/api/desktop-plugins/reconcile").json()["touched"]
    assert touched == [str((hermes_root / "desktop-plugins" / "radar").resolve())]
    assert _loaded(webapp)["radar"][0] == "export default { v: 2 }"

    (source / "plugin.js").unlink()
    webapp.post("/api/desktop-plugins/reconcile")
    # The uninstalled package's copy is pruned; the user's own plugin is left as it was.
    assert _loaded(webapp) == {"notes": ("mine", None)}


def test_a_users_own_plugin_is_never_overwritten_and_the_default_home_wins(webapp, hermes_root):
    mine = hermes_root / "desktop-plugins" / "notes"
    mine.mkdir(parents=True)
    (mine / "plugin.js").write_text("mine", encoding="utf-8")
    _package(hermes_root, "notes", "a package that happens to share the name")
    _package(hermes_root, "radar", "from the default home")
    _package(hermes_root / "profiles" / "work", "radar", "from a profile")
    # A hand-made link to a package's own half (the manual workaround) still loads exactly once.
    meshy = _package(hermes_root, "meshy", "meshy half")
    (hermes_root / "desktop-plugins" / "meshy").symlink_to(meshy, target_is_directory=True)

    loaded = _loaded(webapp)

    assert sorted(loaded) == ["meshy", "notes", "radar"]
    assert loaded["meshy"][0] == "meshy half"
    assert loaded["notes"] == ("mine", None)
    assert loaded["radar"][0] == "from the default home"
    (hermes_root / "plugins" / "notes" / "desktop" / "plugin.js").unlink()
    webapp.post("/api/desktop-plugins/reconcile")
    assert (mine / "plugin.js").read_text(encoding="utf-8") == "mine"


def test_a_root_layout_package_brings_its_desktop_half_as_the_installers_publish_it(webapp, hermes_root, tmp_path):
    # plugin.yaml + __init__.py + a ROOT plugin.js: the install probe reports both halves, so the
    # tab installs the agent half alone and counts on this pass for the desktop one.
    package = hermes_root / "plugins" / "radar"
    (package / ".git").mkdir(parents=True)
    (package / ".git" / "config").write_text('[remote "origin"]\n\turl = https://github.com/example/radar\n')
    (package / "plugin.yaml").write_text("name: radar\n", encoding="utf-8")
    (package / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (package / "plugin.js").write_text("export default { id: 'radar' }", encoding="utf-8")
    (package / "panel.js").write_text("export const panel = 1", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "x.pyc").write_bytes(b"pyc")
    # A root plugin.js the user could reach through a link would publish anything on the host.
    linked = hermes_root / "plugins" / "leaky"
    linked.mkdir()
    (linked / "plugin.js").write_text("export default {}", encoding="utf-8")
    (tmp_path / "secret").write_text("x", encoding="utf-8")
    (linked / "secret").symlink_to(tmp_path / "secret")

    loaded = _loaded(webapp)

    text, marker = loaded["radar"]
    assert text == "export default { id: 'radar' }"
    assert (marker["package"], marker["source"], marker["repo"]) == ("radar", str(package), "https://github.com/example/radar")
    copy = hermes_root / "desktop-plugins" / "radar"
    # The repository root is the plugin folder, its assets with it; the clone's .git and the agent half's bytecode are not.
    assert (copy / "panel.js").read_text(encoding="utf-8") == "export const panel = 1"
    assert not (copy / ".git").exists() and not (copy / "__pycache__").exists()
    assert "leaky" not in loaded

    (package / "plugin.js").unlink()
    webapp.post("/api/desktop-plugins/reconcile")
    assert "radar" not in _loaded(webapp)
