"""The Webapp serves this host's desktop plugins to the browser's plugin loader.

Native Desktop reads ``<root>/desktop-plugins`` from its own disk. A browser tab's only disk is the
server, so ``/api/desktop-plugins/*`` answers the same three reads the loader makes (the root, its
folders, a folder's ``plugin.js`` and package marker), whole, and nothing else. The renderer half is
pinned in ``apps/desktop/src/lib/browser-desktop-plugins.test.ts``.
"""
import concurrent.futures
import json
import os

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.web_routers import desktop_plugins


def _client(monkeypatch, surface):
    for key, value in (("auth_required", False), ("bound_host", None), ("ui_surface", surface)):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


@pytest.fixture
def hermes_root(monkeypatch, tmp_path):
    """A custom Hermes root, served through one of its named profiles."""
    root = tmp_path / "hermes"
    (root / "profiles" / "work").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "work"))
    return root


@pytest.fixture
def webapp(monkeypatch, hermes_root):
    client = _client(monkeypatch, "webapp")
    yield client
    client.close()


def _install(root, name, source, marker=None):
    folder = root / "desktop-plugins" / name
    folder.mkdir(parents=True)
    (folder / "plugin.js").write_text(source, encoding="utf-8")
    if marker is not None:
        (folder / ".hermes-package.json").write_text(json.dumps(marker), encoding="utf-8")
    return folder


def test_webapp_loads_the_app_level_plugins_folder_whole(webapp, hermes_root):
    # Larger than the 512 KiB /api/fs preview cap, which would hand the loader half a module.
    source = "export default { id: 'big' }\n" + "// pad\n" * 120_000
    folder = _install(hermes_root, "big", source, marker={"package": "big-pkg"})

    root = webapp.get("/api/desktop-plugins/root").json()["root"]
    # App-level like Electron's: the root's folder, not the served profile's.
    assert root == str((hermes_root / "desktop-plugins").resolve())

    listed = webapp.get("/api/desktop-plugins/list", params={"path": root}).json()["entries"]
    assert listed == [{"name": "big", "path": str(folder.resolve()), "isDirectory": True}]

    names = {e["name"] for e in webapp.get("/api/desktop-plugins/list", params={"path": listed[0]["path"]}).json()["entries"]}
    assert names == {"plugin.js", ".hermes-package.json"}

    read = webapp.get("/api/desktop-plugins/read", params={"path": str(folder.resolve() / "plugin.js")}).json()
    assert (read["text"], read["truncated"]) == (source, False)
    marker = webapp.get("/api/desktop-plugins/read", params={"path": str(folder.resolve() / ".hermes-package.json")})
    assert json.loads(marker.json()["text"]) == {"package": "big-pkg"}


def test_webapp_reads_nothing_but_plugin_entries_and_markers(webapp, hermes_root):
    folder = _install(hermes_root, "notes", "export default {}")
    (folder / "secrets.txt").write_text("token", encoding="utf-8")
    (folder / "lib").mkdir()
    (folder / "lib" / "plugin.js").write_text("export default {}", encoding="utf-8")
    (hermes_root / ".env").write_text("OPENAI_API_KEY=sk-test", encoding="utf-8")
    (hermes_root / "desktop-plugins" / "plugin.js").write_text("loose", encoding="utf-8")
    plugins = (hermes_root / "desktop-plugins").resolve()

    for path in (
        folder.resolve() / "secrets.txt",
        folder.resolve() / "lib" / "plugin.js",
        plugins / "plugin.js",
        plugins / ".." / ".env",
        hermes_root.resolve() / ".env",
    ):
        assert webapp.get("/api/desktop-plugins/read", params={"path": str(path)}).status_code == 404, path

    for path in (folder.resolve() / "lib", plugins / ".."):
        assert webapp.get("/api/desktop-plugins/list", params={"path": str(path)}).status_code == 404, path


def test_a_linked_plugin_folder_loads_like_native_desktop(webapp, hermes_root, tmp_path):
    # `desktop-plugins/foo -> ~/src/foo` is how plugins get developed; Electron follows the link.
    checkout = tmp_path / "src" / "foo"
    checkout.mkdir(parents=True)
    (checkout / "plugin.js").write_text("export default { id: 'foo' }", encoding="utf-8")
    (checkout / "notes.txt").write_text("private", encoding="utf-8")
    plugins = hermes_root / "desktop-plugins"
    plugins.mkdir()
    try:
        (plugins / "foo").symlink_to(checkout, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks need privileges here")
    link = plugins.resolve() / "foo"

    listed = webapp.get("/api/desktop-plugins/list", params={"path": str(link)})
    assert listed.status_code == 200
    assert {e["name"] for e in listed.json()["entries"]} == {"plugin.js", "notes.txt"}
    read = webapp.get("/api/desktop-plugins/read", params={"path": str(link / "plugin.js")}).json()
    assert read["text"] == "export default { id: 'foo' }"
    # The link opens the folder, not the rest of the checkout's files.
    assert webapp.get("/api/desktop-plugins/read", params={"path": str(link / "notes.txt")}).status_code == 404


@pytest.mark.parametrize("entry", ["plugin.js", ".hermes-package.json"])
def test_a_linked_entry_file_is_never_followed(webapp, hermes_root, entry):
    # Only folders are followed: a linked entry could name `.env`, which /api/fs refuses to read.
    (hermes_root / ".env").write_text("OPENAI_API_KEY=sk-test", encoding="utf-8")
    folder = hermes_root / "desktop-plugins" / "evil"
    folder.mkdir(parents=True)
    try:
        (folder / entry).symlink_to(hermes_root / ".env")
    except OSError:
        pytest.skip("symlinks need privileges here")

    read = webapp.get("/api/desktop-plugins/read", params={"path": str(folder.resolve() / entry)})
    assert read.status_code == 404
    assert "sk-test" not in read.text


def _swap_folder_for_link(folder, hermes_root):
    secrets = hermes_root / "pairing"
    secrets.mkdir()
    (secrets / "plugin.js").write_text("sk-test", encoding="utf-8")
    folder.rename(folder.with_name("evil.old"))
    folder.symlink_to(secrets, target_is_directory=True)


def _swap_entry_for_link(folder, hermes_root):
    (folder / "plugin.js").unlink()
    (folder / "plugin.js").symlink_to(hermes_root / ".env")


def _swap_entry_for_fifo(folder, hermes_root):
    (folder / "plugin.js").unlink()
    os.mkfifo(folder / "plugin.js")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX links and FIFOs")
@pytest.mark.parametrize("swap", [_swap_folder_for_link, _swap_entry_for_link, _swap_entry_for_fifo])
@pytest.mark.parametrize("nofollow", [True, False])
def test_a_swap_between_the_checks_and_the_open_fails_closed(webapp, hermes_root, monkeypatch, swap, nofollow):
    # The checks pass on a real plugin; the swap lands at the open. Without O_NOFOLLOW (Windows)
    # the open follows a final link too, so only the identity check after it may stop the read.
    (hermes_root / ".env").write_text("OPENAI_API_KEY=sk-test", encoding="utf-8")
    folder = _install(hermes_root, "evil", "export default {}").resolve()
    if not nofollow:
        monkeypatch.setattr(desktop_plugins, "_OPEN_FLAGS", desktop_plugins._OPEN_FLAGS & ~getattr(os, "O_NOFOLLOW", 0))
    real_open, pending = os.open, [True]

    def open_after_swap(path, flags, *args, **kwargs):
        if pending and str(path) == str(folder / "plugin.js"):
            pending.clear()
            swap(folder, hermes_root)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(desktop_plugins.os, "open", open_after_swap)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        read = pool.submit(webapp.get, "/api/desktop-plugins/read", params={"path": str(folder / "plugin.js")})
        try:
            response = read.result(timeout=10)
        finally:
            if not read.done() and (folder / "plugin.js").is_fifo():
                os.close(real_open(folder / "plugin.js", os.O_WRONLY | os.O_NONBLOCK))
    assert not pending
    assert response.status_code == 404
    assert "sk-test" not in response.text


@pytest.mark.parametrize("surface", ["gateway", "dashboard", "serve"])
def test_only_the_webapp_serves_desktop_plugins(monkeypatch, hermes_root, surface):
    _install(hermes_root, "notes", "export default {}")
    client = _client(monkeypatch, surface)
    try:
        assert client.get("/api/desktop-plugins/root").status_code == 404
    finally:
        client.close()
