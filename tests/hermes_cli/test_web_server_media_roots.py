"""Unified media-roots policy (M3 / D3): allow/deny contract tests.

Every media-serving dashboard endpoint must enforce ONE policy —
``media.roots`` from config.yaml (hermes_cli/media_roots.py), defaulting to the
gateway workspace plus ~/.hermes/{images,screenshots,cache}:

* ``GET /api/media``                    (images)
* ``GET /api/files/stream``             (audio/video)
* ``GET /api/files/download`` media branch (Sec-Fetch-Dest: audio|video)
* ``GET /api/fs/read-data-url``         (media-extension files)

Behavior contract, not change-detectors: a media file inside a root is served;
the same file outside every root gets 403 ``Path outside media roots``; a
configured extra root serves files under it; a non-media file (or a non-media
download/preview branch) is NOT gated by this policy.
"""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli import media_roots as media_roots_mod


pytest.importorskip("starlette.testclient")


@pytest.fixture
def client():
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous_auth_required


@pytest.fixture
def roots_config(monkeypatch):
    """Point the policy at an explicit config dict and return a setter.

    ``load_config()`` is cached on the real config file's mtime, so tests
    override the module-level loader seam instead of writing config.yaml.
    Passing None as the value restores the default policy (unconfigured).
    """
    state = {"media_roots": None}

    def _fake_load_config():
        if state["media_roots"] is None:
            return {}
        return {"media": {"roots": state["media_roots"]}}

    monkeypatch.setattr(media_roots_mod, "_load_config", _fake_load_config)
    return state.__setitem__


@pytest.fixture
def media_tree(tmp_path):
    """A served tree (workspace root) and an unserved outside tree."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    for d in (workspace, outside):
        d.mkdir()
    return workspace, outside


# ── Policy module unit contract ─────────────────────────────────────────────


def test_default_roots_cover_workspace_and_hermes_media_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for sub in ("images", "screenshots", "cache"):
        (tmp_path / "home" / sub).mkdir(parents=True)

    monkeypatch.chdir(workspace)

    resolved = media_roots_mod.media_roots({})

    names = {p.name for p in resolved}
    assert workspace in resolved
    assert {"images", "screenshots", "cache"} <= names


def test_configured_roots_replace_and_expand_tilde(monkeypatch, tmp_path):
    srv = tmp_path / "srv"
    srv.mkdir()
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_config = {"media": {"roots": [str(srv), "~/gallery"]}}

    resolved = media_roots_mod.media_roots(fake_config)

    assert resolved == [srv, gallery]


def test_relative_or_junk_roots_are_dropped(tmp_path):
    ok = tmp_path / "ok"
    ok.mkdir()
    fake_config = {"media": {"roots": ["relative/path", 17, "", str(ok)]}}

    assert media_roots_mod.media_roots(fake_config) == [ok]


def test_nonexistent_configured_root_is_dropped(tmp_path):
    ok = tmp_path / "ok"
    ok.mkdir()

    fake_config = {"media": {"roots": [str(tmp_path / "missing"), str(ok)]}}

    assert media_roots_mod.media_roots(fake_config) == [ok]


def test_empty_roots_list_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    assert media_roots_mod.media_roots({"media": {"roots": []}}) == (
        media_roots_mod.media_roots({})
    )


def test_containment_rejects_sibling_prefix_paths(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "a.png").write_bytes(b"x")
    sibling = tmp_path / "media-evil"
    sibling.mkdir()
    evil = sibling / "a.png"
    evil.write_bytes(b"x")
    config = {"media": {"roots": [str(root)]}}

    assert media_roots_mod.path_in_media_roots(root / "a.png", config) is True
    assert media_roots_mod.path_in_media_roots(evil, config) is False


def test_symlink_escaping_root_is_denied(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"x")
    link = root / "escape.png"
    link.symlink_to(secret)

    assert media_roots_mod.path_in_media_roots(link) is False


# ── Endpoint allow/deny (the three formerly-diverging surfaces) ─────────────


def test_api_media_serves_inside_and_denies_outside(client, roots_config, media_tree):
    workspace, outside = media_tree
    inside = workspace / "chart.png"
    inside.write_bytes(b"png-inside")
    denied = outside / "secret.png"
    denied.write_bytes(b"png-outside")
    roots_config("media_roots", [str(workspace)])

    ok = client.get("/api/media", params={"path": str(inside)})
    assert ok.status_code == 200
    assert "data:image/png;base64," in ok.json()["data_url"]

    no = client.get("/api/media", params={"path": str(denied)})
    assert no.status_code == 403
    assert no.json()["detail"] == "Path outside media roots"


def test_api_media_configured_extra_root_allows_outside_default_tree(
    client, roots_config, tmp_path
):
    # A file under neither the workspace nor ~/.hermes, but under an
    # operator-configured root, is served — that is the point of the key.
    extra = tmp_path / "exports"
    extra.mkdir()
    media = extra / "render.png"
    media.write_bytes(b"png")
    roots_config("media_roots", [str(extra)])

    assert client.get("/api/media", params={"path": str(media)}).status_code == 200


def test_api_media_default_denies_arbitrary_disk_image(client, roots_config, tmp_path):
    # Default policy (no configured roots): workspace + ~/.hermes media dirs
    # only. /tmp is not a media root — regression for the old unconfined
    # surfaces.
    stray = tmp_path / "stray.png"
    stray.write_bytes(b"png")
    roots_config("media_roots", None)

    assert client.get("/api/media", params={"path": str(stray)}).status_code == 403


def test_files_stream_serves_inside_and_denies_outside(client, roots_config, media_tree):
    workspace, outside = media_tree
    inside = workspace / "demo.mp4"
    inside.write_bytes(b"aaaaa")
    denied = outside / "demo.mp4"
    denied.write_bytes(b"aaaaa")
    roots_config("media_roots", [str(workspace)])

    ok = client.get("/api/files/stream", params={"path": str(inside)})
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "video/mp4"

    no = client.get("/api/files/stream", params={"path": str(denied)})
    assert no.status_code == 403
    assert no.json()["detail"] == "Path outside media roots"


def test_files_download_media_subresource_is_gated_plain_download_is_not(
    client, roots_config, media_tree
):
    workspace, outside = media_tree
    denied = outside / "clip.mp4"
    denied.write_bytes(b"aaaaa")
    roots_config("media_roots", [str(workspace)])

    media_branch = client.get(
        "/api/files/download",
        params={"path": str(denied)},
        headers={"Sec-Fetch-Dest": "video"},
    )
    assert media_branch.status_code == 403

    save_as = client.get("/api/files/download", params={"path": str(denied)})
    assert save_as.status_code == 200
    assert save_as.headers["content-disposition"].startswith("attachment;")


def test_read_data_url_gates_media_ext_allows_text_ext(client, roots_config, media_tree):
    workspace, outside = media_tree
    denied_image = outside / "pic.png"
    denied_image.write_bytes(b"png")
    denied_av = outside / "song.ogg"
    denied_av.write_bytes(b"ogg")
    text_outside = outside / "notes.txt"
    text_outside.write_text("hello")
    roots_config("media_roots", [str(workspace)])

    for denied in (denied_image, denied_av):
        no = client.get("/api/fs/read-data-url", params={"path": str(denied)})
        assert no.status_code == 403, denied
        assert no.json()["detail"] == "Path outside media roots"

    # Non-media previews stay ungated (FS browser/editor behavior unchanged).
    ok_text = client.get("/api/fs/read-data-url", params={"path": str(text_outside)})
    assert ok_text.status_code == 200
    assert "data:text/plain;base64," in ok_text.json()["dataUrl"]


def test_read_data_url_serves_media_inside_roots(client, roots_config, media_tree):
    workspace, _outside = media_tree
    inside_image = workspace / "pic.png"
    inside_image.write_bytes(b"png")
    roots_config("media_roots", [str(workspace)])

    ok = client.get("/api/fs/read-data-url", params={"path": str(inside_image)})
    assert ok.status_code == 200
    assert "data:image/png;base64," in ok.json()["dataUrl"]


def test_all_three_surfaces_share_one_configured_root_set(
    client, roots_config, media_tree, tmp_path
):
    """One config change moves all three formerly-diverging policies together."""
    workspace, _outside = media_tree
    extra = tmp_path / "shared-root"
    extra.mkdir()
    shared = extra / "clip.webm"
    shared.write_bytes(b"webm")
    shared_img = extra / "shot.png"
    shared_img.write_bytes(b"png")
    roots_config("media_roots", [str(extra)])

    assert client.get("/api/files/stream", params={"path": str(shared)}).status_code == 200
    assert client.get("/api/media", params={"path": str(shared_img)}).status_code == 200
    assert (
        client.get("/api/fs/read-data-url", params={"path": str(shared_img)}).status_code == 200
    )
    # And the workspace-only file is now denied on all three — one policy.
    stray = workspace / "now-outside.png"
    stray.write_bytes(b"png")
    workspace_av = workspace / "v.mp4"
    workspace_av.write_bytes(b"mp4")
    assert client.get("/api/media", params={"path": str(stray)}).status_code == 403
    assert (
        client.get("/api/files/stream", params={"path": str(workspace_av)}).status_code == 403
    )
    assert (
        client.get("/api/fs/read-data-url", params={"path": str(stray)}).status_code == 403
    )
