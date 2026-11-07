"""The Webapp is installable as an app: its page links the install manifest, and the manifest and
the icons it names load without a session, since browsers fetch them without cookies. Only the
Webapp's page links it; the dashboard is not installable."""
import json
import shutil
from pathlib import Path
from urllib.parse import urljoin

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

DESKTOP_PUBLIC = Path(__file__).resolve().parents[2] / "apps" / "desktop" / "public"
HTML = "<html><head></head><body>spa shell</body></html>"


def _spa(monkeypatch, dist: Path, *, surface) -> TestClient:
    from hermes_cli.web_server_dashboard import mount_spa

    (dist / "index.html").write_text(HTML)
    app = FastAPI()
    if surface is not None:
        app.state.ui_surface = surface
    app.state.web_dist = dist
    monkeypatch.setattr(web_server, "app", app)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(app)
    return TestClient(app)


@pytest.mark.parametrize("prefix", ["", "/hermes"])
def test_the_webapp_page_links_its_manifest_under_the_proxy_prefix(monkeypatch, tmp_path, prefix):
    page = _spa(monkeypatch, tmp_path, surface="webapp").get("/", headers={"X-Forwarded-Prefix": prefix})

    assert f'<link rel="manifest" href="{prefix}/manifest.webmanifest" />' in page.text


def test_the_dashboard_page_is_not_installable(monkeypatch, tmp_path):
    assert 'rel="manifest"' not in _spa(monkeypatch, tmp_path, surface=None).get("/").text


@pytest.fixture
def gated_webapp(monkeypatch, tmp_path):
    """The gateway's Webapp behind the sign-in gate, serving the shipped manifest and its icons."""
    dist = tmp_path / "dist-webapp"
    dist.mkdir()
    (dist / "index.html").write_text(HTML, encoding="utf-8")
    manifest = json.loads((DESKTOP_PUBLIC / "manifest.webmanifest").read_text(encoding="utf-8"))
    for name in ["manifest.webmanifest", *(icon["src"] for icon in manifest["icons"])]:
        shutil.copyfile(DESKTOP_PUBLIC / name, dist / name)
    (dist / "hermes.png").write_bytes(b"\x89PNG private bytes")
    clear_providers()
    register_provider(StubAuthProvider())
    for key, value in (("bound_host", "0.0.0.0"), ("bound_port", 9119), ("auth_required", True),
                       ("trusted_public_hosts", frozenset()), ("ui_surface", "webapp"),
                       ("listener_public_url", ""), ("web_dist", dist)):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    yield TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    clear_providers()


def test_the_manifest_and_every_icon_it_names_load_without_a_session(gated_webapp):
    response = gated_webapp.get("/manifest.webmanifest", follow_redirects=False)
    assert response.status_code == 200, response.headers.get("location")
    manifest = json.loads(response.content)

    # Relative to the manifest, so an install from behind a proxy prefix stays inside it.
    base = "https://example.test/hermes/manifest.webmanifest"
    assert urljoin(base, manifest["start_url"]) == urljoin(base, manifest["scope"]) == "https://example.test/hermes/"
    for icon in manifest["icons"]:
        response = gated_webapp.get(urljoin("/manifest.webmanifest", icon["src"]), follow_redirects=False)
        assert response.status_code == 200, icon["src"]
        assert response.content.startswith(b"\x89PNG"), icon["src"]


@pytest.mark.parametrize("path", ["/hermes.png", "/manifest.webmanifest.map", "/pwa-192.png/"])
def test_other_files_still_require_sign_in(gated_webapp, path):
    response = gated_webapp.get(path, follow_redirects=False)

    assert response.status_code == 302
    assert "/login" in response.headers["location"]


def test_a_missing_manifest_is_a_404_not_the_page(gated_webapp):
    (web_server.app.state.web_dist / "manifest.webmanifest").unlink()

    response = gated_webapp.get("/manifest.webmanifest", follow_redirects=False)

    assert response.status_code == 404
    assert "html" not in response.headers.get("content-type", "")
