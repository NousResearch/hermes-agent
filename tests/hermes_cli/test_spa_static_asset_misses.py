"""Asset requests never get the SPA shell. The sign-in page's brand fonts load on the Webapp surface
too, whose bundle carries no ``fonts/`` dir, and a missing file under a static asset prefix is a
404 on both surfaces instead of ``index.html`` served as 200 ``text/html``."""
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HTML = "<html><head></head><body>spa shell</body></html>"
WEB_PUBLIC = Path(__file__).resolve().parents[2] / "web" / "public"


def _spa(monkeypatch, dist: Path, *, surface) -> TestClient:
    from hermes_cli import web_server as server
    from hermes_cli.web_server_dashboard import mount_spa

    (dist / "index.html").write_text(HTML)
    (dist / "assets").mkdir(exist_ok=True)
    app = FastAPI()
    if surface is not None:
        app.state.ui_surface = surface
    app.state.web_dist = dist
    monkeypatch.setattr(server, "app", app)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(app)
    return TestClient(app)


def _login_font_urls() -> list[str]:
    from hermes_cli.dashboard_auth.login_page import render_login_html

    urls = re.findall(r"url\('(/fonts/[^']+)'\)", render_login_html())
    assert urls
    return sorted(set(urls))


@pytest.mark.parametrize("surface", ["webapp", None], ids=["webapp", "dashboard"])
def test_the_sign_in_page_fonts_load_as_fonts(monkeypatch, tmp_path, surface):
    client = _spa(monkeypatch, tmp_path, surface=surface)

    for url in _login_font_urls():
        response = client.get(url)
        assert response.status_code == 200, url
        assert response.headers["content-type"] == "font/woff2", url
        assert response.content == (WEB_PUBLIC / url.lstrip("/")).read_bytes()


def test_a_font_in_the_served_bundle_wins(monkeypatch, tmp_path):
    (tmp_path / "fonts").mkdir()
    (tmp_path / "fonts" / "Collapse-Regular.woff2").write_bytes(b"bundle copy")

    response = _spa(monkeypatch, tmp_path, surface="webapp").get("/fonts/Collapse-Regular.woff2")

    assert response.status_code == 200
    assert response.content == b"bundle copy"


def test_sign_in_fonts_load_from_the_packaged_dashboard_without_a_source_tree(monkeypatch, tmp_path):
    """Nix and wheel installs ship no ``web/public``; the packaged dashboard dist has the fonts."""
    from hermes_cli import web_server as server
    from hermes_cli import web_server_static_assets as static_assets

    dashboard = tmp_path / "dashboard"
    (dashboard / "fonts").mkdir(parents=True)
    urls = _login_font_urls()
    for url in urls:
        (dashboard / url.lstrip("/")).write_bytes(b"packaged " + url.encode())
    webapp = tmp_path / "webapp"
    webapp.mkdir()
    monkeypatch.setattr(static_assets, "_WEB_PUBLIC", tmp_path / "no-source-tree" / "web" / "public")
    monkeypatch.setattr(server, "WEB_DIST", dashboard)
    client = _spa(monkeypatch, webapp, surface="webapp")

    for url in urls:
        response = client.get(url)
        assert response.status_code == 200, url
        assert response.content == b"packaged " + url.encode()


@pytest.mark.parametrize("surface", ["webapp", None], ids=["webapp", "dashboard"])
@pytest.mark.parametrize("path", [
    "/fonts/Missing-Regular.woff2",
    "/fonts-terminal/missing.woff2",
    "/ds-assets/missing.png",
    "/assets/missing-abc123.js",
    "/fonts/..%2f..%2fpyproject.toml",
])
def test_a_missing_asset_is_404_not_the_spa_shell(monkeypatch, tmp_path, surface, path):
    response = _spa(monkeypatch, tmp_path, surface=surface).get(path)

    assert response.status_code == 404
    assert "spa shell" not in response.text


@pytest.mark.parametrize("surface", ["webapp", None], ids=["webapp", "dashboard"])
def test_client_routes_still_get_the_spa_shell(monkeypatch, tmp_path, surface):
    response = _spa(monkeypatch, tmp_path, surface=surface).get("/chat/fonts-and-assets")

    assert response.status_code == 200
    assert "spa shell" in response.text
