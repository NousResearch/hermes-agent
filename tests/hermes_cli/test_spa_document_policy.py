"""Every privileged SPA document uses the same bootstrap policy: uncached, prefixed, and tokenless
whenever the gate or the surface withholds the session token. Where the surface's policy keeps the
token out of HTML (the Webapp), ``/index.html`` gets that bootstrap too; the dashboard serves its
``/index.html`` and its bootstrap as on main."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HTML = '<html><head><script src="/assets/app.js"></script></head><body>fixture</body></html>'


def _spa(monkeypatch, tmp_path, *, surface, gated) -> TestClient:
    from hermes_cli import web_server as server
    from hermes_cli.web_server_dashboard import mount_spa

    (tmp_path / "index.html").write_text(HTML)
    (tmp_path / "help.html").write_text("<html><body>help</body></html>")
    app = FastAPI()
    if surface is not None:
        app.state.ui_surface = surface
    app.state.auth_required = gated
    app.state.web_dist = tmp_path
    monkeypatch.setattr(server, "app", app)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(app)
    return TestClient(app)


@pytest.mark.parametrize("surface", ["webapp", None], ids=["webapp", "dashboard"])
@pytest.mark.parametrize("prefix", ["", "/hermes"])
@pytest.mark.parametrize("path", ["/", "/chat/session", "/index.html", "/help.html"])
def test_spa_documents_share_the_bootstrap_policy(monkeypatch, tmp_path, surface, prefix, path):
    from hermes_cli import web_server as server

    response = _spa(monkeypatch, tmp_path, surface=surface, gated=True).get(
        path, headers={"X-Forwarded-Prefix": prefix})
    assert response.status_code == 200
    if path in ("/", "/chat/session") or (path == "/index.html" and surface == "webapp"):
        # On the Webapp /index.html is included: every entry to the app gets the bootstrap.
        assert "no-store" in response.headers.get("cache-control", "")
        assert f'window.__HERMES_BASE_PATH__="{prefix}"' in response.text
        assert f'src="{prefix}/assets/app.js"' in response.text
    assert server._SESSION_TOKEN not in response.text


@pytest.mark.parametrize("gated", [False, True])
def test_the_dashboard_serves_index_html_and_its_bootstrap_as_on_main(monkeypatch, tmp_path, gated):
    """With the Webapp off, ``/index.html`` is the built file itself and the bootstrap names no
    surface: only the Webapp's page carries ``__HERMES_UI_SURFACE__``, which its renderer reads."""
    client = _spa(monkeypatch, tmp_path, surface=None, gated=gated)

    raw = client.get("/index.html")
    assert raw.status_code == 200
    assert raw.text == HTML
    assert "__HERMES_UI_SURFACE__" not in client.get("/").text


def test_the_webapp_page_names_its_surface(monkeypatch, tmp_path):
    client = _spa(monkeypatch, tmp_path, surface="webapp", gated=False)

    for path in ("/", "/index.html"):
        assert 'window.__HERMES_UI_SURFACE__="webapp"' in client.get(path).text
