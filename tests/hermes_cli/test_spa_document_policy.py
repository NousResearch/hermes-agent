"""Every privileged SPA document uses the same bootstrap policy: uncached, prefixed, and tokenless
whenever the gate or the surface withholds the session token."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.parametrize("surface", ["webapp", "dashboard"])
@pytest.mark.parametrize("prefix", ["", "/hermes"])
@pytest.mark.parametrize("path", ["/", "/chat/session", "/index.html", "/help.html"])
def test_spa_documents_share_the_bootstrap_policy(monkeypatch, tmp_path, surface, prefix, path):
    from hermes_cli import web_server as server
    from hermes_cli.web_server_dashboard import mount_spa

    html = '<html><head><script src="/assets/app.js"></script></head><body>fixture</body></html>'
    (tmp_path / "index.html").write_text(html)
    (tmp_path / "help.html").write_text("<html><body>help</body></html>")
    app = FastAPI()
    app.state.ui_surface = surface
    app.state.auth_required = True
    app.state.web_dist = tmp_path
    monkeypatch.setattr(server, "app", app)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(app)
    response = TestClient(app).get(path, headers={"X-Forwarded-Prefix": prefix})
    assert response.status_code == 200
    if path != "/help.html":
        # /index.html included: every entry to the app gets the bootstrap, never the raw file.
        assert "no-store" in response.headers.get("cache-control", "")
        assert f'window.__HERMES_UI_SURFACE__="{surface}"' in response.text
        assert f'window.__HERMES_BASE_PATH__="{prefix}"' in response.text
        assert f'src="{prefix}/assets/app.js"' in response.text
    assert server._SESSION_TOKEN not in response.text
