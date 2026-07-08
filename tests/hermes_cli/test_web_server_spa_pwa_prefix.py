"""SPA/PWA serving tests for path-prefixed dashboard installs."""


def test_spa_rewrites_pwa_links_under_forwarded_prefix(tmp_path, monkeypatch):
    """Install metadata must work when Hermes is served below a VPN/proxy path."""
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import hermes_cli.web_server as ws

    dist = tmp_path / "web_dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "assets" / "index.css").write_text("body{}", encoding="utf-8")
    (dist / "favicon.ico").write_bytes(b"ico")
    (dist / "icon-192.png").write_bytes(b"png192")
    (dist / "icon-512.png").write_bytes(b"png512")
    (dist / "manifest.webmanifest").write_text(
        '{"start_url":".","scope":".","icons":[{"src":"icon-192.png"}]}',
        encoding="utf-8",
    )
    (dist / "index.html").write_text(
        """
        <!doctype html>
        <html><head>
          <link rel="icon" type="image/x-icon" href="/favicon.ico" />
          <link rel="icon" type="image/png" sizes="192x192" href="/icon-192.png" />
          <link rel="apple-touch-icon" href="/icon-192.png" />
          <link rel="manifest" href="/manifest.webmanifest" />
          <link rel="stylesheet" href="/assets/index.css" />
        </head><body></body></html>
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(ws, "WEB_DIST", dist)
    monkeypatch.setattr(ws, "_SESSION_TOKEN", "test-token")
    ws.app.state.auth_required = False

    app = FastAPI()
    ws.mount_spa(app)
    client = TestClient(app)

    response = client.get("/chat", headers={"X-Forwarded-Prefix": "/hermes"})

    assert response.status_code == 200
    html = response.text
    assert 'href="/hermes/favicon.ico"' in html
    assert 'href="/hermes/icon-192.png"' in html
    assert 'href="/hermes/manifest.webmanifest"' in html
    assert 'href="/hermes/assets/index.css"' in html
    assert 'window.__HERMES_BASE_PATH__="/hermes"' in html

    manifest = client.get("/manifest.webmanifest")
    assert manifest.status_code == 200
    assert manifest.json()["start_url"] == "."
    assert manifest.json()["scope"] == "."
