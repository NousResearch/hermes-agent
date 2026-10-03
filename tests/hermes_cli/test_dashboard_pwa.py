from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import web_server


def _build_dist(root: Path) -> None:
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(
        """
<!doctype html>
<html>
  <head>
    <link rel="icon" href="/favicon.ico" />
    <link rel="manifest" href="/manifest.webmanifest" />
    <link rel="apple-touch-icon" href="/pwa-icon-180.png" />
    <script type="module" src="/assets/index.js"></script>
    <link rel="stylesheet" href="/assets/index.css" />
  </head>
  <body><div id="root"></div></body>
</html>
""".strip(),
        encoding="utf-8",
    )
    (root / "manifest.webmanifest").write_text(
        '{"name":"Hermes","start_url":".","scope":"."}',
        encoding="utf-8",
    )
    (root / "sw.js").write_text("self.addEventListener('fetch', () => {});", encoding="utf-8")
    (root / "pwa-icon-180.png").write_bytes(b"png")


def test_dashboard_pwa_links_are_prefix_safe(monkeypatch, tmp_path):
    _build_dist(tmp_path)
    monkeypatch.setattr(web_server, "WEB_DIST", tmp_path)
    web_server.app.state.auth_required = False

    app = FastAPI()
    web_server.mount_spa(app)
    client = TestClient(app)

    response = client.get("/", headers={"X-Forwarded-Prefix": "/hermes"})

    assert response.status_code == 200
    html = response.text
    assert 'href="/hermes/manifest.webmanifest"' in html
    assert 'href="/hermes/pwa-icon-180.png"' in html
    assert 'src="/hermes/assets/index.js"' in html
    assert 'window.__HERMES_BASE_PATH__="/hermes"' in html
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"


def test_dashboard_serves_manifest_and_service_worker_with_pwa_headers(monkeypatch, tmp_path):
    _build_dist(tmp_path)
    monkeypatch.setattr(web_server, "WEB_DIST", tmp_path)

    app = FastAPI()
    web_server.mount_spa(app)
    client = TestClient(app)

    manifest = client.get("/manifest.webmanifest")
    assert manifest.status_code == 200
    assert manifest.headers["content-type"].startswith("application/manifest+json")
    assert manifest.headers["cache-control"] == "public, max-age=3600"

    service_worker = client.get("/sw.js", headers={"X-Forwarded-Prefix": "/hermes"})
    assert service_worker.status_code == 200
    assert service_worker.headers["content-type"].startswith("application/javascript")
    assert service_worker.headers["service-worker-allowed"] == "/hermes/"
    assert service_worker.headers["cache-control"] == "no-cache"
