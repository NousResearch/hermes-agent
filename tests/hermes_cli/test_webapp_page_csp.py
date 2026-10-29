"""The privileged page authorizes its bootstrap and bundle, never arbitrary inline script."""
import base64
import hashlib
from html.parser import HTMLParser

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class Scripts(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.inline = []
        self.external = []
        self.current = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            src = dict(attrs).get("src")
            if src:
                self.external.append(src)
            else:
                self.current = ""

    def handle_data(self, data):
        if self.current is not None:
            self.current += data

    def handle_endtag(self, tag):
        if tag == "script" and self.current is not None:
            self.inline.append(self.current)
            self.current = None


@pytest.fixture
def page(monkeypatch, tmp_path):
    from hermes_cli import web_server as server
    from hermes_cli.web_server_dashboard import mount_spa

    (tmp_path / "index.html").write_text(
        '<html><head><script>window.booted = "<&>";</script>'
        '<script type="module" src="/assets/app.js"></script></head><body></body></html>')
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/app.js").write_text('import("./lazy.js");')
    (tmp_path / "assets/lazy.js").write_text('export const loaded = true;')
    app = FastAPI()
    app.state.ui_surface = "webapp"
    app.state.auth_required = False
    app.state.web_dist = tmp_path
    monkeypatch.setattr(server, "app", app)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(app)
    return app


@pytest.mark.parametrize("origin", ["http://testserver:9120", "https://webapp.example:9443"])
@pytest.mark.parametrize("path", ["/", "/index.html", "/chat/session"])
def test_page_allows_its_scripts_and_chunks_but_locks_privileged_execution(page, origin, path):
    client = TestClient(page, base_url=origin)
    response = client.get(path, headers={"X-Forwarded-Prefix": "/hermes"})
    assert response.status_code == 200
    policy = dict(part.strip().split(" ", 1) for part in response.headers["content-security-policy"].split(";"))
    assert policy["object-src"] == policy["base-uri"] == policy["frame-ancestors"] == "'none'"
    assert policy["script-src-attr"] == "'none'"
    scripts = policy["script-src"].split()
    assert "'self'" in scripts  # the bundle and every relative import() chunk
    assert "blob:" in scripts  # runtime-loader's local plugin modules (#145072)
    assert not {"'unsafe-inline'", "'unsafe-eval'", "https:", "http:", "*"}.intersection(scripts)
    parsed = Scripts(response.text)
    assert parsed.external == ["/hermes/assets/app.js"]
    assert any("__HERMES_BASE_PATH__" in script for script in parsed.inline)
    for script in parsed.inline:
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in scripts
    injected = base64.b64encode(hashlib.sha256(b"window.injected=true").digest()).decode()
    assert f"'sha256-{injected}'" not in scripts
    # Connections stay open (provider-direct voice, catalogs, proxied sockets); execution does not.
    assert policy["connect-src"] == "* ws: wss: data: blob:"  # explicit sockets for older WebKit
    for asset in ("app.js", "lazy.js"):
        chunk = client.get(f"/assets/{asset}")
        assert chunk.status_code == 200
        assert "javascript" in chunk.headers["content-type"]
    assert "'self'" in policy["img-src"].split()
    assert policy["media-src"] == "* data: blob:"  # gateway file routes and http: radio streams
    assert "no-store" in response.headers["cache-control"]
    preview = client.get("/assets/webapp-preview.html")
    preview_scripts = dict(part.strip().split(" ", 1) for part in
                           preview.headers["content-security-policy"].split(";"))["script-src"].split()
    assert {"'unsafe-inline'", "'unsafe-eval'"} <= set(preview_scripts)  # widget scripts, d3.csvParse


@pytest.mark.parametrize("path", ["/%2e%2findex.html", "/%2e/index.html", "/INDEX.HTML"])
def test_every_spelling_of_the_entry_document_gets_the_policy(page, tmp_path, path):
    if path == "/INDEX.HTML" and not (tmp_path / "INDEX.HTML").exists():
        pytest.skip("case-sensitive filesystem")
    response = TestClient(page).get(path)
    assert response.status_code == 200
    assert "'sha256-" in response.headers["content-security-policy"]
    assert "__HERMES_BASE_PATH__" in response.text


def test_dashboard_keeps_its_existing_document_headers(page):
    page.state.ui_surface = "dashboard"
    assert "content-security-policy" not in TestClient(page).get("/").headers
