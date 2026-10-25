"""The Webapp page's Content-Security-Policy (``hermes_cli.webapp_page_policy``).

The gateway serves the Desktop renderer to a browser; its page carries a policy that the renderer, its
lazy chunks, workers, plugins, media, previews and same-origin sockets work under, and that shuts
everything else. The sources below are what the built bundle uses (see the module's docstring).
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _page(monkeypatch, tmp_path, surface: str):
    from hermes_cli import web_server as server
    from hermes_cli.web_server_dashboard import mount_spa

    (tmp_path / "index.html").write_text(
        "<html><head><script>document.documentElement.dataset.boot = '1'</script></head><body></body></html>")
    application = FastAPI()
    application.state.ui_surface = surface
    application.state.auth_required = False
    monkeypatch.setattr(server, "app", application)
    monkeypatch.setattr(server, "WEB_DIST", tmp_path)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(application)
    return TestClient(application)


def _directives(policy: str) -> dict[str, list[str]]:
    parsed = {}
    for part in policy.split(";"):
        name, *sources = part.split()
        assert name not in parsed, f"{name} appears twice"
        parsed[name] = sources
    return parsed


@pytest.mark.parametrize("path", ["/", "/chat", "/settings", "/index.html"])
def test_the_webapp_page_sends_a_content_security_policy(monkeypatch, tmp_path, path):
    from hermes_cli.webapp_page_policy import CONTENT_SECURITY_POLICY

    response = _page(monkeypatch, tmp_path, "webapp").get(path)
    assert response.status_code == 200
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert response.headers["cache-control"].startswith("no-store")


def test_the_dashboards_page_keeps_mains_headers(monkeypatch, tmp_path):
    assert "content-security-policy" not in _page(monkeypatch, tmp_path, "dashboard").get("/").headers


# (directive, source, what in the built renderer needs it)
NEEDED = [
    ("script-src", "'self'", "the entry module and its lazy chunks"),
    ("script-src", "'unsafe-inline'", "the pre-paint and bootstrap scripts, and scripts in srcdoc previews"),
    ("script-src", "blob:", "runtime plugins and the SDK shims they import (contrib/runtime-loader.ts)"),
    ("script-src", "'wasm-unsafe-eval'", "Shiki's Oniguruma WebAssembly"),
    ("worker-src", "blob:", "blob: workers"),
    ("worker-src", "'self'", "same-origin workers"),
    ("connect-src", "'self'", "REST, /api/ws, the host terminal and voice sockets on this origin"),
    ("connect-src", "https:", "public APIs plugins call (Radio's directory and track lookups)"),
    ("connect-src", "blob:", "reads of blob: previews and downloads"),
    ("style-src", "'unsafe-inline'", "inline styles from React, KaTeX, Mermaid and Shiki"),
    ("style-src", "https://fonts.googleapis.com", "a theme's Google Fonts stylesheet"),
    ("font-src", "https://fonts.gstatic.com", "Google's font files"),
    ("font-src", "'self'", "the bundled KaTeX and JetBrains Mono fonts"),
    ("font-src", "data:", "inlined fonts"),
    ("img-src", "data:", "data: previews"),
    ("img-src", "blob:", "blob: previews"),
    ("img-src", "https:", "images a message links"),
    ("media-src", "blob:", "voice playback"),
    ("media-src", "'self'", "gateway media links"),
    ("media-src", "https:", "Radio's streams"),
    ("frame-src", "blob:", "the PDF preview"),
    ("frame-src", "data:", "data: HTML previews"),
    ("frame-src", "https:", "YouTube, Spotify, map and post embeds"),
    ("frame-src", "http:", "the sandboxed preview of a local dev server"),
]


@pytest.mark.parametrize(("directive", "source", "need"), NEEDED, ids=[f"{d} {s}" for d, s, _ in NEEDED])
def test_the_policy_allows_each_source_the_renderer_uses(directive, source, need):
    from hermes_cli.webapp_page_policy import CONTENT_SECURITY_POLICY

    directives = _directives(CONTENT_SECURITY_POLICY)
    assert source in directives.get(directive, directives["default-src"]), need


def test_the_policy_shuts_everything_the_renderer_does_not_use():
    """No string evaluation, no plugins, no <base> or form posts elsewhere, no page framing the
    Webapp, no wildcard source, and no socket or plain-HTTP fetch to another origin."""
    from hermes_cli.webapp_page_policy import CONTENT_SECURITY_POLICY

    directives = _directives(CONTENT_SECURITY_POLICY)
    assert directives["default-src"] == ["'self'"]
    assert directives["object-src"] == ["'none'"]
    assert directives["base-uri"] == ["'self'"]
    assert directives["form-action"] == ["'self'"]
    assert directives["frame-ancestors"] == ["'none'"]
    assert all("'unsafe-eval'" not in sources for sources in directives.values())
    assert all("*" not in sources for sources in directives.values())
    assert not {"http:", "ws:", "wss:"} & set(directives["connect-src"])
    assert set(directives["script-src"]) == {"'self'", "'unsafe-inline'", "blob:", "'wasm-unsafe-eval'"}
