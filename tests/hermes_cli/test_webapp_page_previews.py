"""Preview CDN loads use the isolated document policy, never the app policy."""
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# (directive the load is checked against, its fallbacks, a URL a widget loads)
PREVIEW_LOADS = [
    (("script-src-elem", "script-src", "default-src"), "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"),
    (("style-src-elem", "style-src", "default-src"), "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"),
]


def _page(monkeypatch, tmp_path):
    from hermes_cli import web_server as server
    from hermes_cli.web_server_dashboard import mount_spa

    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html><head></head><body></body></html>")
    application = FastAPI()
    application.state.ui_surface = "webapp"
    application.state.auth_required = False
    monkeypatch.setattr(server, "app", application)
    monkeypatch.setattr(server, "WEB_DIST", tmp_path)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(application)
    return TestClient(application)


def _allows(policy: str, directives: tuple[str, ...], url: str) -> bool:
    """Whether *policy* lets the page load *url* under the first of *directives* it sets (no such
    directive allows it). Only the source forms that could admit a public CDN are matched."""
    parsed = {}
    for part in policy.split(";"):
        name, *sources = part.split() or [""]
        if name:
            parsed.setdefault(name.lower(), sources)
    sources = next((parsed[name] for name in directives if name in parsed), None)
    if sources is None:
        return True
    host = urlsplit(url).hostname or ""
    return any(source in ("*", "https:", host, f"https://{host}")
               or (source.startswith("https://*.") and host.endswith(source[len("https://*"):]))
               for source in sources)


@pytest.mark.parametrize(("directives", "url"), PREVIEW_LOADS, ids=[d[0] for d, _ in PREVIEW_LOADS])
def test_a_live_preview_can_load_its_cdn_libraries_under_its_own_policy(
        monkeypatch, tmp_path, directives, url):
    response = _page(monkeypatch, tmp_path).get("/assets/webapp-preview.html")
    assert response.status_code == 200
    for policy in response.headers.get_list("content-security-policy"):
        assert _allows(policy, directives, url), f"{directives[0]} blocks {url}: {policy}"
