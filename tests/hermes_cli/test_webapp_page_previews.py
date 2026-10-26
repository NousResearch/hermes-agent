"""Live HTML previews on the gateway's Webapp page (``::preview`` widgets and artifacts).

They render in sandboxed ``srcdoc`` frames, which inherit the Content-Security-Policy of the page that
embeds them, and they load their libraries from a public CDN: the agent is told to pin each one as
``https://cdn.jsdelivr.net/npm/<package>@<version>/<file>`` (``agent/onboarding_text.py``). A policy on
the Webapp page that keeps scripts or stylesheets to its own origin blocks them, and the widget renders
nothing (headless Chromium: ``script-src-elem`` violation, the CDN never asked). Until previews move to
documents served with a policy of their own, whatever the page sends must let those loads through.
"""
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


@pytest.mark.parametrize("path", ["/", "/chat"])
@pytest.mark.parametrize(("directives", "url"), PREVIEW_LOADS, ids=[d[0] for d, _ in PREVIEW_LOADS])
def test_a_live_preview_can_load_its_cdn_libraries_under_the_webapp_pages_policy(
        monkeypatch, tmp_path, path, directives, url):
    response = _page(monkeypatch, tmp_path).get(path)
    assert response.status_code == 200
    for policy in response.headers.get_list("content-security-policy"):
        assert _allows(policy, directives, url), f"{directives[0]} blocks {url}: {policy}"
