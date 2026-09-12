"""PWA asset serving for the dashboard SPA.

The dashboard is installable to a phone or laptop home screen, which is the
only way to reach it from a device that cannot host Hermes itself. Installation
depends on three things the server controls: the manifest must be reachable at
the URL ``index.html`` advertises, that URL must survive a reverse-proxy path
prefix, and the service worker must never be pinned by an intermediary cache.

These assert the server contract only. The manifest's own shape and the tags in
``web/index.html`` are the frontend's, and are covered by the vitest suite.
"""

import re

import pytest

import hermes_cli.web_server_dashboard as _web_server_dashboard


def _dist_with_pwa(tmp_path):
    """A built-bundle layout carrying the PWA files Vite copies from public/."""
    dist = tmp_path / "web_dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "icons").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<html><head>"
        '<link rel="manifest" href="/manifest.webmanifest" />'
        '<link rel="apple-touch-icon" href="/icons/apple-touch-icon.png" />'
        "</head><body>SPA</body></html>",
        encoding="utf-8",
    )
    (dist / "manifest.webmanifest").write_text(
        '{"name": "Hermes Agent Dashboard", "start_url": ".", "display": "standalone"}',
        encoding="utf-8",
    )
    (dist / "icons" / "apple-touch-icon.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (dist / "sw.js").write_text("self.addEventListener('fetch', () => {});", encoding="utf-8")
    (dist / "offline.html").write_text("<html><body>offline</body></html>", encoding="utf-8")
    return dist


@pytest.fixture
def pwa_client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from starlette.testclient import TestClient
    import hermes_cli.web_server as ws

    monkeypatch.setattr(ws, "WEB_DIST", _dist_with_pwa(tmp_path))
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    spa_app = FastAPI()
    _web_server_dashboard.mount_spa(spa_app)
    return TestClient(spa_app)


def _hrefs(html: str) -> dict[str, str]:
    """Map ``rel`` → ``href`` for the <link> tags in a served index.html."""
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r'<link rel="([^"]+)" href="([^"]+)"', html)
    }


class TestManifestReachability:
    """Every URL index.html advertises must resolve on the same server."""

    def test_advertised_pwa_urls_are_served(self, pwa_client):
        links = _hrefs(pwa_client.get("/").text)
        assert links["manifest"] and links["apple-touch-icon"]
        for href in (links["manifest"], links["apple-touch-icon"]):
            assert pwa_client.get(href).status_code == 200, href

    def test_manifest_is_served_as_a_manifest(self, pwa_client):
        # A ``rel=manifest`` link served as text/html is ignored by the
        # browser and the install prompt never appears.
        resp = pwa_client.get("/manifest.webmanifest")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/manifest+json")

    def test_advertised_pwa_urls_follow_a_proxy_prefix(self, pwa_client):
        # Behind ``X-Forwarded-Prefix: /hermes`` the manifest lives at
        # /hermes/manifest.webmanifest; an unrewritten "/" href would 404 on
        # the proxy and silently cost installability.
        html = pwa_client.get("/", headers={"X-Forwarded-Prefix": "/hermes"}).text
        links = _hrefs(html)
        assert links["manifest"] == "/hermes/manifest.webmanifest"
        assert links["apple-touch-icon"] == "/hermes/icons/apple-touch-icon.png"

    def test_unprefixed_index_keeps_root_urls(self, pwa_client):
        # The rewrite is prefix-driven, never unconditional.
        links = _hrefs(pwa_client.get("/").text)
        assert links["manifest"] == "/manifest.webmanifest"
        assert links["apple-touch-icon"] == "/icons/apple-touch-icon.png"


class TestServiceWorkerServing:
    """The worker script decides the caching rules for every installed client."""

    def test_worker_is_javascript(self, pwa_client):
        resp = pwa_client.get("/sw.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]

    def test_worker_is_never_stored_by_an_intermediary(self, pwa_client):
        # Browsers bypass their own HTTP cache for the worker script, but a CDN
        # or reverse proxy does not — a pinned worker keeps serving its old
        # rules to clients that already installed it.
        resp = pwa_client.get("/sw.js")
        assert "no-store" in resp.headers["cache-control"]

    def test_worker_404s_rather_than_falling_back_to_the_spa(self, tmp_path, monkeypatch):
        # The SPA catch-all answers unknown paths with index.html. Serving HTML
        # at /sw.js makes registration fail with a confusing MIME error, so a
        # dist built without the worker must 404 instead.
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        import hermes_cli.web_server as ws

        dist = _dist_with_pwa(tmp_path)
        (dist / "sw.js").unlink()
        monkeypatch.setattr(ws, "WEB_DIST", dist)
        monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
        spa_app = FastAPI()
        _web_server_dashboard.mount_spa(spa_app)
        resp = TestClient(spa_app).get("/sw.js")
        assert resp.status_code == 404
        assert "SPA" not in resp.text

    def test_offline_notice_is_reachable_for_precaching(self, pwa_client):
        # The worker precaches this on install; if it 404s the install still
        # succeeds but a failed navigation shows the browser's error page.
        assert pwa_client.get("/offline.html").status_code == 200


class TestGatedDeploymentStaysInstallable:
    """A hosted dashboard is the whole point of installing to a home screen.

    The install surface has to resolve for a browser that has not logged in:
    the manifest is fetched with credentials OMITTED and parsed before any
    session exists, and a worker script that answers with a login redirect
    fails registration on a MIME-type error. Both would leave the hosted
    deployment — the one people reach from a phone — permanently
    un-installable, with no error anywhere.
    """

    @staticmethod
    def _is_public(path: str) -> bool:
        from hermes_cli.dashboard_auth.middleware import _path_is_public

        return _path_is_public(path)

    def test_install_surface_bypasses_the_cookie_gate(self):
        for path in (
            "/manifest.webmanifest",
            "/icons/icon-192.png",
            "/icons/apple-touch-icon.png",
            "/sw.js",
            "/offline.html",
        ):
            assert self._is_public(path), path

    def test_the_app_and_its_data_stay_gated(self):
        # The bypass covers static branding and caching rules only. Widening it
        # to the SPA or the API would hand an unauthenticated caller the
        # dashboard itself.
        for path in ("/", "/chat", "/api/config", "/api/env", "/api/sessions"):
            assert not self._is_public(path), path
