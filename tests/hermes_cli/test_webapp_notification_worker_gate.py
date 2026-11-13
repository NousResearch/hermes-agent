"""The Webapp's notification worker answers without a session, like the page icons.

The browser fetches ``/notification-worker.js`` to register the worker and again for each update
check, which a notification click can start long after the sign-in expired. Behind the gate, a gated
worker script answers that check with the sign-in redirect: the update fails, and the redirect starts
an auto-SSO ``login_start`` with no sign-in behind it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

WORKER = b"self.addEventListener('notificationclick', () => {})\n"


@pytest.fixture
def gated_webapp(monkeypatch, tmp_path):
    """The gateway's Webapp behind the sign-in gate, serving a renderer dist with the worker."""
    dist = tmp_path / "dist-webapp"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>Hermes</title>", encoding="utf-8")
    (dist / "notification-worker.js").write_bytes(WORKER)
    clear_providers()
    register_provider(StubAuthProvider())
    for key, value in (("bound_host", "0.0.0.0"), ("bound_port", 9119), ("auth_required", True),
                       ("trusted_public_hosts", frozenset()), ("ui_surface", "webapp"),
                       ("listener_public_url", ""), ("web_dist", dist)):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    yield TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    clear_providers()


@pytest.mark.parametrize("prefix", [None, "/hermes"])
def test_worker_script_answers_without_a_session(gated_webapp, prefix):
    """Also behind a path-prefix proxy, which strips its prefix before the request arrives."""
    headers = {"X-Forwarded-Prefix": prefix} if prefix else {}

    response = gated_webapp.get("/notification-worker.js", headers=headers, follow_redirects=False)

    assert response.status_code == 200, response.headers.get("location")
    assert response.content == WORKER
    # A service worker registers only from a script served with a JavaScript MIME type.
    assert response.headers["content-type"].split(";")[0] == "text/javascript"


@pytest.mark.parametrize("path", ["/notification-worker.js.map", "/notification-worker.jsx", "/sw.js"])
def test_other_scripts_still_require_sign_in(gated_webapp, path):
    response = gated_webapp.get(path, follow_redirects=False)

    assert response.status_code == 302
    assert "/login" in response.headers["location"]


def test_missing_worker_is_a_404_not_the_page(gated_webapp):
    (web_server.app.state.web_dist / "notification-worker.js").unlink()

    response = gated_webapp.get("/notification-worker.js", follow_redirects=False)

    assert response.status_code == 404
    assert "html" not in response.headers.get("content-type", "")
