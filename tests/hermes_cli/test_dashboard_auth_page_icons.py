"""Page icons answer without a session, like ``/favicon.ico``; other public files stay gated.

Browsers fetch the page icon and the home-screen icon without cookies, so a gated icon answers
with the sign-in redirect and logs an auto-SSO ``login_start`` with no sign-in behind it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def gated_webapp(monkeypatch, tmp_path):
    """The gateway's Webapp behind the sign-in gate, serving a renderer dist with its icons."""
    dist = tmp_path / "dist-webapp"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>Hermes</title>", encoding="utf-8")
    for name in ("apple-touch-icon.png", "favicon.ico", "hermes.png"):
        (dist / name).write_bytes(b"\x89PNG icon bytes")
    clear_providers()
    register_provider(StubAuthProvider())
    for key, value in (("bound_host", "0.0.0.0"), ("bound_port", 9119), ("auth_required", True),
                       ("trusted_public_hosts", frozenset()), ("ui_surface", "webapp"),
                       ("listener_public_url", ""), ("web_dist", dist)):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    yield TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    clear_providers()


@pytest.mark.parametrize("path", ["/apple-touch-icon.png", "/favicon.ico"])
def test_page_icons_answer_without_a_session(gated_webapp, path):
    response = gated_webapp.get(path, follow_redirects=False)

    assert response.status_code == 200, response.headers.get("location")
    assert response.content == b"\x89PNG icon bytes"


@pytest.mark.parametrize("path", ["/hermes.png", "/apple-touch-icon.png.map", "/apple-touch-icon.pngx"])
def test_other_files_still_require_sign_in(gated_webapp, path):
    response = gated_webapp.get(path, follow_redirects=False)

    assert response.status_code == 302
    assert "/login" in response.headers["location"]
