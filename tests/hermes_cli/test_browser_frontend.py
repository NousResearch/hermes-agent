"""Desktop browser selection and reuse of the real server's OAuth/cookie transport."""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.ws_tickets import TicketInvalid, consume_ticket
from hermes_cli.main_browser_frontend import configure_browser_frontend
from hermes_cli.subcommands.dashboard import build_dashboard_parser
from hermes_cli.web_server_dashboard import mount_spa
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


def test_server_frontend_selection_is_opt_in():
    parser = argparse.ArgumentParser()
    build_dashboard_parser(parser.add_subparsers(dest="command"),
                           cmd_dashboard=lambda args: None, cmd_dashboard_register=lambda args: None)
    assert parser.parse_args(["serve"]).ui is None
    assert parser.parse_args(["dashboard"]).ui == "dashboard"
    assert parser.parse_args(["serve", "--ui", "desktop"]).ui == "desktop"
    assert parser.parse_args(["dashboard", "--ui", "desktop"]).ui == "desktop"


def test_selects_prebuilt_renderer_and_clears_headless_state(tmp_path, monkeypatch):
    dist = tmp_path / "apps/desktop/dist/browser"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html>Desktop</html>")
    monkeypatch.delenv("HERMES_WEB_DIST", raising=False)
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")
    configure_browser_frontend(SimpleNamespace(skip_build=True), tmp_path)
    import os
    assert os.environ["HERMES_WEB_DIST"] == str(dist)
    assert "HERMES_SERVE_HEADLESS" not in os.environ


def test_missing_explicit_dist_fails_without_starting_another_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_WEB_DIST", str(tmp_path / "missing"))
    with pytest.raises(SystemExit, match="not built"):
        configure_browser_frontend(SimpleNamespace(skip_build=True), tmp_path)


@pytest.fixture
def browser_app(tmp_path, monkeypatch):
    dist = tmp_path / "browser"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<html><head></head><body>Desktop renderer<script src="/assets/browser.js"></script></body></html>')
    (dist / "assets/browser.js").write_text('console.log("browser fixture")')
    (dist / "hermes-frontend.json").write_text(json.dumps({"frontend": "desktop"}, separators=(",", ":")))
    monkeypatch.setattr(web_server, "WEB_DIST", dist)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "browser.example.test", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_port", 443, raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(web_server.app.state, "trusted_public_hosts", frozenset(), raising=False)
    monkeypatch.setattr(web_server.app.state, "serves_spa", True, raising=False)
    # Reuse the real server's middleware/router assembly, replacing only its SPA
    # catch-all with a fixture build so the test does not require Node.
    prior_routes = list(web_server.app.router.routes)
    web_server.app.router.routes[:] = [route for route in prior_routes if getattr(route, 'path', '') not in ('/{full_path:path}', '/assets', '/assets/{filename}.css')]
    mount_spa(web_server.app)
    clear_providers()
    register_provider(StubAuthProvider())
    yield TestClient(web_server.app, base_url="https://browser.example.test")
    web_server.app.router.routes[:] = prior_routes
    clear_providers()


def _login(client):
    start = client.get('/auth/login?provider=stub', follow_redirects=False)
    assert start.status_code == 302
    callback = client.get(start.headers['location'], follow_redirects=False)
    assert callback.status_code == 302
    return callback


def test_desktop_html_uses_backend_oauth_without_exposing_native_token(browser_app):
    response = browser_app.get('/', follow_redirects=False)
    assert response.status_code == 302
    _login(browser_app)
    response = browser_app.get('/')
    assert response.status_code == 200
    assert 'Desktop renderer' in response.text
    assert '__HERMES_AUTH_REQUIRED__=true' in response.text
    assert '__HERMES_SESSION_TOKEN__=' not in response.text
    assert 'no-store' in response.headers['cache-control']
    assert browser_app.get('/api/auth/me').status_code == 200


def test_authenticated_browser_gets_fresh_single_use_ws_tickets(browser_app):
    assert browser_app.post('/api/auth/ws-ticket', follow_redirects=False).status_code == 401
    _login(browser_app)
    headers = {'Origin': 'https://browser.example.test'}
    first = browser_app.post('/api/auth/ws-ticket', headers=headers).json()['ticket']
    second = browser_app.post('/api/auth/ws-ticket', headers=headers).json()['ticket']
    assert first != second
    assert consume_ticket(first)['provider'] == 'stub'
    with pytest.raises(TicketInvalid):
        consume_ticket(first)
    assert consume_ticket(second)['provider'] == 'stub'


def test_browser_logout_uses_existing_backend_session_route(browser_app):
    _login(browser_app)
    response = browser_app.post('/auth/logout', headers={'Origin': 'https://browser.example.test'}, follow_redirects=False)
    assert response.status_code in (200, 302, 303)
    assert browser_app.get('/api/auth/me', follow_redirects=False).status_code == 401


def test_loopback_html_bootstraps_existing_native_token(tmp_path, monkeypatch):
    dist = tmp_path / 'dist'
    (dist / 'assets').mkdir(parents=True)
    (dist / 'index.html').write_text('<html><head></head><body>Desktop</body></html>')
    monkeypatch.setattr(web_server, 'WEB_DIST', dist)
    monkeypatch.setattr(web_server.app.state, 'auth_required', False, raising=False)
    monkeypatch.delenv('HERMES_SERVE_HEADLESS', raising=False)
    application = FastAPI()
    mount_spa(application)
    response = TestClient(application).get('/')
    assert '__HERMES_SESSION_TOKEN__=' in response.text
    assert '__HERMES_AUTH_REQUIRED__=false' in response.text


def test_browser_build_preserves_other_source_frontend_dependencies(tmp_path, monkeypatch):
    from hermes_cli import source_build
    for workspace in ('ui-tui', 'web'):
        directory = tmp_path / workspace
        directory.mkdir()
        (directory / 'package.json').write_text('{}')
    dist = tmp_path / 'apps/desktop/dist/browser'
    dist.mkdir(parents=True)
    (dist / 'index.html').write_text('<html>Desktop</html>')
    monkeypatch.delenv('HERMES_WEB_DIST', raising=False)
    monkeypatch.setattr(source_build, 'source_build_env', dict)
    prepared = []
    built = []
    monkeypatch.setattr(source_build, 'prepare_source_dependencies', lambda root, workspaces, **kw: prepared.append((root, workspaces)))
    monkeypatch.setattr(source_build, 'run_source_script', lambda root, script, **kw: built.append(script))
    configure_browser_frontend(SimpleNamespace(skip_build=False), tmp_path)
    assert prepared == [(tmp_path, ('ui-tui', 'web', 'apps/desktop'))]
    assert built == ['scripts/build/desktop-browser.mjs']
