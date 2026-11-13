"""The Webapp page names the bundle it boots, and ``/api/webapp/bundle`` names the bundle a page load
gets now and whether it is current. A tab compares the two to move onto a rebuilt bundle, and must
not move onto one ``hermes update`` could not rebuild (its page load is refused)."""
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.webapp import STALE_MARKER


def _index(dist, entry):
    (dist / "index.html").write_text(
        f'<html><head><script type="module" src="/assets/{entry}.js"></script></head><body></body></html>')


@pytest.fixture
def webapp(monkeypatch, tmp_path):
    from hermes_cli import web_server as server
    from hermes_cli.web_routers.webapp import router
    from hermes_cli.web_server_dashboard import mount_spa

    _index(tmp_path, "index-A")
    app = FastAPI()
    app.state.ui_surface = "webapp"
    app.state.auth_required = False
    app.state.web_dist = tmp_path
    monkeypatch.setattr(server, "app", app)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    app.include_router(router)
    mount_spa(app)
    return TestClient(app), tmp_path


def _booted(client) -> str:
    match = re.search(r'window\.__HERMES_WEBAPP_BUNDLE__="([0-9a-f]+)";', client.get("/").text)
    assert match, "the page names the bundle it boots"
    return match.group(1)


def test_page_and_route_name_the_same_bundle_until_a_rebuild_replaces_it(webapp):
    client, dist = webapp
    booted = _booted(client)
    answer = client.get("/api/webapp/bundle")

    assert answer.json() == {"id": booted, "current": True}
    assert answer.headers["cache-control"] == "no-store"

    _index(dist, "index-B")
    rebuilt = client.get("/api/webapp/bundle").json()

    assert rebuilt["current"] is True and rebuilt["id"] != booted
    assert _booted(client) == rebuilt["id"]


def test_a_bundle_the_update_could_not_rebuild_is_not_current(webapp):
    client, dist = webapp
    (dist / STALE_MARKER).write_text("x")

    assert client.get("/api/webapp/bundle").json()["current"] is False


def test_no_bundle_built_is_not_current(webapp):
    client, dist = webapp
    (dist / "index.html").unlink()

    assert client.get("/api/webapp/bundle").json() == {"id": None, "current": False}


def test_only_the_webapp_names_a_bundle(webapp):
    client, _dist = webapp
    client.app.state.ui_surface = "dashboard"

    assert client.get("/api/webapp/bundle").status_code == 404
    assert "__HERMES_WEBAPP_BUNDLE__" not in client.get("/").text
