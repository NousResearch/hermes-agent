"""Nous recommendations come from each Portal's static mirror (the Portal itself when the mirror has
no copy), and the gateway's background refresh fetches them only for a profile with a Nous account."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_constants import get_hermes_home

PAYLOAD = {"paidRecommendedModels": [{"modelName": "mirrored"}]}


@pytest.fixture
def paths(monkeypatch):
    """A local server standing in for the mirror (which holds no copy for a 127.0.0.1 Portal) and for
    a Portal at its own origin, recording the paths it is asked for."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(404 if self.path.startswith("/dynamic/127.0.0.1") else 200)
            self.end_headers()
            self.wfile.write(json.dumps(PAYLOAD).encode())

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    from hermes_cli import models
    monkeypatch.setattr(models, "NOUS_RECOMMENDED_MODELS_MIRROR", f"http://127.0.0.1:{httpd.server_port}/dynamic")
    yield seen
    httpd.shutdown()
    httpd.server_close()


@pytest.mark.parametrize("portal, host", [
    ("", "portal.nousresearch.com"),
    ("https://portal.staging-nousresearch.com/", "portal.staging-nousresearch.com"),
])
def test_recommendations_come_from_the_portal_hosts_mirror(paths, portal, host):
    from hermes_cli.models import fetch_nous_recommended_models

    assert fetch_nous_recommended_models(portal, force_refresh=True) == PAYLOAD
    assert paths == [f"/dynamic/{host}/api/nous/recommended-models"]


def test_a_portal_missing_from_the_mirror_is_asked_directly(paths, monkeypatch):
    from hermes_cli import models

    monkeypatch.setattr(models, "is_loopback_host", lambda _host: False)  # treat the local server as remote
    portal = models.NOUS_RECOMMENDED_MODELS_MIRROR.removesuffix("/dynamic")
    assert models.fetch_nous_recommended_models(portal, force_refresh=True) == PAYLOAD
    host = portal.removeprefix("http://")
    assert paths == [f"/dynamic/{host}/api/nous/recommended-models", "/api/nous/recommended-models"]


@pytest.mark.parametrize("auth_store, host", [
    (None, None),
    ({"providers": {"nous": {"auth_method": "anonymous", "access_token": "guest"}}}, None),
    ({"providers": {"nous": {"access_token": "a", "portal_base_url": "https://portal.staging-nousresearch.com"}}},
     "portal.staging-nousresearch.com"),
    ({"credential_pool": {"nous": [{"access_token": "a"}]}}, "portal.nousresearch.com"),
], ids=["no-account", "guest", "signed-in", "pooled"])
def test_gateway_refresh_fetches_recommendations_only_for_a_nous_account(paths, monkeypatch, auth_store, host):
    from hermes_cli import model_catalog, models

    monkeypatch.setattr(model_catalog, "get_catalog", lambda **_kw: {})
    monkeypatch.setattr(models, "fetch_openrouter_models", lambda **_kw: [])
    if auth_store is not None:
        (get_hermes_home() / "auth.json").write_text(json.dumps(auth_store), encoding="utf-8")

    model_catalog.refresh_catalogs()

    assert paths == ([f"/dynamic/{host}/api/nous/recommended-models"] if host else [])
