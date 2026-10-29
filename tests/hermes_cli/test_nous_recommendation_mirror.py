"""Nous recommendations come from each Portal's static mirror (the Portal itself when the mirror has
no copy or stalls), and the gateway's background refresh fetches them only for a profile with a Nous
account."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_constants import get_hermes_home

PAYLOAD = {"paidRecommendedModels": [{"modelName": "mirrored"}]}


@pytest.fixture
def server(monkeypatch):
    """A local server standing in for the mirror (which holds no copy for a 127.0.0.1 Portal, or stalls
    on it when ``mirror_stalls`` is set) and for a Portal at its own origin, recording the paths asked."""
    state = {"seen": [], "mirror_stalls": False}
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["seen"].append(self.path)
            if self.path.startswith("/dynamic/127.0.0.1") and state["mirror_stalls"]:
                release.wait(30)
                return
            self.send_response(404 if self.path.startswith("/dynamic/127.0.0.1") else 200)
            self.end_headers()
            self.wfile.write(json.dumps(PAYLOAD).encode())

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    from hermes_cli import models
    monkeypatch.setattr(models, "NOUS_RECOMMENDED_MODELS_MIRROR", f"http://127.0.0.1:{httpd.server_port}/dynamic")
    yield state
    release.set()
    httpd.shutdown()
    httpd.server_close()


@pytest.mark.parametrize("portal, host", [
    ("", "portal.nousresearch.com"),
    ("https://portal.staging-nousresearch.com/", "portal.staging-nousresearch.com"),
])
def test_recommendations_come_from_the_portal_hosts_mirror(server, portal, host):
    from hermes_cli.models import fetch_nous_recommended_models

    assert fetch_nous_recommended_models(portal, force_refresh=True) == PAYLOAD
    assert server["seen"] == [f"/dynamic/{host}/api/nous/recommended-models"]


@pytest.mark.parametrize("mirror_stalls", [False, True], ids=["mirror-404", "mirror-stalls"])
def test_a_portal_the_mirror_cannot_serve_is_asked_directly(server, monkeypatch, mirror_stalls):
    from hermes_cli import models

    server["mirror_stalls"] = mirror_stalls
    monkeypatch.setattr(models, "is_loopback_host", lambda _host: False)  # treat the local server as remote
    portal = models.NOUS_RECOMMENDED_MODELS_MIRROR.removesuffix("/dynamic")
    started = time.monotonic()
    assert models.fetch_nous_recommended_models(portal, timeout=10.0, force_refresh=True) == PAYLOAD
    # A stalled mirror costs its own short timeout, not the caller's whole budget.
    assert time.monotonic() - started < 6.0
    host = portal.removeprefix("http://")
    assert server["seen"] == [f"/dynamic/{host}/api/nous/recommended-models", "/api/nous/recommended-models"]


@pytest.mark.parametrize("auth_store, env_key, host", [
    (None, None, None),
    ({"providers": {"nous": {"auth_method": "anonymous", "access_token": "guest"}}}, None, None),
    ({"providers": {"nous": {"access_token": "a", "portal_base_url": "https://portal.staging-nousresearch.com"}}},
     None, "portal.staging-nousresearch.com"),
    ({"credential_pool": {"nous": [{"access_token": "a"}]}}, None, "portal.nousresearch.com"),
    (None, "sk-nous-test", "portal.nousresearch.com"),
], ids=["no-account", "guest", "signed-in", "pooled", "api-key-env"])
def test_gateway_refresh_fetches_recommendations_only_for_a_nous_account(server, monkeypatch, auth_store, env_key, host):
    from hermes_cli import model_catalog, models

    monkeypatch.setattr(model_catalog, "get_catalog", lambda **_kw: {})
    monkeypatch.setattr(models, "fetch_openrouter_models", lambda **_kw: [])
    if auth_store is not None:
        (get_hermes_home() / "auth.json").write_text(json.dumps(auth_store), encoding="utf-8")
    if env_key is not None:
        monkeypatch.setenv("NOUS_API_KEY", env_key)

    model_catalog.refresh_catalogs()

    assert server["seen"] == ([f"/dynamic/{host}/api/nous/recommended-models"] if host else [])
