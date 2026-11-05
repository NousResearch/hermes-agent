"""A browser's page load during a drain or after a failed Webapp rebuild gets a page, not JSON.

`hermes update` drains the gateway that serves a Webapp tab; a tab reloaded in that window used to show
the bare ``{"error":"gateway_not_ready"}`` body as the whole page, and after an update whose rebuild
failed ``/`` showed the stale-bundle reason the same way. API callers keep the JSON contract.
"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

PAGE_LOAD = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


async def _drained(method, path, headers):
    from gateway.run_api import GatewayRuntimeAPI

    reached = []

    async def app(scope, receive, send):
        reached.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"app"})

    runner = SimpleNamespace(_draining=True, _running=True,
                             session_runtime_descriptor={"state": "ready", "drain_reason": "restart_wait"})
    api = GatewayRuntimeAPI(app, runner, SimpleNamespace(state=None))
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)
    await api({"type": "http", "method": method, "path": path, "query_string": b"", "client": ("127.0.0.1", 1),
               "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}, receive, send)
    start = sent[0]
    return (start["status"], dict(start["headers"]).get(b"content-type", b"").decode(),
            b"".join(m.get("body", b"") for m in sent[1:]).decode(), dict(start["headers"]), reached)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/", "/sessions/abc", "/hermes/"])
async def test_a_page_load_during_the_drain_gets_the_restart_page(path):
    status, ctype, body, headers, reached = await _drained("GET", path, PAGE_LOAD)
    assert status == 503 and ctype.startswith("text/html")
    assert "Hermes is restarting" in body and "gateway_not_ready" not in body
    assert headers[b"retry-after"] == b"2" and b"no-store" in headers[b"cache-control"]
    assert headers[b"x-hermes-restart-page"] == b"restarting"
    assert reached == []  # still refused: the app never runs on a draining owner


@pytest.mark.asyncio
async def test_the_page_runs_only_its_own_hashed_script_and_polls_the_app_root_not_its_url():
    """A page load with a one-time effect (an OAuth callback's code) must run once, on the reload:
    the poll asks for the root under the same proxy prefix, never this URL."""
    import base64
    import hashlib
    import re

    _, _, body, headers, _ = await _drained("GET", "/auth/callback", PAGE_LOAD)
    script = re.search(r"<script>(.*)</script>", body, re.DOTALL).group(1)
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    csp = headers[b"content-security-policy"].decode()
    assert f"script-src 'sha256-{digest}';" in csp and "script-src 'unsafe-inline'" not in csp
    assert 's="/auth/callback"' in script and "fetch(base+'/'" in script and "location.href" not in script


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,headers", [
    ("GET", "/api/config", PAGE_LOAD),           # an API route stays JSON even if the caller names HTML
    ("GET", "/hermes/api/config", PAGE_LOAD),    # ... also behind a base path
    ("GET", "/", {"accept": "application/json"}),  # fetch/XHR of a document path
    ("GET", "/", {}),                            # no Accept at all (curl, probes)
    ("POST", "/", PAGE_LOAD),                    # never a page for a write
])
async def test_everything_else_keeps_the_json_refusal(method, path, headers):
    status, ctype, body, _, reached = await _drained(method, path, headers)
    assert (status, ctype) == (503, "application/json")
    assert '"error":"gateway_not_ready"' in body and reached == []


def _webapp(monkeypatch, tmp_path, *, stale):
    from hermes_cli import web_server as server
    from hermes_cli.web_server_dashboard import mount_spa
    from hermes_cli.webapp import STALE_MARKER

    (tmp_path / "index.html").write_text("<!doctype html><html><head></head><body>app</body></html>")
    if stale:
        (tmp_path / STALE_MARKER).write_text("x")
    app = FastAPI()
    app.state.ui_surface, app.state.web_dist = "webapp", tmp_path
    monkeypatch.setattr(server, "app", app)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(app)
    return TestClient(app)


def test_a_page_load_of_a_bundle_the_update_could_not_rebuild_gets_the_rebuild_page(monkeypatch, tmp_path):
    from gateway.webapp_listener import NOT_REBUILT

    client = _webapp(monkeypatch, tmp_path, stale=True)
    page = client.get("/", headers=PAGE_LOAD)
    assert page.status_code == 503 and page.headers["content-type"].startswith("text/html")
    assert "hermes webapp" in page.text and page.headers["x-hermes-restart-page"] == "not_rebuilt"
    api = client.get("/", headers={"accept": "application/json"})
    assert api.status_code == 503 and api.json() == {"error": NOT_REBUILT}


def test_a_current_bundle_is_served_as_before(monkeypatch, tmp_path):
    page = _webapp(monkeypatch, tmp_path, stale=False).get("/", headers=PAGE_LOAD)
    assert page.status_code == 200 and "app" in page.text
