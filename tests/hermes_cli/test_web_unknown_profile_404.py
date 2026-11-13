"""An unknown ``?profile=`` is the profile scope's 404 on every dashboard route,
never a generic 500 from a handler's catch-all."""

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(_isolate_hermes_home):
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    c = TestClient(app, raise_server_exceptions=False)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


@pytest.mark.parametrize("method,path,body", [
    ("PUT", "/api/env", {"key": "FAKE_PROBE_KEY", "value": "fake-value"}),
    ("GET", "/api/learning/graph", None),
    ("GET", "/api/dashboard/plugins/hub", None),
    ("GET", "/api/cron/delivery-targets", None),
    ("GET", "/api/cron/blueprints", None),
    ("GET", "/api/model/recommended-default", None),
    ("GET", "/api/audio/voice-config", None),
    ("GET", "/api/skills/hub/official", None),
])
def test_unknown_profile_is_the_scopes_404(client, method, path, body):
    resp = client.request(method, path, params={"profile": "no-such-profile"}, json=body)

    assert resp.status_code == 404, resp.text
    assert "no-such-profile" in resp.json()["detail"]
