"""A pasted sign-in redirect completes only the connection it is pasted for.

Contract: ``POST /api/gateway-connections/<id>/oauth/complete`` with a redirect whose state belongs to
another connection's sign-in is refused (400) BEFORE the code is redeemed: no token request reaches
the remote, nothing is stored on either connection, and the other connection's flight still completes.
"""
import json

from fastapi.testclient import TestClient

from hermes_cli import gateway_connections as store
from hermes_cli import gateway_connections_oauth as oauth
from hermes_cli import web_server
from tests.hermes_cli import test_gateway_connections_oauth as oauth_tests

_save, remote = oauth_tests._save, oauth_tests.remote


def test_a_redirect_for_another_connection_is_refused_before_its_code_is_redeemed(remote, monkeypatch):
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    client = TestClient(web_server.app, raise_server_exceptions=False,
                        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN})
    first = _save(authMode="oauth")
    second = store.save_connection({"kind": "remote", "label": "Second", "url": "https://second.example",
                                    "authMode": "oauth"})[0]["id"]
    started = oauth.start_sign_in(first, bound_host="127.0.0.1", bound_port=9120)
    code, state = remote.issue_code(started["authorizeUrl"])
    pasted = f"http://127.0.0.1:9120{oauth.CALLBACK_PATH}?code={code}&state={state}"

    refused = client.post(f"/api/gateway-connections/{second}/oauth/complete", json={"url": pasted})
    assert refused.status_code == 400, refused.text
    assert not [p for p, _ in remote.calls if p == "/auth/native/token"]
    assert not store.tokens_path().exists() or json.loads(store.tokens_path().read_text())["tokens"] == {}

    done = client.post(f"/api/gateway-connections/{first}/oauth/complete", json={"url": pasted})
    assert done.status_code == 200 and done.json()["connectionId"] == first
