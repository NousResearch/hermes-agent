"""Cross-process activation must pass gated auth without authorizing other routes (#141878)."""

from __future__ import annotations

import io
import secrets
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway import host_rendezvous
from hermes_cli import plugins, plugins_activation, web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE
from hermes_cli.dashboard_auth.token_auth import clear_token_routes, register_token_route
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

ACTIVATE = "/api/dashboard/agent-plugins/activate"


@pytest.fixture
def backend(tmp_path, monkeypatch):
    root = tmp_path / "hermes-home"
    secondary = root / "profiles" / "secondary"
    for home in (root, secondary):
        plugin = home / "plugins" / "activation-probe"
        plugin.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "plugins:\n  enabled: [activation-probe]\n", encoding="utf-8")
        (plugin / "plugin.yaml").write_text(
            "name: activation-probe\nversion: '1.0'\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(
            "from hermes_constants import get_hermes_home\n\n"
            "def register(ctx):\n"
            "    (get_hermes_home() / 'activation-marker').write_text('loaded', encoding='utf-8')\n",
            encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    token = secrets.token_urlsafe(32)
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_port", 9231, raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    plugins._reset_plugin_managers_for_tests()
    register_provider(StubAuthProvider())
    register_token_route("/api/gateway/drain")
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9231")

    def send_to_app(request, timeout=None):
        # Replace only the network transport. The notifier builds its real request,
        # and the app runs all middleware, route validation and plugin discovery.
        response = client.request(
            request.get_method(), request.full_url, content=request.data,
            headers=dict(request.header_items()), follow_redirects=False)
        if response.status_code >= 400:
            raise urllib.error.HTTPError(
                request.full_url, response.status_code, "refused", Message(),
                io.BytesIO(response.content))
        return io.BytesIO(response.content)

    monkeypatch.setattr(urllib.request, "urlopen", send_to_app)
    yield client, root, secondary, token
    client.close()
    plugins._reset_plugin_managers_for_tests()
    clear_providers()
    clear_token_routes()


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("role", [host_rendezvous.ROLE_SERVE, host_rendezvous.ROLE_DESKTOP_SERVE])
@pytest.mark.parametrize("profile", ["default", "secondary"])
def test_notifier_activates_only_the_requested_profile(backend, monkeypatch, gated, role, profile):
    _client, root, secondary, token = backend
    monkeypatch.setattr(web_server.app.state, "auth_required", gated)
    target, other = (root, secondary) if profile == "default" else (secondary, root)
    assert host_rendezvous.publish_record(role, host="127.0.0.1", port=9231, token=token)

    result = plugins_activation.notify_serve_backend("activation-probe", target)

    assert result is not None and result.get("ok") is True, result
    assert result["activation"]["name"] == "activation-probe"
    assert (target / "activation-marker").read_text(encoding="utf-8") == "loaded"
    assert not (other / "activation-marker").exists()


def test_process_credential_is_activation_only_and_dashboard_login_still_works(backend, tmp_path):
    client, root, secondary, token = backend
    body = {"name": "activation-probe", "home": str(root)}
    valid = {"X-Hermes-Session-Token": token}
    rejected_headers = [
        {}, {"X-Hermes-Session-Token": "stale-token"},
        {"Authorization": f"Bearer {token}"},
        [("X-Hermes-Session-Token", token), ("X-Hermes-Session-Token", token)],
        {**valid, "Origin": "https://example.test"}, {**valid, "Origin": ""},
    ]
    for headers in rejected_headers:
        assert client.post(ACTIVATE, json=body, headers=headers).status_code == 401
    assert client.post(ACTIVATE, json=body, params={"token": token}).status_code == 401
    for method, path in [("GET", ACTIVATE), ("PUT", ACTIVATE),
                         ("POST", "/api/env/reveal"), ("POST", "/api/gateway/drain")]:
        assert client.request(method, path, json={}, headers=valid).status_code == 401
    assert client.post("/api/gateway/drain", json={},
                       headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert not (root / "activation-marker").exists()
    assert not (secondary / "activation-marker").exists()

    assert client.post(ACTIVATE, json={**body, "name": ""}, headers=valid).status_code == 400
    assert client.post(ACTIVATE, json={**body, "home": str(tmp_path / "foreign")},
                       headers=valid).status_code == 400
    assert not (root / "activation-marker").exists()

    login = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert login.status_code == 302, login.text
    callback = client.get(login.headers["location"], follow_redirects=False)
    assert callback.status_code == 302
    response = client.post(ACTIVATE, json=body, headers={"Origin": "http://127.0.0.1:9231"})
    assert response.status_code == 200, response.text
    assert (root / "activation-marker").exists()
    assert not (secondary / "activation-marker").exists()
    access_token = client.cookies.get(SESSION_AT_COOKIE)
    client.cookies.clear()
    assert client.post(ACTIVATE, json=body,
                       headers={"Authorization": f"Bearer {access_token}"}).status_code == 200


def test_plugin_load_failure_is_visible_after_successful_gateway_reload(backend, monkeypatch):
    _client, root, _secondary, token = backend
    private_error = f"Fixture registration failed at {root} with token {token}"
    (root / "plugins" / "activation-probe" / "__init__.py").write_text(
        "from hermes_constants import get_hermes_home\n\n"
        "def register(ctx):\n"
        "    (get_hermes_home() / 'activation-marker').write_text('attempted', encoding='utf-8')\n"
        f"    raise RuntimeError({private_error!r})\n",
        encoding="utf-8")
    assert host_rendezvous.publish_record(
        host_rendezvous.ROLE_SERVE, host="127.0.0.1", port=9231, token=token)
    monkeypatch.setattr("gateway.control_socket.reload_gateway_plugins",
                        lambda *args, **kwargs: {"reloaded": True, "activations": []})

    response = plugins_activation.notify_serve_backend("activation-probe", root)
    result = plugins_activation.activate_plugin_now("activation-probe", in_process=False)
    hint = plugins_activation.activation_hint(result)

    assert (root / "activation-marker").read_text(encoding="utf-8") == "attempted"
    assert response is not None
    assert response == {"ok": False, "error": "Plugin did not load in the backend"}
    assert result["gateway_reloaded"] is True
    assert result["restart_required"] is False
    assert result["backend_error"] == response["error"]
    assert "backend activation failed" in hint and response["error"] in hint
    assert "nothing of this plugin needs" not in hint
    assert token not in str(response) + str(result) + hint
    assert str(root) not in str(response) + str(result) + hint


@pytest.mark.parametrize("gateway_reloaded", [False, True])
@pytest.mark.parametrize("failure, expected", [
    ("http", "HTTP 401"), ("timeout", "TimeoutError"), ("invalid-response", "ValueError")])
def test_backend_failure_is_visible_independently_of_gateway_reload(
        backend, monkeypatch, caplog, failure, expected, gateway_reloaded):
    _client, root, _secondary, token = backend
    assert plugins_activation.notify_serve_backend("activation-probe", root) is None
    assert host_rendezvous.publish_record(
        host_rendezvous.ROLE_SERVE, host="127.0.0.1", port=9231, token=token)

    def failed_request(request, timeout=None):
        if failure == "http":
            raise urllib.error.HTTPError(request.full_url, 401, token, Message(), io.BytesIO(token.encode()))
        if failure == "invalid-response":
            return io.BytesIO(b"[]")
        raise TimeoutError(token)

    monkeypatch.setattr(urllib.request, "urlopen", failed_request)
    monkeypatch.setattr("gateway.control_socket.reload_gateway_plugins",
                        lambda *args, **kwargs: {"reloaded": gateway_reloaded, "activations": []})
    result = plugins_activation.activate_plugin_now("activation-probe", in_process=False)
    hint = plugins_activation.activation_hint(result)

    assert result["gateway_reloaded"] is gateway_reloaded
    assert result["restart_required"] is not gateway_reloaded
    assert result["backend_error"] == expected
    assert "backend" in hint and expected in hint
    assert "nothing of this plugin needs" not in hint
    if not gateway_reloaded:
        assert "hermes gateway restart" in hint
    assert token not in str(result) + hint + caplog.text
