"""On the gateway's Webapp behind the sign-in gate, the deployment session token works on its own.

``HERMES_DASHBOARD_SESSION_TOKEN`` (a literal or ``file:<path>``) registers the bundled session-token
provider, which vouches for that token as a bearer and as a socket's ``?token=``. The Webapp's
listener takes it there too, while a wrong token, a browser login's tokens and the refresh route keep
main's rules, and with the provider turned off (``dashboard.session_token_auth.enabled: false``) the
token is refused as on main. The token counts as the one the provider read at startup: a credential
change takes effect on restart.
"""
from types import SimpleNamespace

import pytest

import plugins.dashboard_auth.session_token as session_token_plugin
from tests.hermes_cli import test_webapp_gated_sign_out as webapp_gate
from tests.hermes_cli.test_webapp_gated_sign_out import (
    _bearer, _gateway_socket_with, _native_app_sign_in, _renew, _refused)

gated = webapp_gate.gated  # the Webapp behind the sign-in gate, password and OAuth providers registered

DEPLOYMENT_TOKEN = "deployment-session-token-" + "d" * 32


def _provision(monkeypatch, tmp_path, *, form="literal", enabled=True):
    """Provision the deployment token and run the plugin's own registration, as plugin discovery does."""
    from hermes_agent.hermes_cli.dashboard_auth import register_provider

    if form == "file":
        secret = tmp_path / "session-token"
        secret.write_text(DEPLOYMENT_TOKEN + "\n")
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", f"file:{secret}")
    else:
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", DEPLOYMENT_TOKEN)
    monkeypatch.setattr(session_token_plugin, "_load_session_token_auth_section",
                        lambda: {} if enabled else {"enabled": False})
    session_token_plugin.register(SimpleNamespace(register_dashboard_auth_provider=register_provider))


@pytest.mark.parametrize("form", ["literal", "file"])
def test_the_deployment_token_works_on_its_own_on_the_webapps_listener(gated, monkeypatch, tmp_path, form):
    _provision(monkeypatch, tmp_path, form=form)
    me = gated.browser().get("/api/auth/me", headers=_bearer(DEPLOYMENT_TOKEN))
    assert me.status_code == 200, me.text
    assert me.json()["provider"] == "session-token"
    assert _gateway_socket_with(gated, DEPLOYMENT_TOKEN) is None


def test_a_wrong_token_is_still_refused(gated, monkeypatch, tmp_path):
    _provision(monkeypatch, tmp_path)
    wrong = "not-the-deployment-session-token-" + "x" * 32
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(wrong)))
    assert _gateway_socket_with(gated, wrong) == 4401


def test_with_the_provider_turned_off_the_token_is_refused_as_on_main(gated, monkeypatch, tmp_path):
    _provision(monkeypatch, tmp_path, enabled=False)
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(DEPLOYMENT_TOKEN)))
    assert _gateway_socket_with(gated, DEPLOYMENT_TOKEN) == 4401


def test_native_app_tokens_and_their_refresh_keep_mains_rules(gated, monkeypatch, tmp_path):
    _provision(monkeypatch, tmp_path)
    access, refresh = _native_app_sign_in(gated)
    assert gated.browser().get("/api/auth/me", headers=_bearer(access)).status_code == 200
    assert _renew(gated, refresh).status_code == 200


def test_the_refresh_route_refuses_the_deployment_token_before_asking_any_provider(gated, monkeypatch, tmp_path):
    """The deployment token has no refresh: the route answers main's 401 and audit reason without
    calling a provider, so an unreachable one cannot turn it into a 503."""
    from hermes_agent.hermes_cli.dashboard_auth import ProviderError, register_provider
    from hermes_agent.hermes_cli.dashboard_auth import routes
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    calls, audits = [], []

    class Unreachable(StubAuthProvider):
        name = "unreachable"

        def refresh_session(self, *, refresh_token):
            calls.append(refresh_token)
            raise ProviderError("unreachable")

    _provision(monkeypatch, tmp_path)
    register_provider(Unreachable())
    monkeypatch.setattr(routes, "audit_log", lambda event, **fields: audits.append((event, fields.get("reason"))))
    _refused(_renew(gated, DEPLOYMENT_TOKEN))
    assert calls == []
    assert audits == [(routes.AuditEvent.REFRESH_FAILURE, "not_a_native_sign_in")]


def test_the_refresh_route_still_goes_through_token_works_alone(gated, monkeypatch):
    """The refresh route's check is ``webapp_sign_in.token_works_alone``, as on main, so patching it
    refuses even a native app's legitimate refresh."""
    from hermes_agent.hermes_cli import webapp_sign_in

    _access, refresh = _native_app_sign_in(gated)
    monkeypatch.setattr(webapp_sign_in, "token_works_alone", lambda request, token: False)
    _refused(_renew(gated, refresh))


def test_a_rewritten_token_file_takes_effect_on_restart(gated, monkeypatch, tmp_path):
    """The Webapp takes the token the provider read at startup, not what the file holds now."""
    _provision(monkeypatch, tmp_path, form="file")
    rotated = "rotated-deployment-session-token-" + "r" * 32
    (tmp_path / "session-token").write_text(rotated + "\n")
    assert gated.browser().get("/api/auth/me", headers=_bearer(DEPLOYMENT_TOKEN)).status_code == 200
    _refused(gated.browser().get("/api/auth/me", headers=_bearer(rotated)))
    assert _gateway_socket_with(gated, rotated) == 4401
