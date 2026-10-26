"""The dashboard chat PTY bootstraps as the login that opened it (#144140).

``_ws_auth_reason`` stamps the WS-upgrade login on the socket, but the PTY child
it spawns bootstrapped its gateway ticket anonymously: the control socket named
the ticket for the host UID, so ``validate_legacy_row`` refused a login-stamped
``desktop`` row the very same login created through the serve backend — the
dashboard listed a conversation it could never open. The login now rides to the
child as ``HERMES_TUI_SESSION_SUBJECT`` for the ticket mint's verified-peer
relabel (#138915). The resume then adopts unowned legacy rows and the rows the
login's own label stamped, never another login's or the host UID's; that label
drops the issuer, so a self-hosted OIDC login's serve rows stay refused. Only a
browser ticket's login is forwarded, and only to a child that mints its own
ticket: the ``?token=`` leg (it stamps no issuer) and a subject the mint would
refuse bootstrap exactly as before, and an unscoped chat attaches to the
dashboard's in-memory gateway on the server's credential.
"""

import time
from pathlib import Path

import pytest

from hermes_cli import web_server
import hermes_cli.web_server_chat as _web_server_chat
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests, mint_ticket
from hermes_cli.web_routers.chat_ws import _pty_session_subject
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider, _sign
from tests.hermes_cli.test_dashboard_auth_ws_auth import _fake_ws

_LOGIN = 'auth:v1:["basic","","david"]'
# The two ways a browser ticket opens a socket, by the credential name ``_ws_auth_reason`` gives each.
_PRESENTED = {
    "ticket": lambda ticket: _fake_ws(query={"ticket": ticket}),
    "ticket-subprotocol": lambda ticket: _fake_ws(query={}, protocols=(
        _web_server_chat._GATEWAY_WS_PROTOCOL, _web_server_chat._GATEWAY_WS_TICKET_PROTOCOL_PREFIX + ticket)),
}


@pytest.fixture
def gated(monkeypatch):
    """The standalone dashboard behind its sign-in gate, with a provider that verifies ``?token=``."""
    _reset_for_tests()
    clear_providers()
    register_provider(StubAuthProvider())
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "dashboard", raising=False)
    yield
    clear_providers()
    _reset_for_tests()


def _forwarded(ws):
    """``(credential, forwarded subject)`` for a ``/api/pty`` upgrade the gate admits."""
    reason, cred = _web_server_chat._ws_auth_reason(ws, socket="pty")
    assert reason is None
    return cred, _pty_session_subject(ws, cred)


@pytest.mark.parametrize("cred", sorted(_PRESENTED))
def test_a_login_ticket_is_forwarded_as_the_structured_subject(gated, cred):
    ws = _PRESENTED[cred](mint_ticket(user_id="david", provider="basic"))
    assert _forwarded(ws) == (cred, _LOGIN)


def test_an_oidc_login_keeps_its_issuer_in_the_subject(gated):
    """The ticket ``POST /api/auth/ws-ticket`` mints carries the issuer: the chat names the subject
    the dashboard's owner forward names for that login (``_session_mutation_context``). Serve's
    ``oidc:user-a`` stamp drops the issuer, so this login's serve rows stay refused to it
    (``issuer_alice``, tests/gateway/test_legacy_desktop_source.py) until serve stamps the issuer."""
    ws = _fake_ws(query={"ticket": mint_ticket(user_id="user-a", provider="oidc",
                                               issuer="https://idp.example")})
    assert _forwarded(ws) == ("ticket", 'auth:v1:["oidc","https://idp.example","user-a"]')


def test_a_subject_the_ticket_mint_refuses_is_not_forwarded(gated):
    """The session-ticket mint refuses a subject over ``MAX_TICKET_SUBJECT_CHARS``: naming one would
    leave that chat failing to bootstrap on every retry, so such a login keeps bootstrapping as the
    OS user (an OIDC ``sub`` may run to 255 characters)."""
    ws = _fake_ws(query={"ticket": mint_ticket(user_id="u" * 255, provider="oidc",
                                               issuer="https://idp.example")})
    assert _forwarded(ws) == ("ticket", None)


def test_a_provider_token_login_is_not_forwarded(gated):
    """The ``?token=`` leg stamps its login without the issuer, so an issuer-bearing login would
    spell another login's issuer-less ``<provider>:<user>`` stamp (``login_owned_by``): its chat
    keeps bootstrapping as the OS user."""
    token = _sign({"sub": "user-a", "email": "a@example.test", "name": "A", "org_id": "org-1",
                   "exp": int(time.time()) + 3600})
    ws = _fake_ws(query={"token": token})
    assert _forwarded(ws) == ("token", None)
    assert ws._hermes_auth_identity == {"user_id": "user-a", "provider": "stub"}


@pytest.fixture
def _chat_env(monkeypatch, _isolate_hermes_home):
    """A loopback dashboard (its in-memory gateway at ``/api/ws``) with one named profile, ``alpha``."""
    import hermes_cli.main_tui_launch as tui_launch
    from hermes_cli import profiles
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    (home / "profiles" / "alpha").mkdir(parents=True)
    (home / "profiles" / "alpha" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: home / "profiles")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(tui_launch, "_make_tui_argv",
                        lambda *_args, **_kwargs: (["node", "fake-tui.js"], Path("/tmp")))
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_port", 9119, raising=False)


def test_the_forwarded_subject_rides_to_the_child_env(_chat_env):
    """A profile-scoped chat mints its own ticket on that profile's gateway: it carries the login."""
    _argv, _cwd, env = _web_server_chat._resolve_chat_argv(profile="alpha", session_subject=_LOGIN)

    assert env is not None
    assert env.get("HERMES_TUI_SESSION_SUBJECT") == _LOGIN
    assert "HERMES_TUI_GATEWAY_URL" not in env


def test_an_unscoped_chat_attaches_on_the_servers_credential_and_names_nobody(_chat_env):
    _argv, _cwd, env = _web_server_chat._resolve_chat_argv(session_subject=_LOGIN)

    assert env is not None
    assert env["HERMES_TUI_GATEWAY_URL"].startswith("ws://127.0.0.1:9119/api/ws?")
    assert "HERMES_TUI_SESSION_SUBJECT" not in env


@pytest.mark.parametrize("profile", [None, "alpha"])
def test_an_anonymous_chat_sets_no_subject(_chat_env, monkeypatch, profile):
    """Not even one the dashboard's own environment carries: only the signed-in login names it."""
    monkeypatch.setenv("HERMES_TUI_SESSION_SUBJECT", 'auth:v1:["basic","","inherited"]')

    _argv, _cwd, env = _web_server_chat._resolve_chat_argv(profile=profile)

    assert env is not None
    assert "HERMES_TUI_SESSION_SUBJECT" not in env
