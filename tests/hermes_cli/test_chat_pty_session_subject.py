"""The dashboard chat PTY bootstraps as the login that opened it (#144140).

``_ws_auth_reason`` stamps the WS-upgrade login on the socket, but the PTY child
it spawns bootstrapped its gateway ticket anonymously: the control socket named
the ticket for the host UID, so ``validate_legacy_row`` refused a login-stamped
``desktop`` row the very same login created through the serve backend — the
dashboard listed a conversation it could never open. The login now rides to the
child as ``HERMES_TUI_SESSION_SUBJECT`` for the ticket mint's verified-peer
relabel (#138915). Only a real dashboard login is forwarded: the server-internal
credential and the identityless legacy-token path bootstrap exactly as before,
and a forwarded login still adopts only rows its own label owns.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.web_routers.chat_ws import _pty_session_subject_kwargs


def _ws(identity):
    return SimpleNamespace(_hermes_auth_identity=identity)


def test_a_login_ticket_is_forwarded_as_the_structured_subject():
    kwargs = _pty_session_subject_kwargs(_ws({"user_id": "david", "provider": "basic"}))
    assert kwargs == {"session_subject": 'auth:v1:["basic","","david"]'}


def test_an_oidc_login_keeps_its_issuer_in_the_subject():
    kwargs = _pty_session_subject_kwargs(_ws({"user_id": "user-a", "provider": "oidc",
                                              "issuer": "https://idp.example"}))
    assert kwargs == {"session_subject": 'auth:v1:["oidc","https://idp.example","user-a"]'}


def test_the_server_internal_credential_is_not_forwarded():
    assert _pty_session_subject_kwargs(_ws({"user_id": "server-internal",
                                            "provider": "server-internal"})) == {}


def test_an_identityless_upgrade_stays_anonymous():
    assert _pty_session_subject_kwargs(SimpleNamespace()) == {}


@pytest.fixture
def _chat_env(monkeypatch, _isolate_hermes_home):
    import hermes_cli.main_tui_launch as tui_launch
    import hermes_cli.web_server as ws

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(tui_launch, "_make_tui_argv",
                        lambda *_args, **_kwargs: (["node", "fake-tui.js"], Path("/tmp")))
    monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_port", 9119, raising=False)


def test_the_forwarded_subject_rides_to_the_child_env(_chat_env):
    from hermes_cli import web_server_chat

    _argv, _cwd, env = web_server_chat._resolve_chat_argv(
        session_subject='auth:v1:["basic","","david"]')

    assert env is not None
    assert env.get("HERMES_TUI_SESSION_SUBJECT") == 'auth:v1:["basic","","david"]'


def test_an_anonymous_chat_sets_no_subject(_chat_env):
    from hermes_cli import web_server_chat

    _argv, _cwd, env = web_server_chat._resolve_chat_argv()

    assert env is not None
    assert "HERMES_TUI_SESSION_SUBJECT" not in env
