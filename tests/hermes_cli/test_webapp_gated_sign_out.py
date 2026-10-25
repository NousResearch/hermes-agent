"""Behind the sign-in gate, signing the Webapp out signs every browser out.

On a LAN or own-URL Webapp a browser signs in with the dashboard's sign-in (a password, or an OAuth
provider) and keeps that login in cookies. Each login on the Webapp is bound to the gateway's browser
credential, so ``hermes webapp --sign-out``, which replaces that credential, ends every one of them:
the login's requests get 401 and the sign-in view, the sockets and host terminals it opened close with
4401, tickets it minted open nothing, and its cookies replayed from anywhere get nothing. The provider
tokens those cookies hold work only as those cookies: the Webapp's listener refuses them as a bearer
token, on the refresh route and as a socket's ``?token=``, before and after sign-out and across a
restart. Signing in again works, and a native app's own sign-in on the same listener (and the sockets
it opens) stays signed in. The standalone dashboard has no
browser credential and keeps its logins as on main.
"""
import json
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.hermes_cli.test_web_host_terminal_persistence import metadata
from tests.hermes_cli.test_webapp_sign_out_sockets import _close_code
from tests.hermes_cli.test_webapp_socket_tickets import _TOKEN, serve_webapp

WEBAPP = "http://hermes.lan:9120"
USER, PASSWORD = "owner", "throwaway-test-password"


@pytest.fixture
def gated(tmp_path, monkeypatch):
    """The Webapp behind the sign-in gate with a password and an OAuth provider registered."""
    from hermes_cli import webapp_sign_in
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit
    from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    webapp = serve_webapp(tmp_path, monkeypatch)
    state = webapp.server.app.state
    monkeypatch.setattr(state, "bound_host", "hermes.lan")
    monkeypatch.setattr(state, "bound_port", 9120, raising=False)
    monkeypatch.setattr(state, "auth_required", True)
    monkeypatch.setattr(state, "webapp_home", tmp_path / ".hermes", raising=False)
    _reset_for_tests()
    webapp_sign_in._reset_for_tests()
    _reset_password_rate_limit()
    clear_providers()
    register_provider(BasicAuthProvider(username=USER, password_hash=hash_password(PASSWORD), secret=b"k" * 32))
    register_provider(StubAuthProvider())
    webapp.browser = lambda: TestClient(webapp.server.app, base_url=WEBAPP)
    try:
        yield webapp
    finally:
        clear_providers()
        _reset_for_tests()
        webapp_sign_in._reset_for_tests()
        _reset_password_rate_limit()


def _sign_in_with_password(client):
    signed_in = client.post("/auth/password-login", json={"provider": "basic", "username": USER, "password": PASSWORD})
    assert signed_in.status_code == 200, signed_in.text


def _sign_in_with_oauth(client):
    started = client.get("/auth/login?provider=stub", follow_redirects=False)
    state = started.headers["location"].split("state=")[1]
    assert client.get(f"/auth/callback?code=stub_code&state={state}", follow_redirects=False).status_code == 302


SIGN_INS = {"password": _sign_in_with_password, "oauth": _sign_in_with_oauth}


@pytest.fixture
def sign_out(gated, tmp_path):
    """Ask the gateway serving *gated* for ``webapp-sign-out`` over its control socket."""
    from gateway.control_socket import GatewayControlServer
    from gateway.webapp_listener import register_control_verbs

    home = tmp_path / ".hermes"
    runner = SimpleNamespace(
        session_control_server=GatewayControlServer(home),
        webapp_status={"state": "serving", "url": WEBAPP, "reason": None},
        session_api=SimpleNamespace(app=SimpleNamespace(web=gated.server)))
    register_control_verbs(runner, home)

    def ask():
        raw = json.dumps({"protocol": 1, "verb": "webapp-sign-out", "id": 1, "params": {}}).encode()
        answer = json.loads(runner.session_control_server.handle_request_line(raw, "same-user"))
        assert answer["ok"], answer
        assert gated.server._SESSION_TOKEN != _TOKEN
        return answer["result"]
    return ask


def _signed_out(response):
    assert response.status_code == 401, response.text
    body = response.json()
    assert body["error"] in {"unauthenticated", "session_expired"} and body["login_url"].startswith("/login")


@pytest.mark.parametrize("sign_in", SIGN_INS)
def test_signing_out_ends_every_login_behind_the_gate(gated, sign_out, sign_in):
    browser = gated.browser()
    SIGN_INS[sign_in](browser)
    assert browser.get("/api/auth/me").status_code == 200
    sign_out()
    _signed_out(browser.get("/api/auth/me"))
    _signed_out(browser.post("/api/webapp/socket-ticket?socket=host-terminal"))
    _signed_out(browser.post("/api/auth/ws-ticket"))
    page = browser.get("/", follow_redirects=False)
    assert page.status_code == 302 and page.headers["location"].startswith("/login")


def test_old_login_cookies_replayed_after_sign_out_get_nothing(gated, sign_out):
    browser = gated.browser()
    _sign_in_with_password(browser)
    stolen = dict(browser.cookies.items())
    sign_out()
    replay = gated.browser()
    replay.cookies.update(stolen)
    _signed_out(replay.get("/api/auth/me"))


def test_signing_in_again_after_sign_out_works(gated, sign_out):
    browser = gated.browser()
    _sign_in_with_password(browser)
    sign_out()
    _signed_out(browser.get("/api/auth/me"))
    _sign_in_with_password(browser)
    assert browser.get("/api/auth/me").status_code == 200
    assert browser.post("/api/webapp/socket-ticket?socket=host-terminal").status_code == 200


def test_a_credential_replaced_while_the_gateway_was_down_signs_logins_out(gated, monkeypatch):
    """The CLI replaces the stored credential when no gateway is running; the next gateway serves
    the new one, and logins bound to the old one are signed out."""
    browser = gated.browser()
    _sign_in_with_password(browser)
    monkeypatch.setattr(gated.server, "_SESSION_TOKEN", "n" * 43)
    _signed_out(browser.get("/api/auth/me"))


def test_signing_out_closes_the_sockets_and_host_terminal_a_login_opened(gated, sign_out):
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket

    browser = gated.browser()
    _sign_in_with_password(browser)
    terminal = browser.post("/api/webapp/socket-ticket?socket=host-terminal").json()["ticket"]
    events = browser.post("/api/auth/ws-ticket").json()["ticket"]
    other = mint_ticket(user_id="someone-else", provider="stub")
    with TestClient(gated.host, base_url=WEBAPP) as shells, \
            shells.websocket_connect(f"ws://hermes.lan:9120/api/host-terminal?ticket={terminal}") as shell, \
            browser.websocket_connect(f"ws://hermes.lan:9120/api/events?channel=tab-1&ticket={events}") as stream, \
            browser.websocket_connect(f"ws://hermes.lan:9120/api/events?channel=kept&ticket={other}") as kept:
        metadata(shell)
        assert _close_code(shell, 0.2) is None and _close_code(stream, 0.2) is None
        result = sign_out()
        assert _close_code(shell) == 4401
        assert _close_code(stream) == 4401
        assert _close_code(kept, 0.5) is None
    assert result["connections_closed"] == 2
    assert result["sign_in_gate"] is True


def test_tickets_a_login_minted_before_sign_out_open_nothing_after(gated, sign_out):
    browser = gated.browser()
    _sign_in_with_password(browser)
    terminal = browser.post("/api/webapp/socket-ticket?socket=host-terminal").json()["ticket"]
    events = browser.post("/api/auth/ws-ticket").json()["ticket"]
    sign_out()
    with TestClient(gated.host, base_url=WEBAPP) as shells:
        for client, url in ((shells, f"ws://hermes.lan:9120/api/host-terminal?ticket={terminal}"),
                            (browser, f"ws://hermes.lan:9120/api/events?channel=tab-1&ticket={events}")):
            with pytest.raises(WebSocketDisconnect) as closed:
                with client.websocket_connect(url) as ws:
                    ws.receive_text()
            assert closed.value.code == 4401
    assert gated.shells == []


def test_the_standalone_dashboard_keeps_its_logins_as_on_main(gated, monkeypatch):
    monkeypatch.setattr(gated.server.app.state, "ui_surface", "dashboard")
    browser = gated.browser()
    _sign_in_with_password(browser)
    assert not any("hermes_webapp_signin" in name for name in browser.cookies.keys())
    monkeypatch.setattr(gated.server, "_SESSION_TOKEN", "n" * 43)
    assert browser.get("/api/auth/me").status_code == 200


def test_two_gateways_on_one_hostname_keep_their_own_sign_ins(gated, monkeypatch, tmp_path):
    """Cookies are not scoped to a port, so one browser sends the Webapps of two gateways on one host
    the same cookies. Signing in to one leaves the other signed in (here both accept the same account
    tokens, as two gateways sharing a sign-in provider do), and signing one out leaves the other."""
    state = gated.server.app.state
    gateways = {9120: (tmp_path / ".hermes", _TOKEN), 9121: (tmp_path / "other-home", "o" * 43)}
    browser = gated.browser()  # one cookie jar for both

    def at(port, path):
        home, token = gateways[port]
        monkeypatch.setattr(state, "webapp_home", home)
        monkeypatch.setattr(state, "bound_port", port)
        monkeypatch.setattr(gated.server, "_SESSION_TOKEN", token)
        return f"http://hermes.lan:{port}{path}"

    for port in (9120, 9121):
        signed_in = browser.post(at(port, "/auth/password-login"),
                                 json={"provider": "basic", "username": USER, "password": PASSWORD})
        assert signed_in.status_code == 200
    assert browser.get(at(9121, "/api/auth/me")).status_code == 200
    assert browser.get(at(9120, "/api/auth/me")).status_code == 200
    gateways[9121] = (gateways[9121][0], "r" * 43)  # the second gateway signs its browsers out
    _signed_out(browser.get(at(9121, "/api/auth/me")))
    assert browser.get(at(9120, "/api/auth/me")).status_code == 200


def test_two_gateways_with_their_own_sign_in_on_one_hostname_keep_their_logins_apart(gated, monkeypatch, tmp_path):
    """The Webapp's listener names every sign-in cookie it sets for its gateway (the login's access and
    refresh token, its provider hint and binding, the OAuth state and the auto sign-in marker), so two
    gateways on one hostname whose sign-in providers differ (two users' or two profiles' gateways, each
    with its own provider secret) keep their own logins in one browser: signing in to one leaves the
    other signed in, and signing out of one in the browser leaves the other."""
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password

    state = gated.server.app.state
    providers = {port: BasicAuthProvider(username=USER, password_hash=hash_password(PASSWORD), secret=secret * 32)
                 for port, secret in ((9120, b"1"), (9121, b"2"))}
    gateways = {9120: (tmp_path / ".hermes", _TOKEN), 9121: (tmp_path / "other-home", "o" * 43)}
    browser = gated.browser()  # one cookie jar for both

    def at(port, path):
        home, token = gateways[port]
        monkeypatch.setattr(state, "webapp_home", home)
        monkeypatch.setattr(state, "bound_port", port)
        monkeypatch.setattr(gated.server, "_SESSION_TOKEN", token)
        clear_providers()
        register_provider(providers[port])
        return f"http://hermes.lan:{port}{path}"

    for port in (9120, 9121):
        signed_in = browser.post(at(port, "/auth/password-login"),
                                 json={"provider": "basic", "username": USER, "password": PASSWORD})
        assert signed_in.status_code == 200
    assert browser.get(at(9121, "/api/auth/me")).status_code == 200
    first = browser.get(at(9120, "/api/auth/me"))
    assert first.status_code == 200, f"signing in to the second gateway signed the first out: {first.text}"
    assert browser.post(at(9121, "/auth/logout"), headers={"Origin": "http://hermes.lan:9121"},
                        follow_redirects=False).status_code == 302
    _signed_out(browser.get(at(9121, "/api/auth/me")))
    assert browser.get(at(9120, "/api/auth/me")).status_code == 200


def test_the_webapp_leaves_cookies_named_for_no_gateway_to_the_dashboard(gated, monkeypatch):
    """The standalone dashboard on the same hostname keeps main's cookie names, as did earlier builds
    of the Webapp. The Webapp's listener neither reads nor clears them: a browser holding only such a
    login gets the sign-in view there, and the dashboard keeps it."""
    monkeypatch.setattr(gated.server.app.state, "ui_surface", "dashboard")
    browser = gated.browser()
    _sign_in_with_password(browser)
    names = set(browser.cookies.keys())
    assert {"hermes_session_at", "hermes_session_rt", "hermes_session_provider"} <= names
    monkeypatch.setattr(gated.server.app.state, "ui_surface", "webapp")
    on_webapp = browser.get("/api/auth/me")
    _signed_out(on_webapp)
    assert "hermes_session_at=" not in on_webapp.headers.get("set-cookie", "")
    signed_out = browser.post("/auth/logout", headers={"Origin": WEBAPP}, follow_redirects=False)
    assert signed_out.status_code == 302
    assert "hermes_session_at=" not in signed_out.headers.get("set-cookie", "")
    monkeypatch.setattr(gated.server.app.state, "ui_surface", "dashboard")
    assert browser.get("/api/auth/me").status_code == 200


# A browser's login keeps the sign-in provider's access and refresh token in its cookies. Copied out of
# them they are the kind of token a native app presents as ``Authorization: Bearer`` and renews on
# ``/auth/native/refresh``. On the Webapp's listener they work only as that login's cookies, so signing
# out ends them everywhere there; a native app's own sign-in on the same listener stays signed in.

@pytest.fixture
def clock(monkeypatch):
    """Moves the password provider's clock forward. It signs each token's expiry in whole seconds, so
    without this a renewal or a second login in the same second gets byte-identical tokens."""
    import time

    from plugins.dashboard_auth import basic

    offset = [0]
    monkeypatch.setattr(basic, "time", SimpleNamespace(time=lambda: time.time() + offset[0]))

    def advance(seconds=5):
        offset[0] += seconds
    return advance


def _cookie(client, bare):
    """The name of *client*'s cookie *bare*: on the Webapp's listener followed by its gateway, elsewhere
    as on main."""
    return next(name for name in client.cookies.keys() if name == bare or name.startswith(f"{bare}_"))


def _tokens(client):
    """The access and refresh token the cookie login on *client* holds, as a copy taken out of the
    browser would present them (the cookie keeps them in quotes)."""
    return tuple(client.cookies.get(_cookie(client, name)).strip('"')
                 for name in ("hermes_session_at", "hermes_session_rt"))


def _bearer(token):
    return {"Authorization": "Bearer " + token}


def _renew(gated, refresh_token):
    """``/auth/native/refresh`` from a client holding nothing but *refresh_token*."""
    return gated.browser().post("/auth/native/refresh", json={"refresh_token": refresh_token})


def _refused(response):
    assert (response.status_code, response.json().get("error")) == (401, "session_expired")


def _refused_alone(gated, access, refresh):
    """Neither token works on its own: the access token as a bearer, the refresh token on the
    refresh route."""
    stolen = gated.browser()
    _refused(stolen.get("/api/auth/me", headers=_bearer(access)))
    _refused(stolen.post("/api/webapp/socket-ticket?socket=host-terminal", headers=_bearer(access)))
    _refused(stolen.post("/api/auth/ws-ticket", headers=_bearer(access)))
    _refused(_renew(gated, refresh))


def _restart(gated, monkeypatch, *, token=None):
    """What a new gateway process starts from: nothing in memory but the files in its home, and the
    browser credential stored there (*token* when the CLI replaced it while no gateway ran)."""
    from hermes_cli import webapp_sign_in

    webapp_sign_in._reset_for_tests()
    if token is not None:
        monkeypatch.setattr(gated.server, "_SESSION_TOKEN", token)


@pytest.mark.parametrize("sign_in", SIGN_INS)
def test_after_sign_out_a_logins_tokens_work_nowhere_on_the_webapp(gated, sign_out, sign_in):
    browser = gated.browser()
    SIGN_INS[sign_in](browser)
    access, refresh = _tokens(browser)
    sign_out()
    _refused_alone(gated, access, refresh)


@pytest.mark.parametrize("sign_in", SIGN_INS)
def test_a_logins_tokens_work_only_as_its_cookies(gated, sign_in):
    """Before any sign-out too: the tokens a browser login holds open nothing without its cookies, in
    the spelling a copy would take from the browser or from the cookie as stored."""
    browser = gated.browser()
    SIGN_INS[sign_in](browser)
    assert browser.get("/api/auth/me").status_code == 200
    access, refresh = _tokens(browser)
    _refused_alone(gated, access, refresh)
    stored = (browser.cookies.get(_cookie(browser, "hermes_session_at")),
              browser.cookies.get(_cookie(browser, "hermes_session_rt")))
    _refused_alone(gated, *stored)


def test_a_pair_renewed_before_sign_out_is_refused_after_it(gated, sign_out, clock):
    browser = gated.browser()
    _sign_in_with_password(browser)
    first = _tokens(browser)
    # The gate renews the browser's pair once its access token lapses.
    clock()
    browser.cookies.delete(_cookie(browser, "hermes_session_at"))
    assert browser.get("/api/auth/me").status_code == 200
    renewed = _tokens(browser)
    assert renewed[0] != first[0]
    # The refresh route renews nothing from a browser login, before sign-out either.
    _refused(_renew(gated, first[1]))
    sign_out()
    _refused_alone(gated, *renewed)


def test_the_refusal_survives_a_gateway_restart(gated, sign_out, monkeypatch):
    browser = gated.browser()
    _sign_in_with_password(browser)
    access, refresh = _tokens(browser)
    sign_out()
    _restart(gated, monkeypatch)
    _refused_alone(gated, access, refresh)


def test_a_credential_replaced_while_the_gateway_was_down_ends_its_logins_tokens(gated, monkeypatch, tmp_path):
    from gateway.webapp_listener import browser_session_token

    browser = gated.browser()
    _sign_in_with_password(browser)
    access, refresh = _tokens(browser)
    # The gateway stops, hermes webapp --sign-out replaces the stored credential, a gateway starts.
    _restart(gated, monkeypatch, token=browser_session_token(tmp_path / ".hermes", rotate=True))
    _signed_out(browser.get("/api/auth/me"))
    _refused_alone(gated, access, refresh)


def test_a_login_after_sign_out_works(gated, sign_out, clock):
    browser = gated.browser()
    _sign_in_with_password(browser)
    sign_out()
    clock()
    _sign_in_with_password(browser)
    assert browser.get("/api/auth/me").status_code == 200
    assert browser.post("/api/webapp/socket-ticket?socket=host-terminal").status_code == 200


def _native_app_sign_in(gated):
    """An RFC 8252 sign-in (the native Desktop's) on the Webapp's listener: authorize in the system
    browser, then redeem the loopback code for bearer tokens. No cookie login takes part."""
    import base64
    import hashlib
    from urllib.parse import parse_qs, urlsplit

    verifier = "native-app-verifier-" + "v" * 40
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    system_browser = gated.browser()
    started = system_browser.get("/auth/native/authorize", follow_redirects=False, params={
        "provider": "stub", "code_challenge": challenge, "code_challenge_method": "S256",
        "redirect_uri": "http://127.0.0.1:53999/cb", "state": "desk"})
    callback = parse_qs(urlsplit(started.headers["location"]).query)
    landed = system_browser.get("/auth/callback", follow_redirects=False,
                                params={"code": callback["code"][0], "state": callback["state"][0]})
    code = parse_qs(urlsplit(landed.headers["location"]).query)["code"][0]
    tokens = gated.browser().post("/auth/native/token", json={"code": code, "code_verifier": verifier})
    assert tokens.status_code == 200
    return tokens.json()["access_token"], tokens.json()["refresh_token"]


def test_a_native_app_signed_in_on_the_webapps_listener_stays_signed_in(gated, sign_out, monkeypatch):
    """Signing the browsers out leaves a native app's own sign-in alone, across a restart too, and the
    pair it renews keeps working."""
    access, refresh = _native_app_sign_in(gated)
    sign_out()
    _restart(gated, monkeypatch)
    assert gated.browser().get("/api/auth/me", headers=_bearer(access)).status_code == 200
    renewed = _renew(gated, refresh)
    assert renewed.status_code == 200
    assert gated.browser().get("/api/auth/me", headers=_bearer(renewed.json()["access_token"])).status_code == 200
    assert _renew(gated, renewed.json()["refresh_token"]).status_code == 200


def test_only_the_spelling_given_to_a_native_app_works(gated):
    """The providers accept other spellings of a token (the password provider and the test provider
    decode base64 leniently); the listener takes only the one it gave out."""
    access, refresh = _native_app_sign_in(gated)
    _refused_alone(gated, f'"{access}"', f'"{refresh}"')


def test_off_the_webapp_any_provider_token_works_on_its_own_as_on_main(gated, monkeypatch):
    monkeypatch.setattr(gated.server.app.state, "ui_surface", "dashboard")
    browser = gated.browser()
    _sign_in_with_password(browser)
    access, refresh = _tokens(browser)
    assert gated.browser().get("/api/auth/me", headers=_bearer(access)).status_code == 200
    assert _socket_close_code(gated.browser(), "/api/events?channel=tab-1", access) is None
    assert _renew(gated, refresh).status_code == 200


def _gateway_socket_with(gated, access):
    """The ``/api/ws`` upgrade from a client holding nothing but *access* (``?token=``): the close
    code the gate answers with, or None when it lets the socket open."""
    from urllib.parse import quote

    try:
        with gated.browser().websocket_connect(f"ws://hermes.lan:9120/api/ws?token={quote(access)}") as ws:
            return _close_code(ws, 0.5)
    except WebSocketDisconnect as closed:
        return closed.code


# Sockets on the Webapp's listener that take a provider token as ``?token=`` (the host terminal takes
# only its own ticket); ``/api/console`` runs the same gate as ``/api/pty``.
TOKEN_SOCKETS = ("/api/ws", "/api/events?channel=tab-1", "/api/pub?channel=tab-1", "/api/pty",
                 "/api/audio/speak-stream", "/api/audio/transcribe-stream")


def _socket_close_code(client, path, token):
    """The code a dial of *path* presenting ``?token=`` *token* is closed with, or None while it
    stays open."""
    joiner = "&" if "?" in path else "?"
    try:
        with client.websocket_connect(f"ws://hermes.lan:9120{path}{joiner}token={token}") as ws:
            return _close_code(ws, 0.5)
    except WebSocketDisconnect as closed:
        return closed.code


def test_a_logins_access_token_opens_no_socket_on_its_own(gated, monkeypatch):
    """The upgrade gate takes a provider token on its own on the same terms as the bearer header:
    on the Webapp only one given to a native app's sign-in, anywhere else any, as on main."""
    browser = gated.browser()
    _sign_in_with_password(browser)
    access, _ = _tokens(browser)
    native_access, _ = _native_app_sign_in(gated)
    assert _gateway_socket_with(gated, access) == 4401
    assert _gateway_socket_with(gated, native_access) is None
    monkeypatch.setattr(gated.server.app.state, "ui_surface", "dashboard")
    assert _gateway_socket_with(gated, access) is None


@pytest.mark.parametrize("path", TOKEN_SOCKETS)
def test_a_logins_access_token_opens_none_of_the_token_sockets(gated, sign_out, path):
    """A socket's ``?token=`` follows the bearer header's rule: a browser login's access token,
    copied out of its cookie, opens no socket on the Webapp's listener, before sign-out or after it."""
    browser = gated.browser()
    _sign_in_with_password(browser)
    access, _refresh = _tokens(browser)
    assert _socket_close_code(gated.browser(), path, access) == 4401
    sign_out()
    assert _socket_close_code(gated.browser(), path, access) == 4401


def test_a_native_apps_token_opens_a_socket_that_stays_open_across_sign_out(gated, sign_out):
    """A token the listener gave a native app's sign-in opens a socket as ``?token=``; signing the
    browsers out leaves that socket open and the token working."""
    access, _refresh = _native_app_sign_in(gated)
    with gated.browser().websocket_connect(f"ws://hermes.lan:9120/api/events?channel=desk&token={access}") as events:
        assert _close_code(events, 0.2) is None
        assert sign_out()["connections_closed"] == 0
        assert _close_code(events, 0.5) is None
    assert _socket_close_code(gated.browser(), "/api/events?channel=desk", access) is None


@pytest.mark.platforms("posix")
def test_the_record_of_native_sign_ins_is_user_only(gated, tmp_path):
    from hermes_cli.webapp_sign_in import RECORD_PATH

    _native_app_sign_in(gated)
    record = tmp_path / ".hermes" / RECORD_PATH
    assert record.stat().st_mode & 0o777 == 0o600
    assert record.parent.stat().st_mode & 0o777 == 0o700
