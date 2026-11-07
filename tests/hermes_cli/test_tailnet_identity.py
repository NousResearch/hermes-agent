"""``webapp.tailscale_logins`` (hermes_cli/tailnet_identity.py): a listed tailnet login that ``tailscale
serve`` proxied gets the ts.net Host accepted and the page's sign-in, like a launch link; it is the owner on
the API and sockets too (native clients), and nothing else reaches it (another login, another peer, a rebinding page,
another site, another local account)."""
import os
import types

import pytest
from starlette.requests import Request
from starlette.websockets import WebSocket

from hermes_cli import tailnet_identity as ti

TS_HOST = "mac-ultra.tailb4d38f.ts.net:8443"
OWNER = "owner@example.com"
SERVER = ("127.0.0.1", 9120)


@pytest.fixture
def logins(monkeypatch):
    """List *names*. The peer-account check (Linux only) has tests of its own below; here it passes, so
    these tests mean the same on every host."""
    def set_(*names):
        monkeypatch.setattr(ti, "allowed_logins", lambda: frozenset(n.lower() for n in names))
        monkeypatch.setattr(ti, "_peer_is_trusted", lambda peer, server: True)
    return set_


def _owner(**extra):
    return {"tailscale-user-login": OWNER, "host": TS_HOST, **extra}


def _state(*, surface: "str | None" = "webapp", auth_required=False):
    return types.SimpleNamespace(ui_surface=surface, auth_required=auth_required, bound_host="127.0.0.1")


async def _no_io(*_args):
    raise AssertionError("the checks under test never read or write the socket")


def _scope(kind, headers, peer=("127.0.0.1", 50000), state=None):
    scope = {"type": kind, "path": "/api/ws" if kind == "websocket" else "/", "query_string": b"",
             "headers": [(k.encode(), v.encode()) for k, v in headers.items()], "client": peer,
             "server": SERVER, "app": types.SimpleNamespace(state=state or _state())}
    if kind == "http":
        scope["method"] = "GET"
    return scope


# --- who counts as the owner -------------------------------------------------------------------------


@pytest.mark.parametrize("peer", [("127.0.0.1", 50000), ("::1", 50000), ("::ffff:127.0.0.1", 50000)])
def test_a_listed_login_proxied_by_serve_is_the_owner(logins, peer):
    logins(OWNER)
    assert ti.tailnet_login(_owner(**{"tailscale-user-login": "Owner@Example.COM"}), peer) == OWNER


@pytest.mark.parametrize("headers,peer", [
    (_owner(**{"tailscale-user-login": "kamel@example.com"}), ("127.0.0.1", 1)),  # not on the list
    (_owner(), ("100.64.0.7", 1)),                                                # a direct tailnet peer
    (_owner(), ("192.168.1.20", 1)),                                              # a LAN peer
    (_owner(), ("not-an-address", 1)),
    (_owner(), None),
    ({"host": TS_HOST}, ("127.0.0.1", 1)),                                        # no identity header
])
def test_nothing_else_is(logins, headers, peer):
    logins(OWNER)
    assert ti.tailnet_login(headers, peer) is None


def test_off_when_no_login_is_listed(logins):
    logins()
    assert ti.tailnet_login(_owner(), ("127.0.0.1", 1)) is None


@pytest.mark.parametrize("host,ok", [
    (TS_HOST, True),
    ("mac-ultra.tailb4d38f.ts.net", True),
    ("rebind.attacker.example:9120", False),  # DNS rebinding: the attacker's name resolving to loopback
    ("127.0.0.1:9120", False),
    ("localhost:9120", False),
    ("[::1]:9120", False),
    ("ts.net", False),
    (".ts.net", False),
    ("evil.ts.net.attacker.example", False),
    ("", False),
])
def test_only_a_tailnet_host(logins, host, ok):
    # A rebinding page is same-origin with itself and may set the header from script: its Host gives it away.
    logins(OWNER)
    headers = _owner(host=host, origin=f"http://{host}", **{"sec-fetch-site": "same-origin"})
    assert (ti.tailnet_login(headers, ("127.0.0.1", 1)) == OWNER) is ok


@pytest.mark.parametrize("extra,method,ok", [
    ({}, "POST", True),                                                              # no Origin: CLI, native app
    ({"origin": f"https://{TS_HOST}", "sec-fetch-site": "same-origin"}, "POST", True),  # the Webapp itself
    ({"sec-fetch-site": "none"}, "GET", True),                                       # typed URL, bookmark
    ({"origin": f"http://{TS_HOST}"}, "GET", True),
    ({"origin": "https://evil.example"}, "POST", False),                             # cross-site fetch or form
    ({"origin": "null"}, "POST", False),                                             # sandboxed frame
    ({"origin": "https://mac-ultra.tailb4d38f.ts.net:9999"}, "GET", False),          # another port's origin
    ({"sec-fetch-site": "cross-site", "sec-fetch-mode": "cors"}, "GET", False),      # cross-site fetch()
    ({"sec-fetch-site": "same-site", "sec-fetch-mode": "no-cors"}, "GET", False),    # a sibling machine's page
    ({"sec-fetch-site": "cross-site", "sec-fetch-mode": "websocket"}, "GET", False),  # cross-site socket
    ({"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"}, "GET", True),
    ({"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"}, "POST", False),
    ({"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "iframe"}, "GET", False),
])
def test_only_first_party_requests(logins, extra, method, ok):
    logins(OWNER)
    assert (ti.tailnet_login(_owner(**extra), ("127.0.0.1", 1), method) == OWNER) is ok


@pytest.mark.parametrize("uid,ok", [(0, True), (os.getuid(), True), (os.getuid() + 1, False), (None, False)])  # windows-footgun: ok — uid only feeds the Linux arm, given as data
def test_on_linux_the_peer_must_be_tailscaled_or_the_gateway_user(monkeypatch, uid, ok):
    from hermes_cli.web_routers import webapp

    seen = []
    monkeypatch.setattr(webapp, "_loopback_peer_uid", lambda client, server: seen.append((client, server)) or uid)
    assert ti._peer_is_trusted(("127.0.0.1", 50000), SERVER, "linux") is ok
    assert seen == [(("127.0.0.1", 50000), SERVER)]


def test_on_linux_a_connection_without_its_server_end_is_refused():
    assert ti._peer_is_trusted(("127.0.0.1", 50000), None, "linux") is False


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_elsewhere_the_peer_account_is_not_read(monkeypatch, platform):
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr(webapp, "_loopback_peer_uid", lambda client, server: pytest.fail("account read"))
    assert ti._peer_is_trusted(("127.0.0.1", 50000), SERVER, platform) is True


def test_the_login_needs_the_peer_check(monkeypatch):
    monkeypatch.setattr(ti, "allowed_logins", lambda: frozenset({OWNER}))
    monkeypatch.setattr(ti, "_peer_is_trusted", lambda peer, server: False)
    assert ti.tailnet_login(_owner(), ("127.0.0.1", 50000), "GET", SERVER) is None


@pytest.mark.platforms("linux")
def test_on_linux_a_real_loopback_peer_of_this_user_passes():
    # Not faked: a real connection, whose client end /proc/net/tcp lists under this process's uid.
    import socket

    with socket.create_server(("127.0.0.1", 0)) as listener:
        server = listener.getsockname()[:2]
        with socket.create_connection(server) as client:
            accepted, _ = listener.accept()
            with accepted:
                assert ti._peer_is_trusted(client.getsockname()[:2], server) is True


@pytest.mark.parametrize("raw,expected", [
    (None, frozenset()),
    ([], frozenset()),
    ("Owner@Example.com", frozenset({OWNER})),
    ([" Owner@Example.com ", "", "kamel@example.com"], frozenset({OWNER, "kamel@example.com"})),
    ({"owner": OWNER}, frozenset()),  # not a list: off rather than guessed
])
def test_the_list_is_read_from_config(monkeypatch, raw, expected):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config", lambda: {"webapp": {"tailscale_logins": raw}})
    ti.allowed_logins.cache_clear()
    try:
        assert ti.allowed_logins() == expected
    finally:
        ti.allowed_logins.cache_clear()


def test_an_unreadable_config_turns_it_off(monkeypatch):
    from hermes_cli import config

    def broken():
        raise OSError("unreadable")

    monkeypatch.setattr(config, "load_config", broken)
    ti.allowed_logins.cache_clear()
    try:
        assert ti.allowed_logins() == frozenset()
    finally:
        ti.allowed_logins.cache_clear()


@pytest.mark.parametrize("state,ok", [
    (_state(), True),
    (_state(auth_required=True), False),                  # the gated Webapp keeps its sign-in
    (_state(surface=None), False),                        # the standalone dashboard is not concerned
    (types.SimpleNamespace(ui_surface="webapp"), False),  # no gate decision recorded: off
])
def test_only_the_webapp_without_the_sign_in_gate(logins, state, ok):
    logins(OWNER)
    assert (ti.connection_login(Request(_scope("http", _owner(), state=state))) == OWNER) is ok
    assert (ti.connection_login(WebSocket(_scope("websocket", _owner(), state=state), _no_io, _no_io)) == OWNER) is ok


# --- what the owner gets -----------------------------------------------------------------------------


@pytest.fixture
def webapp_server(monkeypatch):
    from hermes_cli import web_server

    for name, value in (("bound_host", "127.0.0.1"), ("auth_required", False), ("ui_surface", "webapp")):
        monkeypatch.setattr(web_server.app.state, name, value, raising=False)
    return web_server


def test_the_ts_net_host_is_accepted_for_the_owner_only(logins, webapp_server):
    from fastapi.testclient import TestClient

    logins(OWNER)
    client = TestClient(webapp_server.app, client=("127.0.0.1", 50000))
    assert client.get("/api/status", headers={"host": TS_HOST}).status_code == 400
    assert client.get("/api/status", headers=_owner(**{"tailscale-user-login": "kamel@example.com"})).status_code == 400
    assert client.get("/api/status", headers=_owner(host="rebind.attacker.example")).status_code == 400
    assert client.get("/api/status", headers=_owner()).status_code != 400


def test_the_owner_reaches_the_api_without_a_token(logins, webapp_server):
    # Native clients (the iOS app) have no page to read the token from: the listed login is the owner,
    # as the page that hands that same request the token already treats it.
    from fastapi.testclient import TestClient

    logins(OWNER)
    client = TestClient(webapp_server.app, client=("127.0.0.1", 50000))
    assert client.get("/api/sessions", headers=_owner()).status_code != 401
    assert client.get("/api/sessions", headers={"host": "127.0.0.1:9120"}).status_code == 401


def test_the_owner_opens_a_socket_and_no_one_else(logins, webapp_server):
    from hermes_cli.web_server_chat import _ws_auth_reason

    logins(OWNER)
    def reason(headers):
        return _ws_auth_reason(WebSocket(_scope("websocket", headers, state=webapp_server.app.state),
                                         _no_io, _no_io))[0]
    assert reason(_owner(origin=f"https://{TS_HOST}")) is None
    assert reason(_owner(**{"tailscale-user-login": "kamel@example.com", "origin": f"https://{TS_HOST}"})) is not None
    assert reason(_owner(origin="https://evil.example")) is not None


@pytest.mark.parametrize("headers,ok", [
    (_owner(origin=f"https://{TS_HOST}"), True),
    ({"host": TS_HOST, "origin": f"https://{TS_HOST}"}, False),
    (_owner(origin="https://evil.example"), False),
    (_owner(host="rebind.attacker.example", origin="http://rebind.attacker.example"), False),
])
def test_a_socket_upgrade_passes_the_host_check_for_the_owner_only(logins, headers, ok):
    from hermes_cli.web_server_boundary import ws_host_origin_reason

    logins(OWNER)
    state = _state()
    ws = WebSocket(_scope("websocket", headers, state=state), _no_io, _no_io)
    assert (ws_host_origin_reason(ws, state) is None) is ok


def _spa(monkeypatch, tmp_path, *, surface="webapp", auth_required=False):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_cli import web_server as server
    from hermes_cli.web_server_dashboard import mount_spa

    (tmp_path / "index.html").write_text("<html><head></head><body>x</body></html>")
    app = FastAPI()
    app.state.ui_surface = surface
    app.state.auth_required = auth_required
    app.state.web_dist = tmp_path
    monkeypatch.setattr(server, "app", app)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "tok-123")
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    mount_spa(app)
    return TestClient(app, client=("127.0.0.1", 50000))


def test_the_page_signs_the_owner_in(logins, monkeypatch, tmp_path):
    logins(OWNER)
    page = _spa(monkeypatch, tmp_path).get("/", headers=_owner())
    assert "hermes.webapp.session.v1:" in page.text and '"tok-123"' in page.text
    assert "no-store" in page.headers.get("cache-control", "")


@pytest.mark.parametrize("headers", [
    {"host": TS_HOST},
    _owner(**{"tailscale-user-login": "kamel@example.com"}),
    _owner(host="rebind.attacker.example", origin="http://rebind.attacker.example"),
    _owner(**{"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "iframe"}),
])
def test_no_other_page_gets_the_credential(logins, monkeypatch, tmp_path, headers):
    logins(OWNER)
    page = _spa(monkeypatch, tmp_path).get("/", headers=headers).text
    assert "tok-123" not in page and "hermes.webapp.session.v1:" not in page


@pytest.mark.parametrize("surface,auth_required", [(None, False), ("webapp", True)])
def test_the_dashboard_and_the_gated_webapp_never_inject_it(logins, monkeypatch, tmp_path, surface, auth_required):
    logins(OWNER)
    page = _spa(monkeypatch, tmp_path, surface=surface, auth_required=auth_required).get("/", headers=_owner()).text
    assert "hermes.webapp.session.v1:" not in page
