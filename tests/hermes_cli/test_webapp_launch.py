"""The private launch ticket crosses a browser launcher's argv, so only the server user may spend it.

Minting the link and opening the browser belong to whatever launches the Webapp; these tests cover
the handoff route the ticket is spent at (``/webapp/window-session``).
"""
import os

import pytest


@pytest.mark.platforms("linux")
def test_launch_ticket_is_spent_only_over_the_server_users_own_connection(monkeypatch):
    """Linux publishes every process's argv, so another local user can read a launch ticket
    out of the browser launcher before the browser spends it. That user's connection is
    refused, and the refusal does not burn the ticket for the operator's browser."""
    from fastapi.testclient import TestClient
    from hermes_cli import web_server as server
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server.app.state, "webapp_window_tickets", {}, raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "a" * 43)
    client = TestClient(server.app, base_url="http://127.0.0.1")
    headers = {"Origin": "http://127.0.0.1", "X-Hermes-Window-Ticket": webapp.mint_launch_ticket(server.app.state)}

    own_uid = os.getuid()  # windows-footgun: ok — platforms("linux") test
    monkeypatch.setattr(webapp, "_loopback_peer_uid", lambda client, server: own_uid + 1)
    stolen = client.post("/webapp/window-session", headers=headers)
    assert stolen.status_code == 403 and "a" * 43 not in stolen.text
    monkeypatch.setattr(webapp, "_loopback_peer_uid", lambda client, server: own_uid)
    granted = client.post("/webapp/window-session", headers=headers)
    assert granted.status_code == 200 and granted.json()["token"] == "a" * 43


@pytest.mark.platforms("macos")
def test_macos_launch_handoff_fails_closed_without_consuming_child_window_authority(monkeypatch):
    """An argv launch ticket is unusable until macOS has an authenticated OS handoff.

    This native platform contract makes no claim about cross-account argv visibility.
    Authenticated child-window tickets never pass through a launcher and still work.
    """
    from fastapi.testclient import TestClient
    from hermes_cli import web_server as server
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server.app.state, "webapp_window_tickets", {}, raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "a" * 43)
    client = TestClient(server.app, base_url="http://127.0.0.1")
    try:
        launch = webapp.mint_launch_ticket(server.app.state)
        headers = {"Origin": "http://127.0.0.1", "X-Hermes-Window-Ticket": launch}
        for _ in range(2):
            denied = client.post("/webapp/window-session", headers=headers)
            assert denied.status_code == 403 and server._SESSION_TOKEN not in denied.text
            assert launch in server.app.state.webapp_window_tickets

        issued = client.post("/api/webapp/window-ticket", headers={
            server._SESSION_HEADER_NAME: server._SESSION_TOKEN,
        })
        assert issued.status_code == 200
        headers["X-Hermes-Window-Ticket"] = issued.json()["ticket"]
        assert client.post("/webapp/window-session", headers=headers).json() == {"token": server._SESSION_TOKEN}
        assert client.post("/webapp/window-session", headers=headers).status_code == 403
    finally:
        client.close()


def test_a_link_for_the_user_to_open_by_hand_spends_once_without_a_peer_check(monkeypatch):
    """Where nothing identifies the connection's owner (macOS, the BSDs, WSL), ``hermes webapp`` prints a
    link for the user to open by hand. On every host its ticket spends without consulting the peer
    check, once and within 60 seconds, as a window ticket does. (That an argv launch ticket never
    spends there is the platforms("macos") test above.)"""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient
    from hermes_cli import web_server as server
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server.app.state, "webapp_window_tickets", {}, raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "a" * 43)
    def peer_check(request):
        raise AssertionError("a printed launch ticket consulted the peer check")

    monkeypatch.setattr(webapp, "_peer_is_server_user", peer_check)
    client = TestClient(server.app, base_url="http://127.0.0.1")

    def redeem(ticket):
        return client.post("/webapp/window-session", headers={"Origin": "http://127.0.0.1",
                                                              "X-Hermes-Window-Ticket": ticket})
    try:
        printed = webapp.mint_printed_launch_ticket(server.app.state)
        assert redeem(printed).json() == {"token": "a" * 43}
        assert redeem(printed).status_code == 403
        late = webapp.mint_printed_launch_ticket(server.app.state)
        now = webapp.time.monotonic()
        monkeypatch.setattr(webapp, "time", SimpleNamespace(monotonic=lambda: now + webapp.LAUNCH_TICKET_TTL_SECONDS + 1))
        assert redeem(late).status_code == 403
    finally:
        client.close()


def _loopback_request():
    from types import SimpleNamespace
    return SimpleNamespace(client=SimpleNamespace(host="127.0.0.1", port=50123),
                           scope={"server": ("127.0.0.1", 9120)})


@pytest.mark.parametrize("platform", ["darwin", "freebsd14", "openbsd7", "sunos5"])
def test_where_no_peer_check_exists_a_launch_ticket_is_never_spent(platform):
    """macOS and the other Unixes show every process's argv to all local users by default and Hermes
    cannot tell who owns a loopback connection there. So the launcher prints a link for the user to
    open by hand rather than passing a launch ticket to a browser launcher, and a launch ticket is
    never spent. The platform is passed in as data; the marked tests below run each host's own path."""
    from gateway import webapp_listener
    from hermes_cli.web_routers import webapp

    assert webapp._peer_is_server_user(_loopback_request(), platform=platform) is False
    assert webapp_listener._one_use_links_work(platform=platform) is False


def test_windows_spends_a_launch_ticket_without_a_peer_check():
    """Windows does not show a process's command line to other standard users (reading it needs a
    handle the process's DACL grants only its own user, SYSTEM and elevated administrators), so the
    ticket in the browser launcher's argv stays the user's own, as the stored credential does. The
    platform is passed in as data; the platforms("windows") test below runs the real path."""
    from gateway import webapp_listener
    from hermes_cli.web_routers import webapp

    assert webapp._peer_is_server_user(_loopback_request(), platform="win32") is True
    assert webapp_listener._one_use_links_work(platform="win32") is True


@pytest.mark.platforms("windows")
def test_on_windows_the_launcher_opens_the_launch_link_and_its_browser_spends_it(tmp_path, monkeypatch):
    """On Windows ``hermes webapp`` gets a launch link to open itself (not marked private), and the
    browser it opens spends the ticket without a peer check."""
    client, link = _launch_link(tmp_path, monkeypatch)
    assert link["private"] is False
    assert _spend(client, link).status_code == 200


@pytest.mark.platforms("macos")
def test_on_macos_the_launch_link_is_printed_and_spends_without_a_peer_check(tmp_path, monkeypatch):
    """On macOS nothing identifies who owns the browser's connection: ``hermes webapp`` gets a link
    marked private, to print for the user to open by hand, and it spends without the peer check."""
    client, link = _launch_link(tmp_path, monkeypatch)
    assert link["private"] is True
    assert _spend(client, link).status_code == 200


@pytest.mark.platforms("linux")
def test_on_linux_the_peer_check_names_the_owner_of_a_real_loopback_connection():
    """Linux reads who owns the browser's end of a loopback connection from ``/proc/net/tcp*``: a
    connection this process holds passes the peer check, and one to a port nobody listens on (no
    such connection) does not."""
    import socket
    from types import SimpleNamespace

    from hermes_cli.web_routers import webapp

    with socket.create_server(("127.0.0.1", 0)) as listener, \
            socket.create_connection(listener.getsockname()) as browser:
        served, _ = listener.accept()
        with served:
            host, port = browser.getsockname()
            own = SimpleNamespace(client=SimpleNamespace(host=host, port=port),
                                  scope={"server": listener.getsockname()})
            assert webapp._peer_is_server_user(own) is True
            unknown = SimpleNamespace(client=SimpleNamespace(host=host, port=port), scope={"server": (host, 1)})
            assert webapp._peer_is_server_user(unknown) is False


@pytest.mark.parametrize("page", ["/webapp/launch", "/webapp/window"])
def test_a_handoff_page_keeps_the_credential_out_of_every_url(monkeypatch, page):
    """The browser records every address a tab navigates to in its history (and syncs it), and the
    credential a handoff page receives lasts until ``hermes webapp --sign-out``. So the launch and
    child-window pages put it in this tab's sessionStorage, under the key the Webapp reads at boot,
    and then replace themselves with the Webapp's own address: no URL they navigate to carries it.
    Each page is uncached and sends no Referer."""
    from fastapi.testclient import TestClient
    from hermes_cli import web_server as server

    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "a" * 43)
    served = TestClient(server.app, base_url="http://127.0.0.1").get(page)
    assert served.status_code == 200
    assert "hermes-session" not in served.text, "the page navigates with the credential in the URL"
    assert "sessionStorage.setItem(`hermes.webapp.session.v1:${JSON.stringify([location.origin, basePath])}`, token)" \
        in served.text
    assert "a" * 43 not in served.text
    assert served.headers["cache-control"] == "no-store"
    assert served.headers["referrer-policy"] == "no-referrer"


def _launch_link(tmp_path, monkeypatch):
    """The loopback Webapp of a gateway serving *tmp_path*, and the ``webapp-launch`` answer its control
    socket gives ``hermes webapp``."""
    import json
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from gateway.control_socket import GatewayControlServer
    from gateway.webapp_listener import register_control_verbs
    from hermes_cli import web_server as server

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server.app.state, "webapp_window_tickets", {}, raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "a" * 43)
    runner = SimpleNamespace(session_control_server=GatewayControlServer(home),
                             webapp_status={"state": "serving", "url": "http://127.0.0.1:9120", "reason": None},
                             session_api=SimpleNamespace(app=SimpleNamespace(web=server)))
    register_control_verbs(runner, home)
    raw = json.dumps({"protocol": 1, "verb": "webapp-launch", "id": 1, "params": {}}).encode()
    answer = json.loads(runner.session_control_server.handle_request_line(raw, "same-user"))
    assert answer["ok"], answer
    return TestClient(server.app, base_url="http://127.0.0.1"), answer["result"]


def _spend(client, link):
    from urllib.parse import parse_qs, urlsplit

    ticket = parse_qs(urlsplit(link["url"]).fragment)["ticket"][0]
    return client.post("/webapp/window-session", headers={"Origin": "http://127.0.0.1", "X-Hermes-Window-Ticket": ticket})


@pytest.mark.platforms("linux")
def test_under_wsl_the_launch_link_is_printed_and_the_windows_browser_spends_it(tmp_path, monkeypatch):
    """Under WSL2 the Windows browser reaches the gateway's loopback port through WSL's localhost relay,
    so inside Linux its connection belongs to root (uid 0, measured live), and any Windows account can
    reach that port. Nothing identifies the user's own browser, so, as on macOS, ``hermes webapp`` gets
    a link marked private, to print for the user to open by hand, and that link spends without the
    peer check instead of being refused every time."""
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr("hermes_constants.is_wsl", lambda: True)
    monkeypatch.setattr(webapp, "_loopback_peer_uid", lambda client, server: 0)  # WSL's localhost relay
    client, link = _launch_link(tmp_path, monkeypatch)
    assert link["private"] is True
    spent = _spend(client, link)
    assert spent.status_code == 200, f"the WSL user's own Windows browser cannot spend the link: {spent.text}"


@pytest.mark.platforms("linux")
def test_native_linux_keeps_the_peer_check_on_the_launch_link(tmp_path, monkeypatch):
    """Outside WSL, Linux names the owner of the browser's loopback connection: ``hermes webapp`` opens
    the link itself, and a connection another user owns cannot spend it."""
    import os

    from hermes_cli.web_routers import webapp

    monkeypatch.setattr("hermes_constants.is_wsl", lambda: False)
    monkeypatch.setattr(webapp, "_loopback_peer_uid", lambda client, server: os.getuid() + 1)  # windows-footgun: ok — platforms("linux") test
    client, link = _launch_link(tmp_path, monkeypatch)
    assert link["private"] is False
    assert _spend(client, link).status_code == 403
